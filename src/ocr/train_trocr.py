import argparse
import os
from pathlib import Path
from typing import Any

import jiwer
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from src.utils.config import load_yaml_config

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_TARGET_LENGTH = 10
DEFAULT_WEIGHT_DECAY = 0.01


class MeterOCRDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels_csv: str | Path,
        processor: TrOCRProcessor,
        image_column: str = "filename",
        text_column: str = "text",
        max_target_length: int = 16,
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.image_column = image_column
        self.text_column = text_column
        self.max_target_length = max_target_length

        self.df = pd.read_csv(labels_csv, dtype={text_column: str})

        if image_column not in self.df.columns:
            raise ValueError(f"Column '{image_column}' not found in {labels_csv}")

        if text_column not in self.df.columns:
            raise ValueError(f"Column '{text_column}' not found in {labels_csv}")

        self.df[text_column] = self.df[text_column].astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_name = row[self.image_column]
        text = row[self.text_column]

        image_path = self.images_dir / image_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids

        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR model for meter OCR.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)

    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if "train" not in config:
        config["train"] = {}

    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate

    if args.max_target_length is not None:
        config["train"]["max_target_length"] = args.max_target_length

    if args.model_name is not None:
        config["model"]["name"] = args.model_name

    if args.run_name is not None:
        config["experiment"]["run_name"] = args.run_name

    return config


def apply_default_train_values(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config["train"]

    train_cfg.setdefault("epochs", DEFAULT_EPOCHS)
    train_cfg.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    train_cfg.setdefault("learning_rate", DEFAULT_LEARNING_RATE)
    train_cfg.setdefault("max_target_length", DEFAULT_MAX_TARGET_LENGTH)
    train_cfg.setdefault("weight_decay", DEFAULT_WEIGHT_DECAY)

    return config


def load_model_and_processor(model_name: str, max_target_length: int):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id

    model.generation_config.max_length = max_target_length
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    return processor, model


def character_error_rate(true_text: str, predicted_text: str) -> float:
    return jiwer.cer(true_text, predicted_text)


def word_error_rate(true_text: str, predicted_text: str) -> float:
    return jiwer.wer(true_text, predicted_text)


def digit_accuracy(true_text: str, predicted_text: str) -> float:
    max_len = max(len(true_text), len(predicted_text))

    if max_len == 0:
        return 1.0

    correct = 0

    for i in range(max_len):
        true_char = true_text[i] if i < len(true_text) else None
        predicted_char = predicted_text[i] if i < len(predicted_text) else None

        if true_char == predicted_char:
            correct += 1

    return correct / max_len


def numeric_error(true_text: str, predicted_text: str) -> float | None:
    try:
        true_number = float(true_text)
        predicted_number = float(predicted_text)
    except ValueError:
        return None

    return abs(true_number - predicted_number)


def compute_ocr_metrics_from_texts(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    predictions = [str(prediction).strip() for prediction in predictions]
    references = [str(reference).strip() for reference in references]

    exact_match_values = []
    cer_values = []
    wer_values = []
    digit_accuracy_values = []
    numeric_error_values = []

    for true_text, predicted_text in zip(references, predictions, strict=False):
        exact_match_values.append(true_text == predicted_text)
        cer_values.append(character_error_rate(true_text, predicted_text))
        wer_values.append(word_error_rate(true_text, predicted_text))
        digit_accuracy_values.append(digit_accuracy(true_text, predicted_text))

        num_error = numeric_error(true_text, predicted_text)
        if num_error is not None:
            numeric_error_values.append(num_error)

    metrics = {
        "exact_match_accuracy": float(np.mean(exact_match_values)),
        "mean_cer": float(np.mean(cer_values)),
        "mean_wer": float(np.mean(wer_values)),
        "mean_digit_accuracy": float(np.mean(digit_accuracy_values)),
    }

    if numeric_error_values:
        metrics["mean_numeric_error"] = float(np.mean(numeric_error_values))
        metrics["median_numeric_error"] = float(np.median(numeric_error_values))

    return metrics


def build_compute_metrics(processor: TrOCRProcessor):
    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(
            labels != -100,
            labels,
            processor.tokenizer.pad_token_id,
        )

        decoded_predictions = processor.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        decoded_labels = processor.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        return compute_ocr_metrics_from_texts(
            predictions=decoded_predictions,
            references=decoded_labels,
        )

    return compute_metrics


def log_params(config: dict[str, Any]) -> None:
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    mlflow.log_params(
        {
            "model_name": model_cfg["name"],
            "train_images": data_cfg["train_images"],
            "train_labels": data_cfg["train_labels"],
            "val_images": data_cfg["val_images"],
            "val_labels": data_cfg["val_labels"],
            "image_column": data_cfg["image_column"],
            "text_column": data_cfg["text_column"],
            "output_dir": train_cfg["output_dir"],
            "epochs": train_cfg["epochs"],
            "batch_size": train_cfg["batch_size"],
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
            "max_target_length": train_cfg["max_target_length"],
            "fp16": torch.cuda.is_available(),
        }
    )


def log_trainer_history_to_mlflow(trainer: Seq2SeqTrainer) -> None:
    for log in trainer.state.log_history:
        step = log.get("step")

        if step is None:
            continue

        for metric_name, metric_value in log.items():
            if metric_name in {"step", "epoch"}:
                continue

            if isinstance(metric_value, int | float):
                mlflow.log_metric(metric_name, metric_value, step=step)


def log_loss_plot(trainer: Seq2SeqTrainer, output_dir: Path) -> None:
    train_epochs = []
    train_losses = []

    eval_epochs = []
    eval_losses = []

    for log in trainer.state.log_history:
        if "loss" in log and "epoch" in log:
            train_epochs.append(log["epoch"])
            train_losses.append(log["loss"])

        if "eval_loss" in log and "epoch" in log:
            eval_epochs.append(log["epoch"])
            eval_losses.append(log["eval_loss"])

    if not train_losses and not eval_losses:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    if train_losses:
        plt.plot(train_epochs, train_losses, marker="o", label="Train loss")

    if eval_losses:
        plt.plot(eval_epochs, eval_losses, marker="o", label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and validation loss")
    plt.legend()
    plt.grid(True)

    plot_path = output_dir / "loss_curve.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(plot_path), artifact_path="plots")


def log_model_artifacts(output_dir: Path) -> None:
    artifacts = [
        output_dir / "config.json",
        output_dir / "generation_config.json",
        output_dir / "model.safetensors",
        output_dir / "pytorch_model.bin",
        output_dir / "preprocessor_config.json",
        output_dir / "tokenizer_config.json",
        output_dir / "vocab.json",
        output_dir / "merges.txt",
        output_dir / "special_tokens_map.json",
        output_dir / "training_args.bin",
        output_dir / "trainer_state.json",
    ]

    for artifact in artifacts:
        if artifact.exists():
            mlflow.log_artifact(str(artifact), artifact_path="model")

    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            mlflow.log_artifacts(
                str(checkpoint_dir),
                artifact_path=f"checkpoints/{checkpoint_dir.name}",
            )


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)
    config = apply_overrides(config, args)
    config = apply_default_train_values(config)

    experiment_cfg = config["experiment"]
    mlflow_cfg = config["mlflow"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["train"]

    os.environ["MLFLOW_TRACKING_URI"] = mlflow_cfg["tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_cfg["name"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(experiment_cfg["name"])

    with mlflow.start_run(run_name=experiment_cfg["run_name"]):
        mlflow.log_artifact(args.config, artifact_path="config")
        log_params(config)

        processor, model = load_model_and_processor(
            model_cfg["name"],
            max_target_length=train_cfg["max_target_length"],
        )

        train_dataset = MeterOCRDataset(
            images_dir=data_cfg["train_images"],
            labels_csv=data_cfg["train_labels"],
            processor=processor,
            image_column=data_cfg["image_column"],
            text_column=data_cfg["text_column"],
            max_target_length=train_cfg["max_target_length"],
        )

        val_dataset = MeterOCRDataset(
            images_dir=data_cfg["val_images"],
            labels_csv=data_cfg["val_labels"],
            processor=processor,
            image_column=data_cfg["image_column"],
            text_column=data_cfg["text_column"],
            max_target_length=train_cfg["max_target_length"],
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            per_device_eval_batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            predict_with_generate=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            processing_class=processor,
            compute_metrics=build_compute_metrics(processor),
        )

        train_result = trainer.train()

        output_dir = Path(train_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        train_metrics = train_result.metrics
        mlflow.log_metrics(train_metrics)

        eval_metrics = trainer.evaluate()
        mlflow.log_metrics(eval_metrics)

        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        log_trainer_history_to_mlflow(trainer)
        log_loss_plot(trainer, output_dir)
        log_model_artifacts(output_dir)

        print(f"Training finished. Model saved to: {output_dir}")
        print("Training metrics, validation OCR metrics, and loss plot logged to MLflow.")


if __name__ == "__main__":
    main()
