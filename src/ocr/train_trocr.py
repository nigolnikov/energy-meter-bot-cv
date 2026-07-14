import argparse
import json
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import jiwer
import matplotlib

matplotlib.use("Agg")

import albumentations as A
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
    set_seed,
)

from src.ocr.augmentations import build_train_augmentations
from src.ocr.reporting import REPORT_DIRNAME, log_evaluation_report, plot_augmentation_samples
from src.ocr.text import decode_label, encode_label
from src.utils.config import load_yaml_config

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-5

DEFAULT_MAX_TARGET_LENGTH = 13
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42

PLOTS_DIRNAME = "plots"


class MeterOCRDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels_csv: str | Path,
        processor: TrOCRProcessor,
        image_column: str = "filename",
        text_column: str = "text",
        max_target_length: int = 16,
        augmentations: A.Compose | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.image_column = image_column
        self.text_column = text_column
        self.max_target_length = max_target_length
        self.augmentations = augmentations

        self.df = pd.read_csv(labels_csv, dtype={text_column: str})

        if image_column not in self.df.columns:
            raise ValueError(f"Column '{image_column}' not found in {labels_csv}")

        if text_column not in self.df.columns:
            raise ValueError(f"Column '{text_column}' not found in {labels_csv}")

        self.df[text_column] = self.df[text_column].astype(str).str.strip()

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

        if self.augmentations is not None:
            image_np = np.array(image)
            image_np = self.augmentations(image=image_np)["image"]
            image = Image.fromarray(image_np)

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            encode_label(text),
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
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Force a local-only run even if the config has an mlflow section.",
    )

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

    if args.length_penalty is not None:
        config["train"]["length_penalty"] = args.length_penalty

    if args.num_beams is not None:
        config["train"]["num_beams"] = args.num_beams

    if args.model_name is not None:
        config["model"]["name"] = args.model_name

    if args.run_name is not None:
        config.setdefault("experiment", {})["run_name"] = args.run_name

    if args.seed is not None:
        config["train"]["seed"] = args.seed

    if args.no_mlflow:
        config.pop("mlflow", None)

    return config


def apply_default_train_values(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config["train"]

    train_cfg.setdefault("epochs", DEFAULT_EPOCHS)
    train_cfg.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    train_cfg.setdefault("learning_rate", DEFAULT_LEARNING_RATE)
    train_cfg.setdefault("max_target_length", DEFAULT_MAX_TARGET_LENGTH)
    train_cfg.setdefault("weight_decay", DEFAULT_WEIGHT_DECAY)
    train_cfg.setdefault("length_penalty", 1.0)
    train_cfg.setdefault("num_beams", 4)
    train_cfg.setdefault("seed", DEFAULT_SEED)

    # train_cfg.setdefault("warmup_ratio", 0.1)
    # train_cfg.setdefault("lr_scheduler_type", "cosine")

    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_label_encoding(processor: TrOCRProcessor, dataset: Dataset) -> None:
    tokenizer = processor.tokenizer
    texts = dataset.df[dataset.text_column].tolist()

    lengths = np.array([len(tokenizer(encode_label(text)).input_ids) for text in texts])
    truncated = int((lengths > dataset.max_target_length).sum())

    sample = texts[0]
    tokens = tokenizer.tokenize(encode_label(sample))

    print(f"Label encoding check on {len(texts)} labels:")
    print(f"  {sample!r} -> {encode_label(sample)!r} -> {tokens}")
    print(f"  tokens per label: max={lengths.max()}, max_target_length={dataset.max_target_length}")

    expected = len(sample) + 2
    actual = len(tokenizer(encode_label(sample)).input_ids)

    if actual != expected:
        print(
            f"  WARNING: {sample!r} has {len(sample)} chars but encodes to {actual} tokens "
            f"(expected {expected}). The tokenizer is still merging glyphs."
        )

    if truncated:
        raise ValueError(
            f"{truncated} labels exceed max_target_length={dataset.max_target_length} "
            f"(longest is {lengths.max()}). Raise it, or ground truth is being cut off."
        )


def load_model_and_processor(
    model_name: str,
    max_target_length: int,
    length_penalty: float,
    num_beams: int,
):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, use_safetensors=True)

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
    model.generation_config.length_penalty = length_penalty
    model.generation_config.num_beams = num_beams

    return processor, model


def character_error_rate(true_text: str, predicted_text: str) -> float:
    return jiwer.cer(true_text, predicted_text)


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
    exact_match_values = []
    cer_values = []
    digit_accuracy_values = []
    digits_only_match_values = []
    dot_match_values = []
    numeric_error_values = []

    parse_failures = 0

    for true_text, predicted_text in zip(references, predictions, strict=False):
        exact_match_values.append(true_text == predicted_text)
        cer_values.append(character_error_rate(true_text, predicted_text))
        digit_accuracy_values.append(digit_accuracy(true_text, predicted_text))

        digits_only_match_values.append(
            true_text.replace(".", "") == predicted_text.replace(".", "")
        )
        dot_match_values.append(true_text.find(".") == predicted_text.find("."))

        num_error = numeric_error(true_text, predicted_text)

        if num_error is not None:
            numeric_error_values.append(num_error)
        else:
            parse_failures += 1

    metrics = {
        "exact_match_accuracy": float(np.mean(exact_match_values)),
        "mean_cer": float(np.mean(cer_values)),
        "mean_digit_accuracy": float(np.mean(digit_accuracy_values)),
        "digits_only_accuracy": float(np.mean(digits_only_match_values)),
        "dot_position_accuracy": float(np.mean(dot_match_values)),
        "invalid_format_rate": parse_failures / max(len(predictions), 1),
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

        decoded_predictions = [
            decode_label(text)
            for text in processor.batch_decode(predictions, skip_special_tokens=True)
        ]

        decoded_labels = [
            decode_label(text) for text in processor.batch_decode(labels, skip_special_tokens=True)
        ]

        return compute_ocr_metrics_from_texts(
            predictions=decoded_predictions,
            references=decoded_labels,
        )

    return compute_metrics


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def build_params(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    experiment_cfg = config.get("experiment", {})

    return {
        "run_name": experiment_cfg.get("run_name"),
        "experiment_name": experiment_cfg.get("name"),
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
        "length_penalty": train_cfg["length_penalty"],
        "num_beams": train_cfg["num_beams"],
        "label_encoding": "char_level",
        "fp16": torch.cuda.is_available(),
        "seed": train_cfg["seed"],
        # "warmup_ratio": train_cfg["warmup_ratio"],
        # "lr_scheduler_type": train_cfg["lr_scheduler_type"],
    }


def save_run_inputs(config: dict[str, Any], config_path: str, plots_dir: Path) -> None:
    save_json(build_params(config), plots_dir / "params.json")
    shutil.copy(config_path, plots_dir / f"config{Path(config_path).suffix}")


def mlflow_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("mlflow", {}).get("tracking_uri"))


def start_mlflow_run(config: dict[str, Any]):
    if not mlflow_enabled(config):
        return nullcontext()

    mlflow_cfg = config["mlflow"]
    experiment_cfg = config.get("experiment", {})

    os.environ["MLFLOW_TRACKING_URI"] = mlflow_cfg["tracking_uri"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    if experiment_cfg.get("name"):
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_cfg["name"]
        mlflow.set_experiment(experiment_cfg["name"])

    return mlflow.start_run(run_name=experiment_cfg.get("run_name"))


def log_trainer_history_to_mlflow(trainer: Seq2SeqTrainer) -> None:
    for log in trainer.state.log_history:
        step = log.get("step")

        if step is None:
            continue

        for name, value in log.items():
            if name in {"step", "epoch"}:
                continue

            if isinstance(value, int | float):
                mlflow.log_metric(name, value, step=step)


def log_numeric_metrics_to_mlflow(metrics: dict[str, Any]) -> None:
    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})


def save_trainer_history(trainer: Seq2SeqTrainer, plots_dir: Path) -> None:
    save_json(trainer.state.log_history, plots_dir / "training_history.json")

    eval_rows = [log for log in trainer.state.log_history if "eval_loss" in log]

    if eval_rows:
        pd.DataFrame(eval_rows).to_csv(plots_dir / "eval_history.csv", index=False)


def save_loss_plot(trainer: Seq2SeqTrainer, plots_dir: Path) -> None:
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

    plots_dir.mkdir(parents=True, exist_ok=True)

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

    plt.savefig(plots_dir / "loss_curve.png", bbox_inches="tight", dpi=150)
    plt.close()


def save_metric_curves(trainer: Seq2SeqTrainer, plots_dir: Path) -> None:
    tracked = {
        "eval_mean_cer": "Validation CER",
        "eval_exact_match_accuracy": "Exact match",
        "eval_digits_only_accuracy": "Digits correct (ignoring dot)",
        "eval_dot_position_accuracy": "Dot position correct",
    }

    series: dict[str, tuple[list[float], list[float]]] = {}

    for key in tracked:
        epochs = []
        values = []

        for log in trainer.state.log_history:
            if key in log and "epoch" in log:
                epochs.append(log["epoch"])
                values.append(log[key])

        if values:
            series[key] = (epochs, values)

    if not series:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    for key, (epochs, values) in series.items():
        plt.plot(epochs, values, marker="o", label=tracked[key])

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation metrics by epoch")
    plt.legend()
    plt.grid(True)

    plt.savefig(plots_dir / "metric_curves.png", bbox_inches="tight", dpi=150)
    plt.close()


def save_final_metrics(
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    plots_dir: Path,
) -> None:
    save_json(
        {"train": train_metrics, "eval": eval_metrics},
        plots_dir / "final_metrics.json",
    )


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)
    config = apply_overrides(config, args)
    config = apply_default_train_values(config)

    use_mlflow = mlflow_enabled(config)

    with start_mlflow_run(config):
        model_cfg = config["model"]
        data_cfg = config["data"]
        train_cfg = config["train"]

        seed_everything(train_cfg["seed"])

        output_dir = Path(train_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = output_dir / PLOTS_DIRNAME
        plots_dir.mkdir(parents=True, exist_ok=True)

        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "0"

        save_run_inputs(config, args.config, plots_dir)

        if use_mlflow:
            mlflow.log_artifact(args.config, artifact_path="config")
            mlflow.log_params(build_params(config))

        processor, model = load_model_and_processor(
            model_cfg["name"],
            max_target_length=train_cfg["max_target_length"],
            length_penalty=train_cfg["length_penalty"],
            num_beams=train_cfg["num_beams"],
        )

        train_augmentations = build_train_augmentations()

        train_dataset = MeterOCRDataset(
            images_dir=data_cfg["train_images"],
            labels_csv=data_cfg["train_labels"],
            processor=processor,
            image_column=data_cfg["image_column"],
            text_column=data_cfg["text_column"],
            max_target_length=train_cfg["max_target_length"],
            augmentations=train_augmentations,
        )

        val_dataset = MeterOCRDataset(
            images_dir=data_cfg["val_images"],
            labels_csv=data_cfg["val_labels"],
            processor=processor,
            image_column=data_cfg["image_column"],
            text_column=data_cfg["text_column"],
            max_target_length=train_cfg["max_target_length"],
            augmentations=None,
        )

        check_label_encoding(processor, train_dataset)

        plot_augmentation_samples(
            train_dataset,
            plots_dir / "augmentations.png",
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            per_device_eval_batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            seed=train_cfg["seed"],
            data_seed=train_cfg["seed"],
            predict_with_generate=True,
            generation_max_length=train_cfg["max_target_length"],
            generation_num_beams=train_cfg["num_beams"],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_exact_match_accuracy",
            greater_is_better=True,
            report_to="none",
            # warmup_ratio=train_cfg["warmup_ratio"],
            # lr_scheduler_type=train_cfg["lr_scheduler_type"],
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
        eval_metrics = trainer.evaluate()

        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        save_trainer_history(trainer, plots_dir)
        save_loss_plot(trainer, plots_dir)
        save_metric_curves(trainer, plots_dir)
        save_final_metrics(train_result.metrics, eval_metrics, plots_dir)

        log_evaluation_report(
            model=trainer.model,
            processor=processor,
            dataset=val_dataset,
            output_dir=output_dir,
            batch_size=train_cfg["batch_size"],
        )

        if use_mlflow:
            log_numeric_metrics_to_mlflow(train_result.metrics)
            log_numeric_metrics_to_mlflow(eval_metrics)
            log_trainer_history_to_mlflow(trainer)

            mlflow.log_artifacts(str(plots_dir), artifact_path=PLOTS_DIRNAME)
            mlflow.log_artifacts(str(output_dir / REPORT_DIRNAME), artifact_path=REPORT_DIRNAME)

            if config["mlflow"].get("log_model"):
                mlflow.log_artifacts(str(output_dir), artifact_path="model")

        print(f"Training finished. Model saved to: {output_dir}")
        print(f"Plots, params and metrics: {plots_dir}")
        print(f"Training metrics: {train_result.metrics}")
        print(f"Validation metrics: {eval_metrics}")

        if use_mlflow:
            print(f"Logged to MLflow at: {config['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
