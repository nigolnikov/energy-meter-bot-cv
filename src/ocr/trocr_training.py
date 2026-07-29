"""
Shared building blocks for the TrOCR fine-tuning scripts.
"""

import json
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    set_seed,
)

from src.ocr.text import decode_label, encode_label
from src.utils.metrics import character_error_rate, digit_accuracy, numeric_error

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


def add_common_train_args(parser) -> None:
    """CLI flags shared by every TrOCR training script."""
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


def apply_common_overrides(config: dict[str, Any], args) -> dict[str, Any]:
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


def apply_common_train_defaults(
    config: dict[str, Any],
    *,
    default_epochs: int,
    default_batch_size: int,
    default_learning_rate: float,
    default_max_target_length: int,
    default_weight_decay: float,
    default_seed: int,
) -> dict[str, Any]:
    train_cfg = config["train"]

    train_cfg.setdefault("epochs", default_epochs)
    train_cfg.setdefault("batch_size", default_batch_size)
    train_cfg.setdefault("learning_rate", default_learning_rate)
    train_cfg.setdefault("max_target_length", default_max_target_length)
    train_cfg.setdefault("weight_decay", default_weight_decay)
    train_cfg.setdefault("length_penalty", 1.0)
    train_cfg.setdefault("num_beams", 3)
    train_cfg.setdefault("seed", default_seed)

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


def compute_ocr_metrics_from_texts(
    predictions: list[str],
    references: list[str],
    track_length: bool = False,
) -> dict[str, float]:
    exact_match_values = []
    cer_values = []
    digit_accuracy_values = []
    digits_only_match_values = []
    dot_match_values = []
    numeric_error_values = []
    length_delta_values = []

    parse_failures = 0

    for true_text, predicted_text in zip(references, predictions, strict=False):
        exact_match_values.append(true_text == predicted_text)
        cer_values.append(character_error_rate(true_text, predicted_text))
        digit_accuracy_values.append(digit_accuracy(true_text, predicted_text))

        digits_only_match_values.append(
            true_text.replace(".", "") == predicted_text.replace(".", "")
        )
        dot_match_values.append(true_text.find(".") == predicted_text.find("."))

        if track_length:
            length_delta_values.append(len(predicted_text) - len(true_text))

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

    if track_length:
        metrics["length_error_rate"] = float(np.mean([d != 0 for d in length_delta_values]))
        metrics["mean_length_delta"] = float(np.mean(length_delta_values))

    if numeric_error_values:
        metrics["mean_numeric_error"] = float(np.mean(numeric_error_values))
        metrics["median_numeric_error"] = float(np.median(numeric_error_values))

    return metrics


def build_compute_metrics(processor: TrOCRProcessor, track_length: bool = False):
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
            track_length=track_length,
        )

    return compute_metrics


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def build_common_params(config: dict[str, Any]) -> dict[str, Any]:
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
    }


def save_run_inputs(
    config_path: str,
    plots_dir: Path,
    params: dict[str, Any],
) -> None:
    save_json(params, plots_dir / "params.json")
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

    eval_rows = [log for log in trainer.state.log_history if any(k.endswith("_loss") for k in log)]

    if eval_rows:
        pd.DataFrame(eval_rows).to_csv(plots_dir / "eval_history.csv", index=False)


def series_from_history(trainer: Seq2SeqTrainer, key: str) -> tuple[list[float], list[float]]:
    epochs: list[float] = []
    values: list[float] = []

    for log in trainer.state.log_history:
        if key in log and "epoch" in log:
            epochs.append(log["epoch"])
            values.append(log[key])

    return epochs, values


def save_loss_plot(trainer: Seq2SeqTrainer, plots_dir: Path) -> None:
    train_epochs, train_losses = series_from_history(trainer, "loss")
    val_epochs, val_losses = series_from_history(trainer, "eval_loss")

    if not train_losses and not val_losses:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    if train_losses:
        plt.plot(train_epochs, train_losses, marker="o", label="Train loss")

    if val_losses:
        plt.plot(val_epochs, val_losses, marker="o", label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and validation loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(plots_dir / "loss_curve.png", bbox_inches="tight", dpi=150)
    plt.close()


def save_metric_curves(
    trainer: Seq2SeqTrainer,
    plots_dir: Path,
    tracked: dict[str, tuple[str, str]],
) -> None:
    """Plot validation metrics found in the trainer log history.

    `tracked` maps a log_history key (e.g. "eval_mean_cer") to a
    (label, linestyle) pair, so each training script controls which metrics
    it cares about without duplicating the plotting boilerplate.
    """
    series: dict[str, tuple[list[float], list[float], str]] = {}

    for key, (label, style) in tracked.items():
        epochs, values = series_from_history(trainer, key)

        if values:
            series[label] = (epochs, values, style)

    if not series:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    for label, (epochs, values, style) in series.items():
        plt.plot(epochs, values, marker="o", linestyle=style, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation metrics by epoch")
    plt.legend(fontsize=8)
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
