import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import mlflow
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

from src.ocr.augmentations_analog import build_train_augmentations
from src.ocr.reporting import REPORT_DIRNAME, log_evaluation_report, plot_augmentation_samples
from src.ocr.trocr_training import (
    PLOTS_DIRNAME,
    MeterOCRDataset,
    add_common_train_args,
    apply_common_overrides,
    apply_common_train_defaults,
    build_common_params,
    build_compute_metrics,
    check_label_encoding,
    load_model_and_processor,
    log_numeric_metrics_to_mlflow,
    log_trainer_history_to_mlflow,
    mlflow_enabled,
    save_loss_plot,
    save_metric_curves,
    save_run_inputs,
    save_trainer_history,
    seed_everything,
    start_mlflow_run,
)
from src.utils.config import load_yaml_config

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-5

DEFAULT_MAX_TARGET_LENGTH = 13
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42

BEST_METRIC_BASE = "eval_exact_match_accuracy"
BEST_METRIC_GREATER_IS_BETTER = True

METRIC_CURVES = {
    "eval_mean_cer": ("Validation CER", "-"),
    "eval_exact_match_accuracy": ("Exact match", "-"),
    "eval_length_error_rate": ("Wrong length", "-"),
    "eval_invalid_format_rate": ("Unparseable", "-"),
    "eval_digits_only_accuracy": ("Digits correct", "--"),
    "eval_dot_position_accuracy": ("Dot position correct", "--"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR model for meter OCR.")
    add_common_train_args(parser)

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable train-time augmentation (debugging only).",
    )

    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = apply_common_overrides(config, args)

    if args.no_augment:
        config["train"]["augment"] = False

    return config


def apply_default_train_values(config: dict[str, Any]) -> dict[str, Any]:
    config = apply_common_train_defaults(
        config,
        default_epochs=DEFAULT_EPOCHS,
        default_batch_size=DEFAULT_BATCH_SIZE,
        default_learning_rate=DEFAULT_LEARNING_RATE,
        default_max_target_length=DEFAULT_MAX_TARGET_LENGTH,
        default_weight_decay=DEFAULT_WEIGHT_DECAY,
        default_seed=DEFAULT_SEED,
    )
    config["train"].setdefault("augment", True)

    return config


def best_metric_name(config: dict[str, Any]) -> str:
    return BEST_METRIC_BASE


def build_params(config: dict[str, Any]) -> dict[str, Any]:
    params = build_common_params(config)
    params["augment"] = config["train"]["augment"]
    params["best_metric"] = best_metric_name(config)

    return params


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

        params = build_params(config)
        save_run_inputs(args.config, plots_dir, params)

        if use_mlflow:
            mlflow.log_artifact(args.config, artifact_path="config")
            mlflow.log_params(params)

        processor, model = load_model_and_processor(
            model_cfg["name"],
            max_target_length=train_cfg["max_target_length"],
            length_penalty=train_cfg["length_penalty"],
            num_beams=train_cfg["num_beams"],
        )

        train_augmentations = build_train_augmentations() if train_cfg["augment"] else None

        if train_augmentations is None:
            print("WARNING: training WITHOUT augmentation.")

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

        if train_augmentations is not None:
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
            metric_for_best_model=best_metric_name(config),
            greater_is_better=BEST_METRIC_GREATER_IS_BETTER,
            report_to="none",
        )

        print(
            f"Selecting the best checkpoint on {best_metric_name(config)} "
            f"(greater_is_better={BEST_METRIC_GREATER_IS_BETTER})."
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            processing_class=processor,
            compute_metrics=build_compute_metrics(processor, track_length=True),
        )

        train_result = trainer.train()

        val_metrics = trainer.evaluate(
            eval_dataset=val_dataset,
            metric_key_prefix="eval",
        )

        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        save_trainer_history(trainer, plots_dir)
        save_loss_plot(trainer, plots_dir)
        save_metric_curves(trainer, plots_dir, METRIC_CURVES)

        log_evaluation_report(
            model=trainer.model,
            processor=processor,
            dataset=val_dataset,
            output_dir=output_dir / "val",
            batch_size=train_cfg["batch_size"],
        )

        if use_mlflow:
            log_numeric_metrics_to_mlflow(train_result.metrics)

            log_numeric_metrics_to_mlflow(val_metrics)

            log_trainer_history_to_mlflow(trainer)

            mlflow.log_artifacts(str(plots_dir), artifact_path=PLOTS_DIRNAME)
            mlflow.log_artifacts(
                str(output_dir / "val" / REPORT_DIRNAME),
                artifact_path=f"{REPORT_DIRNAME}_val",
            )

            if config["mlflow"].get("log_model"):
                mlflow.log_artifacts(str(output_dir), artifact_path="model")

        print(f"Training finished. Model saved to: {output_dir}")
        print(f"Plots, params and metrics: {plots_dir}")
        print(f"Training metrics: {train_result.metrics}")

        print(f"Validation metrics: {val_metrics}")

        if use_mlflow:
            print(f"Logged to MLflow at: {config['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
