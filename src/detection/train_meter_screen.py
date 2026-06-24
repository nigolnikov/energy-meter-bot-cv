import argparse
from pathlib import Path
from typing import Any

import mlflow
from ultralytics import YOLO

from src.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO OBB model for meter screen detection.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    if args.imgsz is not None:
        config["train"]["imgsz"] = args.imgsz

    if args.batch is not None:
        config["train"]["batch"] = args.batch

    if args.model is not None:
        config["train"]["model"] = args.model

    if args.run_name is not None:
        config["experiment"]["run_name"] = args.run_name
        config["train"]["name"] = args.run_name

    return config


def log_artifacts(save_dir: Path) -> None:
    artifacts = [
        save_dir / "results.csv",
        save_dir / "args.yaml",
        save_dir / "weights" / "best.pt",
        save_dir / "weights" / "last.pt",
    ]

    for artifact in artifacts:
        if artifact.exists():
            mlflow.log_artifact(str(artifact))

    for image_path in save_dir.glob("*.png"):
        mlflow.log_artifact(str(image_path))


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)
    config = apply_overrides(config, args)

    experiment_cfg = config["experiment"]
    mlflow_cfg = config["mlflow"]
    train_cfg = config["train"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(experiment_cfg["name"])

    with mlflow.start_run(run_name=experiment_cfg["run_name"]):
        mlflow.log_param("config_path", args.config)
        mlflow.log_params(train_cfg)

        model = YOLO(train_cfg["model"])

        results = model.train(
            task=train_cfg["task"],
            data=train_cfg["data"],
            epochs=train_cfg["epochs"],
            imgsz=train_cfg["imgsz"],
            batch=train_cfg["batch"],
            project=train_cfg["project"],
            name=train_cfg["name"],
        )

        save_dir = Path(results.save_dir)
        log_artifacts(save_dir)

        print(f"Training finished. Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
