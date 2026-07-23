import argparse
import csv
from collections import Counter
from pathlib import Path

from pipeline import run_pipeline
from src.utils.logger import logger
from src.utils.metrics import character_error_rate, digit_accuracy, exact_match


def load_test_set(csv_path: Path) -> list[tuple[str, str]]:
    """
    Reads test-csv with columns image_path, true_value.
    Returns list of pairs (path_to_photo, true_value)
    """
    pairs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["image_path"], row["true_value"]))
    logger.info(f"Loaded {len(pairs)} test samples from {csv_path}")
    return pairs


def evaluate(pairs: list[tuple[str, str]], enhancement: str, crop_mode: str) -> dict:
    results = []

    for image_path, true_value in pairs:
        try:
            result = run_pipeline(image_path, enhancement=enhancement, crop_mode=crop_mode)
            pred_value = result.value

            results.append(
                {
                    "image": image_path,
                    "true": true_value,
                    "pred": pred_value,
                    "status": result.status,
                    "is_exact": exact_match(true_value, pred_value),
                    "digit_acc": digit_accuracy(true_value, pred_value),
                    "cer": character_error_rate(true_value, pred_value),
                }
            )
        except Exception as e:
            logger.error(f"Pipeline failed on {image_path}: {e}")
            results.append(
                {
                    "image": image_path,
                    "true": true_value,
                    "pred": "",
                    "status": "pipeline_error",
                    "is_exact": False,
                    "digit_acc": 0.0,
                    "cer": 1.0,
                }
            )

    return aggregate(results)


def aggregate(results: list[dict]) -> dict:
    total = len(results)

    exact_count = sum(r["is_exact"] for r in results)
    e2e_exact_match = exact_count / total

    digit_sum = sum(r["digit_acc"] for r in results)
    mean_digit_accuracy = digit_sum / total

    cer_sum = sum(r["cer"] for r in results)
    mean_cer = cer_sum / total

    status_breakdown = dict(Counter(r["status"] for r in results))

    return {
        "total": total,
        "exact_matches": exact_count,
        "e2e_exact_match": e2e_exact_match,
        "mean_digit_accuracy": mean_digit_accuracy,
        "mean_cer": mean_cer,
        "status_breakdown": status_breakdown,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--enhancement", choices=["none", "clahe", "zerodce"], default="none")
    parser.add_argument("--crop-mode", choices=["axis", "warp"], default="warp")
    args = parser.parse_args()

    pairs = load_test_set(args.test_csv)
    metrics = evaluate(pairs, enhancement=args.enhancement, crop_mode=args.crop_mode)

    logger.info(f"E2E results (enhancement={args.enhancement}):")
    for key, val in metrics.items():
        logger.info(f"   {key}: {val}")


if __name__ == "__main__":
    main()
