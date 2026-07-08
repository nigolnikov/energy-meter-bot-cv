import argparse
from pathlib import Path

import albumentations as A
import cv2
import jiwer
import numpy as np
import pandas as pd
from PIL import Image

from src.ocr.infer_trocr import ocr_infer


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    return str(text).strip().replace(" ", "")


def compute_digit_accuracy(true_text: str, pred_text: str) -> float:
    true_text = normalize_text(true_text)
    pred_text = normalize_text(pred_text)

    if len(true_text) == 0:
        return 0.0

    max_len = max(len(true_text), len(pred_text))

    true_padded = true_text.ljust(max_len)
    pred_padded = pred_text.ljust(max_len)

    correct = sum(t == p for t, p in zip(true_padded, pred_padded, strict=False))

    return correct / max_len


def build_robustness_transforms() -> dict[str, A.Compose | None]:
    return {
        "clean": None,
        "perspective_mild": A.Compose(
            [
                A.Perspective(
                    scale=(0.015, 0.025),
                    keep_size=True,
                    fit_output=False,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=1.0,
                ),
            ]
        ),
        "perspective_medium": A.Compose(
            [
                A.Perspective(
                    scale=(0.03, 0.05),
                    keep_size=True,
                    fit_output=False,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=1.0,
                ),
            ]
        ),
        "perspective_strong": A.Compose(
            [
                A.Perspective(
                    scale=(0.05, 0.075),
                    keep_size=True,
                    fit_output=False,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=1.0,
                ),
            ]
        ),
        "blur_mild": A.Compose(
            [
                A.MotionBlur(
                    blur_limit=(7, 13),
                    angle_range=(0, 360),
                    p=1.0,
                ),
            ]
        ),
        "blur_medium": A.Compose(
            [
                A.MotionBlur(
                    blur_limit=(15, 25),
                    angle_range=(0, 360),
                    p=1.0,
                ),
            ]
        ),
        "blur_strong": A.Compose(
            [
                A.MotionBlur(
                    blur_limit=(30, 45),
                    angle_range=(0, 360),
                    p=1.0,
                ),
            ]
        ),
        "downscale_mild": A.Compose(
            [
                A.Downscale(
                    scale_range=(0.45, 0.65),
                    interpolation_pair={
                        "downscale": cv2.INTER_NEAREST,
                        "upscale": cv2.INTER_NEAREST,
                    },
                    p=1.0,
                ),
            ]
        ),
        "downscale_medium": A.Compose(
            [
                A.Downscale(
                    scale_range=(0.25, 0.45),
                    interpolation_pair={
                        "downscale": cv2.INTER_NEAREST,
                        "upscale": cv2.INTER_NEAREST,
                    },
                    p=1.0,
                ),
            ]
        ),
        "downscale_strong": A.Compose(
            [
                A.Downscale(
                    scale_range=(0.12, 0.25),
                    interpolation_pair={
                        "downscale": cv2.INTER_NEAREST,
                        "upscale": cv2.INTER_NEAREST,
                    },
                    p=1.0,
                ),
            ]
        ),
        "shadow_mild": A.Compose(
            [
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 1),
                    shadow_dimension=3,
                    p=1.0,
                ),
            ]
        ),
        "shadow_medium": A.Compose(
            [
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 2),
                    shadow_dimension=5,
                    p=1.0,
                ),
            ]
        ),
        "shadow_strong": A.Compose(
            [
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(2, 3),
                    shadow_dimension=7,
                    p=1.0,
                ),
            ]
        ),
    }


def apply_transform(image: np.ndarray, transform: A.Compose | None) -> np.ndarray:
    if transform is None:
        return image

    augmented = transform(image=image)

    return augmented["image"]


def get_status(delta: float) -> str:
    if delta >= -3.0:
        return "OK"

    if delta >= -8.0:
        return "WARN"

    return "FAIL"


def evaluate_transform(
    images_dir: Path,
    labels_df: pd.DataFrame,
    image_column: str,
    text_column: str,
    test_name: str,
    transform: A.Compose | None,
) -> tuple[dict, list[dict]]:
    prediction_rows = []

    exact_matches = []
    digit_accuracies = []
    cers = []
    wers = []
    numeric_errors = []

    for _, row in labels_df.iterrows():
        filename = str(row[image_column])
        true_text = normalize_text(row[text_column])

        image_path = images_dir / filename

        if not image_path.exists():
            print(f"[WARNING] Image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        transformed_image = apply_transform(image_np, transform)

        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

        prediction = ocr_infer(transformed_image)

        pred_text = normalize_text(prediction.text)

        is_exact = true_text == pred_text

        cer = jiwer.cer(true_text, pred_text)
        wer = jiwer.wer(true_text, pred_text)
        digit_accuracy = compute_digit_accuracy(true_text, pred_text)

        exact_matches.append(is_exact)
        digit_accuracies.append(digit_accuracy)
        cers.append(cer)
        wers.append(wer)

        prediction_rows.append(
            {
                "test_name": test_name,
                "filename": filename,
                "true_text": true_text,
                "predicted_text": pred_text,
                "exact_match": is_exact,
                "cer": cer,
                "wer": wer,
                "digit_accuracy": digit_accuracy,
            }
        )

    total = len(prediction_rows)

    if total == 0:
        summary = {
            "robustness_test": test_name,
            "total": 0,
            "accuracy": 0.0,
            "cer": 0.0,
            "wer": 0.0,
            "digit_accuracy": 0.0,
            "numeric_error": None,
        }

        return summary, prediction_rows

    summary = {
        "robustness_test": test_name,
        "total": total,
        "accuracy": float(np.mean(exact_matches)) * 100.0,
        "cer": float(np.mean(cers)),
        "wer": float(np.mean(wers)),
        "digit_accuracy": float(np.mean(digit_accuracies)) * 100.0,
        "numeric_error": float(np.mean(numeric_errors)) if numeric_errors else None,
    }

    return summary, prediction_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Path to test images directory",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Path to test labels CSV",
    )
    parser.add_argument(
        "--image-column",
        default="filename",
        help="CSV column with image filenames",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="CSV column with ground truth text",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/robustness_trocr",
        type=Path,
        help="Directory where results will be saved",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(
        args.labels,
        dtype={
            args.image_column: str,
            args.text_column: str,
        },
    )

    transforms = build_robustness_transforms()

    summary_rows = []
    all_prediction_rows = []

    baseline_accuracy = None

    for test_name, transform in transforms.items():
        print(f"Running robustness test: {test_name}")

        summary, prediction_rows = evaluate_transform(
            images_dir=args.images,
            labels_df=labels_df,
            image_column=args.image_column,
            text_column=args.text_column,
            test_name=test_name,
            transform=transform,
        )

        if test_name == "clean":
            baseline_accuracy = summary["accuracy"]
            delta = 0.0
            status = "BASELINE"
        else:
            delta = summary["accuracy"] - baseline_accuracy
            status = get_status(delta)

        summary["delta"] = delta
        summary["status"] = status

        summary_rows.append(summary)
        all_prediction_rows.extend(prediction_rows)

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.DataFrame(all_prediction_rows)

    summary_df = summary_df[
        [
            "robustness_test",
            "total",
            "accuracy",
            "delta",
            "cer",
            "wer",
            "digit_accuracy",
            "status",
        ]
    ]

    summary_df["accuracy"] = summary_df["accuracy"].round(2)
    summary_df["delta"] = summary_df["delta"].round(2)
    summary_df["cer"] = summary_df["cer"].round(4)
    summary_df["wer"] = summary_df["wer"].round(4)
    summary_df["digit_accuracy"] = summary_df["digit_accuracy"].round(2)

    summary_path = args.output_dir / "robustness_summary.csv"
    predictions_path = args.output_dir / "robustness_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print()
    print(summary_df)
    print()
    print(f"Saved summary to: {summary_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
