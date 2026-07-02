# TrOCR — evaluation for meter reading OCR.
# Responsible for comparing model predictions with true labels.

import argparse
from pathlib import Path

import jiwer
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.ocr.infer_trocr import infer
from src.utils.logger import logger

MODEL_PATH = "models/trocr-meter-finetuned"


def load_model(
    model_name_or_path: str = MODEL_PATH,
) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = TrOCRProcessor.from_pretrained(model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)

    model.to(device)
    model.eval()

    logger.info(f"Using device: {device}")
    logger.info(f"Loaded TrOCR model from: {model_name_or_path}")

    return processor, model, device


def character_error_rate(true_text: str, predicted_text: str) -> float:
    cer = jiwer.cer(true_text, predicted_text)
    return cer


def word_error_rate(true_text: str, predicted_text: str) -> float:
    wer = jiwer.wer(true_text, predicted_text)
    return wer


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrOCR model")

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to directory with test/validation images.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels CSV file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/small-stage1",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processor, model, device = load_model(args.model)

    labels = pd.read_csv(
        args.labels,
        dtype={
            "filename": str,
            "text": str,
        },
    )

    results = []

    for _, row in labels.iterrows():
        image_name = row["filename"]
        true_text = str(row["text"]).strip()

        image_path = args.images / image_name

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        with Image.open(image_path) as image:
            result = infer(
                image=image,
                processor=processor,
                model=model,
                device=device,
                max_new_tokens=9,
            )

        predicted_text = result.text.strip()

        exact_match = true_text == predicted_text
        cer = character_error_rate(true_text, predicted_text)
        wer = word_error_rate(true_text, predicted_text)
        digit_acc = digit_accuracy(true_text, predicted_text)
        num_error = numeric_error(true_text, predicted_text)

        print(
            f"Image: {image_name} | "
            f"True: {true_text} | "
            f"Predicted: {predicted_text} | "
            f"Exact: {exact_match}"
        )

        results.append(
            {
                "filename": image_name,
                "true_text": true_text,
                "predicted_text": predicted_text,
                "confidence": result.confidence,
                "exact_match": exact_match,
                "cer": cer,
                "wer": wer,
                "digit_accuracy": digit_acc,
                "numeric_error": num_error,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv("trocr_predictions.csv", index=False)

    summary = {
        "num_images": len(results_df),
        "exact_match_accuracy": results_df["exact_match"].mean(),
        "mean_cer": results_df["cer"].mean(),
        "mean_wer": results_df["wer"].mean(),
        "mean_digit_accuracy": results_df["digit_accuracy"].mean(),
        "mean_numeric_error": results_df["numeric_error"].dropna().mean(),
        "median_numeric_error": results_df["numeric_error"].dropna().median(),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("trocr_summary.csv", index=False)

    logger.info("Saved evaluation results to trocr_predictions.csv")
    logger.info("Saved evaluation summary to trocr_summary.csv")

    print("\nEvaluation summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
