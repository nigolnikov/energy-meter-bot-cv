import argparse
from pathlib import Path

import jiwer
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.ocr.infer_trocr import infer
from src.ocr.text import decode_label, is_valid_reading
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


def classify_error(true_text: str, predicted_text: str) -> str:
    if true_text == predicted_text:
        return "correct"

    digits_match = true_text.replace(".", "") == predicted_text.replace(".", "")
    dot_match = true_text.find(".") == predicted_text.find(".")

    if digits_match and not dot_match:
        return "dot_only"

    if not digits_match and dot_match:
        return "digits_only"

    return "both"


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
        default=MODEL_PATH,
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="trocr",
        help="Prefix for the predictions/summary CSVs.",
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
            )

        predicted_text = decode_label(result.text)

        exact_match = true_text == predicted_text
        cer = character_error_rate(true_text, predicted_text)
        digit_acc = digit_accuracy(true_text, predicted_text)
        num_error = numeric_error(true_text, predicted_text)
        error_type = classify_error(true_text, predicted_text)

        print(
            f"Image: {image_name} | "
            f"True: {true_text} | "
            f"Predicted: {predicted_text} | "
            f"Exact: {exact_match} | "
            f"Error: {error_type}"
        )

        results.append(
            {
                "filename": image_name,
                "true_text": true_text,
                "predicted_text": predicted_text,
                "confidence": result.confidence,
                "exact_match": exact_match,
                "cer": cer,
                "digit_accuracy": digit_acc,
                "numeric_error": num_error,
                "error_type": error_type,
                "valid_format": is_valid_reading(predicted_text),
                "digits_match": true_text.replace(".", "") == predicted_text.replace(".", ""),
                "dot_match": true_text.find(".") == predicted_text.find("."),
            }
        )

    results_df = pd.DataFrame(results)

    predictions_path = f"{args.output_prefix}_predictions.csv"
    summary_path = f"{args.output_prefix}_summary.csv"

    results_df.to_csv(predictions_path, index=False)

    error_share = results_df["error_type"].value_counts(normalize=True)

    summary = {
        "num_images": len(results_df),
        "exact_match_accuracy": results_df["exact_match"].mean(),
        "mean_cer": results_df["cer"].mean(),
        "mean_digit_accuracy": results_df["digit_accuracy"].mean(),
        # Digits right but dot misplaced, vs. dot right but digits misread.
        "digits_match_accuracy": results_df["digits_match"].mean(),
        "dot_position_accuracy": results_df["dot_match"].mean(),
        "error_dot_only": error_share.get("dot_only", 0.0),
        "error_digits_only": error_share.get("digits_only", 0.0),
        "error_both": error_share.get("both", 0.0),
        "invalid_format_rate": 1 - results_df["valid_format"].mean(),
        "mean_numeric_error": results_df["numeric_error"].dropna().mean(),
        "median_numeric_error": results_df["numeric_error"].dropna().median(),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"Saved evaluation results to {predictions_path}")
    logger.info(f"Saved evaluation summary to {summary_path}")

    print("\nEvaluation summary:")
    print(summary_df.to_string(index=False))

    print("\nWorst predictions:")
    worst = results_df[~results_df["exact_match"]].sort_values("cer", ascending=False).head(10)
    print(
        worst[["filename", "true_text", "predicted_text", "error_type", "cer"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
