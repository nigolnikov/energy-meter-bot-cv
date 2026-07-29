import json
from collections import Counter
from pathlib import Path

import jiwer
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.ocr.text import decode_label, is_valid_reading

REPORT_DIRNAME = "report"


@torch.no_grad()
def predict_dataset(
    model,
    processor,
    dataset,
    batch_size: int = 8,
    device: str | None = None,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: torch.stack([item["pixel_values"] for item in batch]),
    )

    predictions: list[str] = []
    confidences: list[float] = []

    for pixel_values in loader:
        pixel_values = pixel_values.to(device)

        output = model.generate(
            pixel_values,
            return_dict_in_generate=True,
            output_scores=True,
        )

        decoded = processor.batch_decode(output.sequences, skip_special_tokens=True)
        predictions.extend(decode_label(text) for text in decoded)

        transition = model.compute_transition_scores(
            output.sequences,
            output.scores,
            getattr(output, "beam_indices", None),
            normalize_logits=True,
        )

        finite = torch.isfinite(transition)
        transition = transition.masked_fill(~finite, 0.0)
        lengths = finite.sum(dim=1).clamp(min=1)

        confidences.extend((transition.sum(dim=1) / lengths).float().cpu().tolist())

    df = dataset.df.copy()
    df = df.rename(
        columns={
            dataset.image_column: "filename",
            dataset.text_column: "reference",
        }
    )[["filename", "reference"]]

    df["reference"] = df["reference"].astype(str).str.strip()
    df["prediction"] = predictions
    df["confidence"] = confidences

    df["cer"] = [
        jiwer.cer(ref, pred) if ref else float(pred != "")
        for ref, pred in zip(df["reference"], df["prediction"], strict=True)
    ]

    df["exact_match"] = df["reference"] == df["prediction"]
    df["valid_format"] = df["prediction"].map(is_valid_reading)
    df["length_delta"] = df["prediction"].str.len() - df["reference"].str.len()

    df["digits_match"] = df["reference"].str.replace(".", "", regex=False) == df[
        "prediction"
    ].str.replace(".", "", regex=False)
    df["dot_match"] = df["reference"].str.find(".") == df["prediction"].str.find(".")

    df["error_type"] = np.select(
        [
            df["exact_match"],
            df["digits_match"] & ~df["dot_match"],
            ~df["digits_match"] & df["dot_match"],
        ],
        ["correct", "dot_only", "digits_only"],
        default="both",
    )

    return df


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def plot_error_types(df: pd.DataFrame, path: Path) -> Path:
    order = ["correct", "dot_only", "digits_only", "both"]
    counts = df["error_type"].value_counts()
    values = [counts.get(name, 0) for name in order]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(order, values, edgecolor="black")

    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n({value / len(df):.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylabel("Samples")
    plt.title("Error breakdown (digits vs. decimal dot)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

    return path


def plot_coverage_accuracy(df: pd.DataFrame, path: Path) -> Path:
    ordered = df.sort_values("confidence", ascending=False)
    accepted = ordered["exact_match"].to_numpy()

    coverage = np.arange(1, len(accepted) + 1) / len(accepted)
    accuracy = np.cumsum(accepted) / np.arange(1, len(accepted) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(coverage * 100, accuracy * 100)
    plt.axhline(accepted.mean() * 100, linestyle="--", color="grey", label="Accept everything")
    plt.xlabel("Coverage (% of readings auto-accepted)")
    plt.ylabel("Exact-match accuracy on accepted (%)")
    plt.title("Accuracy vs. coverage under a confidence threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

    return path


def plot_char_confusion(df: pd.DataFrame, path: Path) -> Path:
    substitutions: Counter = Counter()
    deletions: Counter = Counter()
    insertions: Counter = Counter()

    for reference, prediction in zip(df["reference"], df["prediction"], strict=True):
        if not reference:
            continue

        output = jiwer.process_characters(reference, prediction)

        for chunk in output.alignments[0]:
            ref_chunk = reference[chunk.ref_start_idx : chunk.ref_end_idx]
            hyp_chunk = prediction[chunk.hyp_start_idx : chunk.hyp_end_idx]

            if chunk.type == "substitute":
                for ref_char, hyp_char in zip(ref_chunk, hyp_chunk, strict=False):
                    substitutions[(ref_char, hyp_char)] += 1
            elif chunk.type == "delete":
                deletions.update(ref_chunk)
            elif chunk.type == "insert":
                insertions.update(hyp_chunk)

    labels = sorted(
        {char for pair in substitutions for char in pair} | set(deletions) | set(insertions)
    )

    if not labels:
        return path

    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels) + 2))

    for (ref_char, hyp_char), count in substitutions.items():
        matrix[index[ref_char], index[hyp_char]] = count

    for ref_char, count in deletions.items():
        matrix[index[ref_char], len(labels)] = count

    for hyp_char, count in insertions.items():
        matrix[index[hyp_char], len(labels) + 1] = count

    plt.figure(figsize=(1 + 0.6 * len(labels), 0.6 * len(labels) + 2))
    plt.imshow(matrix, cmap="Reds", aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(range(len(labels) + 2), [*labels, "(deleted)", "(inserted)"], rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Character-level error matrix (errors only)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                plt.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8)

    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

    return path


def plot_position_accuracy(df: pd.DataFrame, path: Path) -> Path:
    max_len = int(df["reference"].str.len().max())

    left = np.zeros(max_len)
    left_total = np.zeros(max_len)
    right = np.zeros(max_len)
    right_total = np.zeros(max_len)

    for reference, prediction in zip(df["reference"], df["prediction"], strict=True):
        for i, char in enumerate(reference):
            left_total[i] += 1
            if i < len(prediction) and prediction[i] == char:
                left[i] += 1

        for i, char in enumerate(reversed(reference)):
            right_total[i] += 1
            if i < len(prediction) and prediction[len(prediction) - 1 - i] == char:
                right[i] += 1

    positions = np.arange(max_len)
    width = 0.4

    plt.figure(figsize=(9, 5))
    plt.bar(
        positions - width / 2,
        np.divide(left, left_total, out=np.zeros(max_len), where=left_total > 0),
        width,
        label="Position from left",
    )
    plt.bar(
        positions + width / 2,
        np.divide(right, right_total, out=np.zeros(max_len), where=right_total > 0),
        width,
        label="Position from right",
    )
    plt.xticks(positions, [str(i) for i in positions])
    plt.ylim(0, 1)
    plt.xlabel("Character index")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by character position")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

    return path


def plot_worst_predictions(
    df: pd.DataFrame,
    images_dir: str | Path,
    path: Path,
    top_n: int = 16,
    columns: int = 4,
) -> Path:
    worst = df.sort_values(["cer", "confidence"], ascending=[False, True]).head(top_n)

    if worst.empty:
        return path

    rows = int(np.ceil(len(worst) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, worst.iterrows(), strict=False):
        image = Image.open(Path(images_dir) / row["filename"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(
            f"true: {row['reference']}\n"
            f"pred: {row['prediction']}  ({row['error_type']}, cer={row['cer']:.2f})",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(worst) :]:
        ax.axis("off")

    fig.suptitle("Worst predictions", fontsize=14)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return path


def plot_augmentation_samples(
    dataset,
    path: str | Path,
    n_samples: int = 8,
    columns: int = 4,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = int(np.ceil(n_samples / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    for ax, idx in zip(axes, indices, strict=False):
        row = dataset.df.iloc[idx]
        image = Image.open(Path(dataset.images_dir) / row[dataset.image_column]).convert("RGB")

        if dataset.augmentations is not None:
            image = Image.fromarray(dataset.augmentations(image=np.array(image))["image"])

        ax.imshow(image)
        ax.set_title(str(row[dataset.text_column]), fontsize=10)
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    fig.suptitle("Augmented training samples", fontsize=14)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return path


def log_evaluation_report(
    model,
    processor,
    dataset,
    output_dir: str | Path,
    batch_size: int = 8,
    numeric_tolerance: float = 1.0,
) -> pd.DataFrame:
    report_dir = Path(output_dir) / REPORT_DIRNAME
    report_dir.mkdir(parents=True, exist_ok=True)

    df = predict_dataset(model, processor, dataset, batch_size=batch_size)

    df.to_csv(report_dir / "val_predictions.csv", index=False)

    plot_error_types(df, report_dir / "error_types.png")
    plot_coverage_accuracy(df, report_dir / "coverage_accuracy.png")
    plot_char_confusion(df, report_dir / "char_confusion.png")
    plot_position_accuracy(df, report_dir / "position_accuracy.png")
    plot_worst_predictions(df, dataset.images_dir, report_dir / "worst_predictions.png")

    numeric = df[df["prediction"].map(_is_number) & df["reference"].map(_is_number)]

    within_tolerance = (
        float(
            (
                (numeric["prediction"].astype(float) - numeric["reference"].astype(float)).abs()
                <= numeric_tolerance
            ).mean()
        )
        if len(numeric)
        else 0.0
    )

    error_share = df["error_type"].value_counts(normalize=True)

    summary = {
        "final_exact_match": float(df["exact_match"].mean()),
        "final_mean_cer": float(df["cer"].mean()),
        "final_median_cer": float(df["cer"].median()),
        "final_p90_cer": float(df["cer"].quantile(0.9)),
        "final_digits_match": float(df["digits_match"].mean()),
        "final_dot_match": float(df["dot_match"].mean()),
        "invalid_format_rate": float(1 - df["valid_format"].mean()),
        "within_tolerance": within_tolerance,
        "error_dot_only": float(error_share.get("dot_only", 0.0)),
        "error_digits_only": float(error_share.get("digits_only", 0.0)),
        "error_both": float(error_share.get("both", 0.0)),
        "accuracy_at_80pct_coverage": _accuracy_at_coverage(df, 0.8),
        "accuracy_at_50pct_coverage": _accuracy_at_coverage(df, 0.5),
    }

    with (report_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation report:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    print(f"  written to: {report_dir}")

    return df


def _accuracy_at_coverage(df: pd.DataFrame, coverage: float) -> float:
    keep = max(1, int(len(df) * coverage))
    top = df.sort_values("confidence", ascending=False).head(keep)

    return float(top["exact_match"].mean())
