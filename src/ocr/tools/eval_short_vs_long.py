#!/usr/bin/env python3
"""
Baseline eval: run the SAME model on a SHORT batch and a LONG (8-digit) batch,
report both stratified and side by side.

    PYTHONPATH=. python -m src.ocr.tools.eval_short_vs_long \
        --short-csv data/test-short/labels.csv \
        --long-csv  data/ocr_analog_meter_v2/labels/test.csv \
        --out-dir eval_out
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.ocr.infer_trocr import ocr_infer
from src.ocr.text import decode_label
from src.utils.logger import logger
from src.utils.metrics import digit_accuracy, exact_match


def normalize(s: str) -> str:
    return decode_label(s) if s is not None else ""


def count_digits(s: str) -> int:
    return sum(c.isdigit() for c in s)


def load_image(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return np.array(img)


def detect_img_dir(csv_path: Path, first_file: str, override: str) -> Path:
    p = csv_path.parent
    cands: list[Path] = []
    if override:
        cands.append(Path(override))
    cands += [p, p / "images"]
    if p.name == "labels":
        base = p.parent
        cands += [base / "images" / csv_path.stem, base / "images"]
    cands += [p.parent / "images" / csv_path.stem, p.parent / "images"]

    for c in cands:
        if (c / first_file).exists():
            return c
    raise FileNotFoundError(
        f"Could not locate images for {csv_path}. Tried:\n  "
        + "\n  ".join(str(c / first_file) for c in cands)
        + "\nPass --short-img-dir / --long-img-dir explicitly."
    )


def load_test_set(csv_path: Path, img_dir: str = "", keep_digits: set[int] | None = None):
    """Read filename/text, resolve real image paths; optionally keep only
    rows with these digit counts."""
    rows, dropped = [], 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            true_value = normalize(row["text"])
            if keep_digits is not None and count_digits(true_value) not in keep_digits:
                dropped += 1
                continue
            rows.append((row["filename"].strip(), true_value))

    if not rows:
        logger.warning(f"No usable rows in {csv_path}")
        return []

    resolved = detect_img_dir(csv_path, rows[0][0], img_dir)
    pairs = [(str(resolved / fn), tv) for fn, tv in rows]
    missing = [pth for pth, _ in pairs if not Path(pth).exists()]

    msg = f"Loaded {len(pairs)} samples from {csv_path} (images: {resolved})"
    if keep_digits is not None:
        msg += f" [kept {sorted(keep_digits)} digits, dropped {dropped}]"
    logger.info(msg)
    if missing:
        logger.warning(f"{len(missing)} image(s) not found under {resolved}, " f"e.g. {missing[0]}")
    return pairs


def evaluate(pairs):
    results = []
    for image_path, true_value in pairs:
        try:
            image = load_image(image_path)
            result = ocr_infer(image)  # OCRResult(text, confidence)
            pred_value = normalize(result.text)
            confidence = float(result.confidence)
            status = "ok"
        except Exception as e:
            logger.error(f"ocr_infer failed on {image_path}: {e}")
            pred_value, confidence, status = "", 0.0, "infer_error"

        lt, lp = count_digits(true_value), count_digits(pred_value)
        results.append(
            {
                "image": image_path,
                "true": true_value,
                "pred": pred_value,
                "status": status,
                "confidence": confidence,
                "is_exact": exact_match(true_value, pred_value),
                "digit_acc": digit_accuracy(true_value, pred_value),
                "len_true": lt,
                "len_pred": lp,
                "len_delta": lp - lt,
            }
        )
    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"total": 0}
    return {
        "total": n,
        "exact_match": sum(r["is_exact"] for r in results) / n,
        "mean_digit_accuracy": sum(r["digit_acc"] for r in results) / n,
        "length_acc": sum(r["len_delta"] == 0 for r in results) / n,
        "over_gen_rate": sum(r["len_delta"] > 0 for r in results) / n,
        "under_gen_rate": sum(r["len_delta"] < 0 for r in results) / n,
        "status_breakdown": dict(Counter(r["status"] for r in results)),
    }


def by_length(results: list[dict]) -> dict[int, dict]:
    groups = defaultdict(list)
    for r in results:
        groups[r["len_true"]].append(r)
    return {L: aggregate(rs) for L, rs in sorted(groups.items())}


# ----------------------- reporting -----------------------
def _row(name, a: dict) -> str:
    if a.get("total", 0) == 0:
        return f"  {name:<12} (no samples)"
    return (
        f"  {name:<12} n={a['total']:<4d} exact={a['exact_match']:.3f} "
        f"digit_acc={a['mean_digit_accuracy']:.3f} len_acc={a['length_acc']:.3f} "
        f"over={a['over_gen_rate']:.3f} under={a['under_gen_rate']:.3f}"
    )


def print_report(short_res, long_res):
    short_agg, long_agg = aggregate(short_res), aggregate(long_res)
    print("\n" + "=" * 78)
    print("BASELINE  (freeze these numbers; diff against the post-training run)")
    print("-" * 78)
    print(_row("SHORT", short_agg))
    print(_row("LONG(8)", long_agg))
    print("-" * 78)
    print("SHORT broken down by length (this is where the fix should move):")
    for L, a in by_length(short_res).items():
        print(_row(f"  {L} digits", a))
    print("=" * 78 + "\n")


def dump_csv(results, path: Path):
    cols = [
        "image",
        "true",
        "pred",
        "status",
        "confidence",
        "is_exact",
        "digit_acc",
        "len_true",
        "len_pred",
        "len_delta",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in cols})
    logger.info(f"Wrote {len(results)} rows to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-csv", type=Path, required=True)
    ap.add_argument(
        "--short-img-dir", default="", help="image dir for short batch (auto-detected if omitted)"
    )
    ap.add_argument("--long-csv", type=Path, required=True)
    ap.add_argument(
        "--long-img-dir", default="", help="image dir for long batch (auto-detected if omitted)"
    )
    ap.add_argument(
        "--long-digits",
        type=int,
        default=8,
        help="keep only long rows with exactly this many digits",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_out"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    short_pairs = load_test_set(args.short_csv, args.short_img_dir)
    long_pairs = load_test_set(args.long_csv, args.long_img_dir, keep_digits={args.long_digits})

    logger.info("Evaluating SHORT batch (ocr_infer, no pipeline)...")
    short_res = evaluate(short_pairs)

    logger.info("Evaluating LONG batch (ocr_infer, no pipeline)...")
    long_res = evaluate(long_pairs)

    dump_csv(short_res, args.out_dir / "baseline_short.csv")
    dump_csv(long_res, args.out_dir / "baseline_long.csv")

    print_report(short_res, long_res)

    for name, res in (("SHORT", short_res), ("LONG(8)", long_res)):
        misses = [r for r in res if not r["is_exact"]]
        logger.info(f"{name} non-exact ({len(misses)}):")
        for m in misses:
            logger.info(
                f"   {m['image']} true={m['true']!r} pred={m['pred']!r} "
                f"len_delta={m['len_delta']:+d} status={m['status']}"
            )


if __name__ == "__main__":
    main()
