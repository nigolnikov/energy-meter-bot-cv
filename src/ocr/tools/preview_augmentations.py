"""
Visual diagnostics for the digital-meter augmentation pipeline.

    PYTHONPATH=. python -m src.ocr.tools.preview_augmentations --mode stages \\
        --image data/ocr_analog_meter/images/test/img_213.png --samples 5

    PYTHONPATH=. python -m src.ocr.tools.preview_augmentations --mode grid \\
        --image data/crops/val/some_meter.png --samples 20

--mode stages: every stage of the real train pipeline in isolation (forced to
p=1.0), plus deliberately extreme "stress" tiers for the robustness eval.

  train_*   -- the real stages, one per file, each at p=1.0.

  stress_*  -- deliberately extreme tiers for the robustness eval.

--mode grid: N draws of the whole real pipeline at its real probabilities,
laid out as one numbered contact sheet, so the unreadable ones can be counted
by eye.
"""

import argparse
import copy
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from src.ocr.augmentations_digital import (
    EdgeVignette,
    LCDGlare,
    SealThreads,
    build_train_augmentations,
)

# --------------------------------------------------------------------------
# stages mode: every pipeline stage in isolation, plus stress tiers
# --------------------------------------------------------------------------


def stage_name(transform: A.BasicTransform) -> str:
    """A readable name for a stage. OneOf is named after its children."""
    if isinstance(transform, A.OneOf):
        children = "_or_".join(child.__class__.__name__ for child in transform.transforms)
        return f"oneof_{children}".lower()

    return transform.__class__.__name__.lower()


def build_stage_previews(seed: int | None) -> dict[str, A.Compose]:
    pipeline = build_train_augmentations(seed=seed)

    previews: dict[str, A.Compose] = {}

    for i, transform in enumerate(pipeline.transforms):
        stage = copy.deepcopy(transform)
        stage.p = 1.0

        previews[f"train_{i:02d}_{stage_name(stage)}"] = A.Compose([stage], seed=seed)

        if isinstance(stage, A.OneOf):
            for child in stage.transforms:
                branch = copy.deepcopy(child)
                branch.p = 1.0

                name = f"train_{i:02d}_{branch.__class__.__name__.lower()}"
                previews[name] = A.Compose([branch], seed=seed)

    return previews


def build_stress_transforms(seed: int | None) -> dict[str, A.Compose]:
    return {
        "stress_blur_strong": A.Compose(
            [
                A.MotionBlur(blur_limit=(30, 45), angle_range=(0, 360), p=1.0),
            ],
            seed=seed,
        ),
        "stress_downscale_strong": A.Compose(
            [
                A.Downscale(
                    scale_range=(0.12, 0.25),
                    interpolation_pair={
                        "downscale": cv2.INTER_NEAREST,
                        "upscale": cv2.INTER_NEAREST,
                    },
                    p=1.0,
                ),
            ],
            seed=seed,
        ),
        "stress_perspective_strong": A.Compose(
            [
                A.Perspective(
                    scale=(0.05, 0.075),
                    keep_size=True,
                    fit_output=False,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=1.0,
                ),
            ],
            seed=seed,
        ),
        "stress_threads_strong": A.Compose(
            [
                SealThreads(
                    num_threads=(3, 4),
                    alpha=(0.85, 1.0),
                    thickness_frac=(0.018, 0.030),
                    max_tilt_deg=70.0,
                    softness=(0.0, 0.3),
                    p=1.0,
                ),
            ],
            seed=seed,
        ),
        "stress_threads_glare": A.Compose(
            [
                SealThreads(
                    num_threads=(1, 2),
                    alpha=(0.5, 0.8),
                    thickness_frac=(0.006, 0.016),
                    softness=(0.2, 1.0),
                    p=1.0,
                ),
                LCDGlare(alpha_range=(0.35, 0.6), blur_range=(31, 91), p=1.0),
            ],
            seed=seed,
        ),
        "edge_vignette": A.Compose([EdgeVignette(p=1.0)]),
    }


def build_preview_transforms(seed: int | None) -> dict[str, A.Compose | None]:
    transforms: dict[str, A.Compose | None] = {"clean": None}

    transforms.update(build_stage_previews(seed))

    transforms["train_full_pipeline"] = build_train_augmentations(seed=seed)

    transforms.update(build_stress_transforms(seed))

    return transforms


def apply_transform(image: np.ndarray, transform: A.Compose | None) -> np.ndarray:
    if transform is None:
        return image

    return transform(image=image)["image"]


def save_image(image: np.ndarray, output_path: Path) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(output_path)


def run_stages(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_np = np.array(Image.open(args.image).convert("RGB"))

    transforms = build_preview_transforms(args.seed)

    for name, transform in transforms.items():
        n_samples = 1 if transform is None else args.samples

        for i in range(n_samples):
            print(f"Saving: {name}" + (f" [{i}]" if n_samples > 1 else ""))

            transformed = apply_transform(image_np, transform)

            suffix = f"_{i}" if n_samples > 1 else ""
            save_image(transformed, args.output_dir / f"{name}{suffix}.png")

    print()
    print(f"Saved all preview images to: {args.output_dir}")
    print("train_* mirrors build_train_augmentations(). stress_* is eval-only.")


# --------------------------------------------------------------------------
# grid mode: N draws of the whole pipeline as one contact sheet
# --------------------------------------------------------------------------


def contact_sheet(tiles: list[np.ndarray], cols: int, pad: int = 6) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)

    canvas_tiles = []

    for i, tile in enumerate(tiles):
        canvas = np.full((h, w, 3), 32, dtype=np.uint8)
        canvas[: tile.shape[0], : tile.shape[1]] = tile

        label = "clean" if i == 0 else str(i)
        cv2.putText(
            canvas, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            canvas, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 80), 2, cv2.LINE_AA
        )

        canvas_tiles.append(np.pad(canvas, ((pad, pad), (pad, pad), (0, 0)), constant_values=200))

    rows = []

    for start in range(0, len(canvas_tiles), cols):
        row = canvas_tiles[start : start + cols]

        while len(row) < cols:
            row.append(np.full_like(canvas_tiles[0], 200))

        rows.append(np.hstack(row))

    return np.vstack(rows)


def run_grid(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(args.image).convert("RGB"))

    transform = build_train_augmentations(seed=args.seed)

    tiles = [image]
    tiles += [transform(image=image)["image"] for _ in range(args.samples)]

    sheet = contact_sheet(tiles, cols=args.cols)

    Image.fromarray(sheet).save(args.output)

    print(f"{args.samples} draws + clean reference -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["stages", "grid"],
        required=True,
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to one input image")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=Path("reports/augmentation_preview"),
        type=Path,
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/augmentation_preview/train_pipeline_sheet.png"),
    )

    args = parser.parse_args()

    if args.samples is None:
        args.samples = 5 if args.mode == "stages" else 20

    return args


def main() -> None:
    args = parse_args()

    if args.mode == "stages":
        run_stages(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
