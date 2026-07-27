import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.ocr.text import decode_label


def load_images(images_dir: Path, limit: int) -> list[Image.Image]:
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    return [Image.open(paths[i % len(paths)]).convert("RGB") for i in range(limit)]


@torch.no_grad()
def benchmark(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    images: list[Image.Image],
    device: str,
    num_beams: int,
    warmup: int,
) -> dict[str, float]:
    model = model.to(device).eval()
    model.generation_config.num_beams = num_beams

    is_cuda = device.startswith("cuda")

    def sync() -> None:
        if is_cuda:
            torch.cuda.synchronize()

    # Warmup: CUDA context, cuDNN autotune, allocator, lazy CPU init.
    for image in images[:warmup]:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        model.generate(pixel_values)

    sync()

    preprocess_ms: list[float] = []
    generate_ms: list[float] = []
    total_ms: list[float] = []
    n_tokens: list[int] = []

    for image in images[warmup:]:
        start = time.perf_counter()

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        sync()
        after_preprocess = time.perf_counter()

        sequences = model.generate(pixel_values)
        sync()
        end = time.perf_counter()

        preprocess_ms.append((after_preprocess - start) * 1000)
        generate_ms.append((end - after_preprocess) * 1000)
        total_ms.append((end - start) * 1000)
        n_tokens.append(int(sequences.shape[1]))

    text = decode_label(processor.batch_decode(sequences, skip_special_tokens=True)[0])

    return {
        "device": device,
        "num_beams": num_beams,
        "n": len(total_ms),
        "preprocess_median_ms": statistics.median(preprocess_ms),
        "generate_median_ms": statistics.median(generate_ms),
        "total_median_ms": statistics.median(total_ms),
        "total_p90_ms": float(np.percentile(total_ms, 90)),
        "total_min_ms": min(total_ms),
        "throughput_img_s": 1000.0 / statistics.median(total_ms),
        "mean_tokens": statistics.mean(n_tokens),
        "sample_prediction": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OCR latency per cropped image.")
    parser.add_argument("--model", required=True, type=Path, help="Trained model dir.")
    parser.add_argument("--images", required=True, type=Path, help="Dir of cropped images.")
    parser.add_argument("--device", default="both", choices=["cpu", "cuda", "both"])
    parser.add_argument("--num-beams", type=int, nargs="+", default=[4])
    parser.add_argument("--runs", type=int, default=30, help="Timed crops per configuration.")
    parser.add_argument("--warmup", type=int, default=5, help="Discarded crops before timing.")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="torch.set_num_threads. Default: all cores. Set to 1 for a per-core number.",
    )
    args = parser.parse_args()

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)

    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]

    if "cuda" in devices and not torch.cuda.is_available():
        print("CUDA not available -- skipping the GPU rows.")
        devices = [d for d in devices if d != "cuda"]

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, use_safetensors=True)

    images = load_images(args.images, args.runs + args.warmup)

    print(f"Model:  {args.model}")
    print(
        f"CPU:    {platform.processor() or platform.machine()}, "
        f"{torch.get_num_threads()} threads"
    )

    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    print(f"Crops:  {args.runs} timed, {args.warmup} warmup (discarded)")
    print()

    header = (
        f"{'device':<7}{'beams':>6}{'preproc':>10}{'generate':>10}"
        f"{'total':>9}{'p90':>9}{'img/s':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for device in devices:
        for num_beams in args.num_beams:
            r = benchmark(model, processor, images, device, num_beams, args.warmup)
            results.append(r)

            print(
                f"{r['device']:<7}{r['num_beams']:>6}"
                f"{r['preprocess_median_ms']:>9.1f}m"
                f"{r['generate_median_ms']:>9.1f}m"
                f"{r['total_median_ms']:>8.1f}m"
                f"{r['total_p90_ms']:>8.1f}m"
                f"{r['throughput_img_s']:>8.1f}"
            )

    print()
    print(
        "Median ms per crop. Sanity check on the last prediction: "
        f"{results[-1]['sample_prediction']!r}"
    )


if __name__ == "__main__":
    main()
