import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.utils.contracts import OCRResult
from src.utils.logger import logger


def infer(
    image: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: torch.device,
    num_beams: int = 4,
) -> OCRResult:
    image = image.convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():
        output = model.generate(
            pixel_values,
            num_beams=num_beams,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    predicted_text = processor.batch_decode(
        output.sequences,
        skip_special_tokens=True,
    )[0].strip()

    transition_scores_kwargs = {
        "normalize_logits": True,
    }

    if num_beams > 1:
        transition_scores_kwargs["beam_indices"] = output.beam_indices

    transition_scores = model.compute_transition_scores(
        output.sequences,
        output.scores,
        **transition_scores_kwargs,
    )[0]

    generated_token_ids = output.sequences[0, 1:]

    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, list | tuple):
        eos_token_id = eos_token_id[0]

    eos_positions = (generated_token_ids == eos_token_id).nonzero()

    if len(eos_positions) > 0:
        valid_length = int(eos_positions[0]) + 1
    else:
        valid_length = generated_token_ids.shape[0]

    valid_log_probs = transition_scores[:valid_length]
    valid_probs = valid_log_probs.exp()

    if valid_length == 0:
        confidence = 0.0
    else:
        confidence = float(valid_log_probs.mean().exp())

    min_confidence = float(valid_probs.min()) if valid_length > 0 else 0.0

    logger.info(
        f"TrOCR prediction: {predicted_text!r}, "
        f"confidence={confidence:.4f}, "
        f"min_confidence={min_confidence:.4f}"
    )

    return OCRResult(
        text=predicted_text,
        confidence=confidence,
    )


def ocr_infer(image: np.ndarray) -> OCRResult:
    if image is None:
        raise ValueError("ocr_infer received image=None")

    logger.info(f"TrOCR infer called, image shape: {image.shape}")

    model_path = "model_best/trocr-meter-finetuned"

    if not hasattr(ocr_infer, "processor"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading TrOCR model from: {model_path}")
        logger.info(f"Using device: {device}")

        ocr_infer.processor = TrOCRProcessor.from_pretrained(model_path)
        ocr_infer.model = VisionEncoderDecoderModel.from_pretrained(model_path)

        ocr_infer.model.to(device)
        ocr_infer.model.eval()

        ocr_infer.device = device

    processor = ocr_infer.processor
    model = ocr_infer.model
    device = ocr_infer.device

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        pil_image = Image.fromarray(image).convert("RGB")

    elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
        pil_image = Image.fromarray(image).convert("RGB")

    else:
        raise ValueError(f"Unsupported image shape for OCR: {image.shape}")

    return infer(
        image=pil_image,
        processor=processor,
        model=model,
        device=device,
        num_beams=2,
    )
