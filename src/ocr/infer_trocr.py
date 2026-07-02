# TrOCR — inference for meter reading OCR.
# Responsible only for loading the model and predicting text from images.


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
    max_new_tokens: int = 14,
) -> OCRResult:
    image = image.convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return OCRResult(
        text=text.strip(),
        confidence=0.88,  # placeholder for now
    )


def ocr_infer(image: np.ndarray) -> OCRResult:
    """
    TrOCR inference for one cropped meter-reading image.

    Input:
        image: np.ndarray
            Shape (H, W) or (H, W, 3)

    Output:
        OCRResult:
            text: raw model prediction
            confidence: 0.0 placeholder, because confidence is not used
    """

    logger.info(f"TrOCR infer called, image shape: {image.shape}")

    MODEL_PATH = "models/trocr-meter-finetuned"

    if not hasattr(ocr_infer, "processor"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading TrOCR model from: {MODEL_PATH}")
        logger.info(f"Using device: {device}")

        ocr_infer.processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
        ocr_infer.model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

        ocr_infer.model.to(device)
        ocr_infer.model.eval()

        ocr_infer.device = device

    processor = ocr_infer.processor
    model = ocr_infer.model
    device = ocr_infer.device

    if image is None:
        raise ValueError("ocr_infer received image=None")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        pil_image = Image.fromarray(image).convert("RGB")

    elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
        pil_image = Image.fromarray(image).convert("RGB")

    else:
        raise ValueError(f"Unsupported image shape for OCR: {image.shape}")

    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=10,
            num_beams=4,
            early_stopping=True,
        )

    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    logger.info(f"TrOCR prediction: {predicted_text!r}")

    return OCRResult(
        text=predicted_text,
        confidence=0.88,
    )
