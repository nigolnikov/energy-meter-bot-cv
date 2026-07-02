import json

import cv2

from src.detection.infer_meter_screen import infer as yolo1_infer
from src.detection.infer_reading_area import infer as yolo2_infer
from src.ocr.infer_trocr import ocr_infer
from src.utils.contracts import PipelineResult
from src.utils.logger import logger
from src.utils.preprocessing import (
    apply_clahe,
    crop_bbox,
    crop_obb,
    obb_to_bbox,
    preprocess_for_ocr,
    validate_reading,
)
from src.utils.visualization import draw_pipeline_result, save_visualization


def run_pipeline(image_path: str) -> PipelineResult:
    """
    Main function which will take the way to image, gives the result.
    """
    # 1. Loading
    logger.info(f"Loading image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")
    logger.info(f"Image loaded: {image.shape[1]}x{image.shape[0]}")

    # 2. YOLO #1
    logger.info("Running YOLO #1 (meter + screen detection)...")
    detection_1 = yolo1_infer(image)  # List[Detection]
    logger.info(f"YOLO #1 found: {[d.cls for d in detection_1]}")

    meter_det = next((d for d in detection_1 if d.cls == "meter"), None)
    digital_det = next((d for d in detection_1 if d.cls == "digital_display"), None)
    analog_det = next((d for d in detection_1 if d.cls == "analog_register"), None)

    screen_det = digital_det or analog_det

    if not screen_det:
        logger.warning("Screen not found (neither digital_display nor analog_register)")
        return PipelineResult(
            meter_bbox=[],
            screen_bbox=[],
            screen_type="",
            reading_bbox=[],
            raw_text="",
            value="",
            confidence=0.0,
            status="no_screen",
        )
    logger.info(f"Using screen type: '{screen_det.cls}'")

    # 3. Crop + Enhancement
    screen_crop = crop_obb(image, screen_det.bbox)

    # 4. Enhancement
    logger.info("Applting CLAHE enhancement")
    screen_enhanced = apply_clahe(screen_crop)

    # 5. YOLO #2
    logger.info("Running YOLO #2 (reading area detection)...")
    detection_2 = yolo2_infer(screen_enhanced)
    logger.info(f"YOLO #2 found: {[d.cls for d in detection_2]}")

    reading_det = next((d for d in detection_2 if d.cls == "reading_area"), None)

    if not reading_det:
        logger.warning("Reading area not found — returning no_reading")
        return PipelineResult(
            meter_bbox=meter_det.bbox if meter_det else [],
            screen_bbox=obb_to_bbox(screen_det.bbox),
            screen_type=screen_det.cls,
            reading_bbox=[],
            raw_text="",
            value="",
            confidence=0.0,
            status="no_reading",
        )

    # 6. Crop показания + preprocessing
    reading_crop = crop_bbox(screen_enhanced, reading_det.bbox)
    reading_ready = preprocess_for_ocr(reading_crop)

    # 7. TrOCR
    logger.info("Running TrOCR...")
    ocr_result = ocr_infer(reading_ready)
    logger.info(f"TrOCR result: '{ocr_result.text}' (conf={ocr_result.confidence:.2f})")

    # 8. Post-processing
    clean_value = ocr_result.text.replace(" ", "").strip()

    screen_bbox = obb_to_bbox(screen_det.bbox)
    final_confidence = min(screen_det.confidence, reading_det.confidence, ocr_result.confidence)

    if not validate_reading(clean_value):
        return PipelineResult(
            meter_bbox=obb_to_bbox(meter_det.bbox) if meter_det else [],
            screen_bbox=screen_bbox,
            screen_type=screen_det.cls,
            reading_bbox=reading_det.bbox,
            raw_text=ocr_result.text,
            value=clean_value,
            confidence=round(final_confidence, 3),
            status="invalid_reading",
        )

    status = "ok" if final_confidence >= 0.5 else "low_confidence"

    logger.info(f"Pipeline done: value='{clean_value}', status='{status}'")

    # 9. Visualization - сохраняем фото с bbox

    screen_bbox = obb_to_bbox(screen_det.bbox)

    pipeline_result = PipelineResult(
        meter_bbox=obb_to_bbox(meter_det.bbox) if meter_det else [],
        screen_bbox=screen_bbox,
        screen_type=screen_det.cls,
        reading_bbox=reading_det.bbox,
        raw_text=ocr_result.text,
        value=clean_value,
        confidence=round(final_confidence, 3),
        status=status,
    )
    vis = draw_pipeline_result(image, pipeline_result)
    save_visualization(vis, "output.jpg")

    return pipeline_result


if __name__ == "__main__":
    import sys

    # Можно передать путь к фото аргументом: python pipeline.py my_photo.jpg
    image_path = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"

    result = run_pipeline(image_path)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
