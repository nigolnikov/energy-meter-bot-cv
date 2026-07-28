import json

import cv2
import numpy as np

from src.detection.infer_meter_screen import infer as yolo1_infer
from src.detection.infer_reading_area import infer as yolo2_infer
from src.detection.obb_crop_robust import crop_and_warp_obb
from src.ocr.infer_trocr import ocr_infer
from src.utils.contracts import PipelineResult
from src.utils.logger import logger
from src.utils.preprocessing import (
    apply_enhancement,
    crop_bbox,
    crop_obb,
    obb_to_bbox,
    preprocess_for_ocr,
    reading_to_original,
    validate_reading,
)
from src.utils.visualization import draw_pipeline_result, save_visualization


def run_pipeline(
    image_path: str, enhancement: str = "none", crop_mode: str = "warp"
) -> PipelineResult:
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

    if meter_det is None:
        logger.warning("Meter not found")
        return PipelineResult(
            meter_bbox=[],
            screen_bbox=[],
            screen_type="",
            reading_bbox=[],
            raw_text="",
            value="",
            confidence=0.0,
            status="no_meter",
        )

    if screen_det is None:
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

    # 3. Crop screen
    if crop_mode == "warp":
        screen_crop, _, warp_M = crop_and_warp_obb(image, screen_det.bbox, return_matrix=True)
        if screen_crop is None:
            logger.warning("Screen box degenerate — cannot warp")
            return PipelineResult(
                meter_bbox=obb_to_bbox(meter_det.bbox) if meter_det else [],
                screen_bbox=obb_to_bbox(screen_det.bbox),
                screen_type=screen_det.cls,
                reading_bbox=[],
                raw_text="",
                value="",
                confidence=0.0,
                status="no_screen",
            )
    else:
        screen_crop = crop_obb(image, screen_det.bbox)
        warp_M = None

    # 4. Enhancement
    logger.info(f"Applying enhancement: {enhancement}")
    screen_enhanced = apply_enhancement(screen_crop, enhancement)

    # 5. YOLO #2
    logger.info("Running YOLO #2 (reading area detection)...")
    detection_2 = yolo2_infer(screen_enhanced)
    logger.info(f"YOLO #2 found: {[d.cls for d in detection_2]}")

    reading_det = next((d for d in detection_2 if d.cls == "reading_area"), None)

    if not reading_det:
        logger.warning("Reading area not found — returning no_reading")
        return PipelineResult(
            meter_bbox=obb_to_bbox(meter_det.bbox) if meter_det else [],
            screen_bbox=obb_to_bbox(screen_det.bbox),
            screen_type=screen_det.cls,
            reading_bbox=[],
            raw_text="",
            value="",
            confidence=0.0,
            status="no_reading",
        )

    screen_bbox = obb_to_bbox(screen_det.bbox)
    if crop_mode == "warp":
        rx1, ry1, rx2, ry2 = reading_det.bbox[:4]
        corners = np.array(
            [[[rx1, ry1]], [[rx2, ry1]], [[rx2, ry2]], [[rx1, ry2]]], dtype="float32"
        )
        orig = cv2.perspectiveTransform(corners, np.linalg.inv(warp_M)).reshape(-1, 2)
        reading_abs = [
            int(orig[:, 0].min()),
            int(orig[:, 1].min()),
            int(orig[:, 0].max()),
            int(orig[:, 1].max()),
        ]
        reading_quad = orig.reshape(-1).tolist()  # 8 координат — повёрнутая рамка показания
    else:
        reading_abs = reading_to_original(reading_det.bbox, screen_bbox)
        reading_quad = None

    # 6. Crop показания + preprocessing
    reading_crop = crop_bbox(screen_enhanced, reading_det.bbox)

    reading_ready = preprocess_for_ocr(reading_crop)
    reading_ready = cv2.cvtColor(reading_ready, cv2.COLOR_BGR2GRAY)

    # 7. TrOCR
    if screen_det.cls == "analog_register":
        ocr_model_path = "models/trocr-meter-analog"
    else:
        ocr_model_path = "models/trocr-meter-finetuned"

    logger.info("Running TrOCR...")
    ocr_result = ocr_infer(reading_ready, model_path=ocr_model_path)
    logger.info(f"TrOCR result: '{ocr_result.text}' (conf={ocr_result.confidence:.2f})")

    # 8. Post-processing
    clean_value = ocr_result.text.replace(" ", "").strip()
    final_confidence = min(screen_det.confidence, reading_det.confidence, ocr_result.confidence)

    if not validate_reading(clean_value):
        return PipelineResult(
            meter_bbox=obb_to_bbox(meter_det.bbox) if meter_det else [],
            screen_bbox=screen_bbox,
            screen_type=screen_det.cls,
            reading_bbox=reading_abs,
            raw_text=ocr_result.text,
            value=clean_value,
            confidence=round(final_confidence, 3),
            status="invalid_reading",
        )

    status = "ok" if final_confidence >= 0.5 else "low_confidence"

    logger.info(f"Pipeline done: value='{clean_value}', status='{status}'")

    # 9. Visualization - сохраняем фото с bbox

    pipeline_result = PipelineResult(
        meter_bbox=obb_to_bbox(meter_det.bbox) if meter_det else [],
        screen_bbox=screen_bbox,
        screen_type=screen_det.cls,
        reading_bbox=reading_abs,
        raw_text=ocr_result.text,
        value=clean_value,
        confidence=round(final_confidence, 3),
        status=status,
    )

    vis = draw_pipeline_result(
        image,
        pipeline_result,
        meter_obb=meter_det.bbox if meter_det else None,  # 8 коорд OBB счётчика
        screen_obb=screen_det.bbox,  # 8 коорд OBB экрана
        reading_obb=reading_quad,  # 8 коорд повёрнутой рамки показания
    )

    save_visualization(vis, "output.jpg")

    return pipeline_result


if __name__ == "__main__":
    import sys

    # Можно передать путь к фото аргументом: python pipeline.py my_photo.jpg
    image_path = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"

    result = run_pipeline(image_path)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
