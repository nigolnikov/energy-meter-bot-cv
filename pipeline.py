import json

import cv2

from src.detection.infer_meter_screen import infer as yolo1_infer
from src.detection.infer_reading_area import infer as yolo2_infer
from src.ocr.infer_trocr import infer as ocr_infer
from src.utils.contracts import PipelineResult
from src.utils.preprocessing import apply_clahe, crop_bbox


def run_pipeline(image_path: str) -> PipelineResult:
    """
    Main function which will take the way to image, gives the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    detection_1 = yolo1_infer(image)  # List[Detection]

    screen_det = next((d for d in detection_1 if d.cls == "screen"), None)
    meter_det = next((d for d in detection_1 if d.cls == "meter"), None)

    if not screen_det:
        return PipelineResult(
            meter_box=[],
            screen_box=[],
            reading_box=[],
            raw_text="",
            value="",
            confidence=0.0,
            status="no_meter",
        )

    # Crop + Enhancement
    screen_crop = crop_bbox(image, screen_det.bbox)
    screen_enhanced = apply_clahe(screen_crop)

    # YOLO #2
    detection_2 = yolo2_infer(screen_enhanced)
    reading_det = next((d for d in detection_2 if d.cls == "reafing"), None)

    if not reading_det:
        return PipelineResult(
            meter_box=meter_det.bbox if meter_det else [],
            screen_box=screen_det.bbox,
            reading_box=[],
            raw_text="",
            value="",
            confidence=0.0,
            status="no_reading",
        )

    # OCR
    reading_crop = crop_bbox(screen_enhanced, reading_det.bbox)
    ocr_result = ocr_infer(reading_crop)

    # Post-processing
    clean_value = ocr_result.text.replace(" ", "").strip()

    final_confidence = min(screen_det.confidence, reading_det.confidence, ocr_result.confidence)

    status = "ok" if final_confidence >= 0.5 else "low_confidence"

    return PipelineResult(
        meter_box=meter_det.bbox if meter_det else [],
        screen_box=screen_det.bbox,
        raw_text=ocr_result.text,
        value=clean_value,
        confidence=round(final_confidence, 3),
        status=status,
    )


if __name__ == "__main__":
    result = run_pipeline("example.jpg")
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
