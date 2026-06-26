import cv2
import numpy as np

from src.utils.contracts import Detection, PipelineResult

COLORS = {
    "meter": (0, 255, 0),  # green
    "screen": (255, 165, 0),  # orange
    "reading": (0, 0, 255),  # red
}


def draw_bbox(image, bbox, label, confidence=None):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = label
    if confidence is not None:
        text += f" {confidence:.2f}"

    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    result = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = COLORS.get(det.cls, (255, 255, 255))

        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=2)

        label = f"{det.cls}: {det.confidence:.3f}"
        cv2.putText(
            result,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )
    return result


def draw_pipeline_result(image: np.ndarray, result: PipelineResult) -> np.ndarray:
    vis = image.copy()

    boxes = [
        (result.meter_box, "meter", COLORS["meter"]),
        (result.screen_box, "screen", COLORS["screen"]),
        (result.reading_box, "reading", COLORS["reading"]),
    ]

    for bbox, label, color in boxes:
        if bbox:  # if box is not empty
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    status_text = f"Value: {result.value} ({result.status})"
    cv2.putText(
        vis,
        status_text,
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return vis


# save the visualization
def save_visualization(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)
    print(f"Saved: {path}")
