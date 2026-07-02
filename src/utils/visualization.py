import cv2
import numpy as np

from src.utils.contracts import Detection, PipelineResult

COLORS = {
    "meter": (0, 255, 0),  # green
    "digital_display": (255, 165, 0),  # orange
    "analog_register": (255, 165, 0),  # orange
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
        color = COLORS.get(det.cls, (255, 255, 255))
        label = f"{det.cls}: {det.confidence:.2f}"

        if len(det.bbox) == 8:
            # OBB - 8 координат (YOLO #1)
            pts = np.array(det.bbox, dtype=np.float32).reshape(4, 2).astype(np.int32)
            cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(
                result, label, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        else:
            # Обычный bbox — 4 координаты (YOLO #2)
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return result


def draw_pipeline_result(image: np.ndarray, result: PipelineResult) -> np.ndarray:
    vis = image.copy()

    boxes = [
        (result.meter_bbox, "meter", COLORS["meter"]),
        (result.screen_bbox, result.screen_type, COLORS.get(result.screen_type, (255, 165, 0))),
        (result.reading_bbox, "reading", COLORS["reading"]),
    ]

    for bbox, label, color in boxes:
        if not bbox:  # if box is not empty
            continue
        if len(bbox) == 8:
            pts = np.array(bbox, dtype=np.float32).reshape(4, 2).astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(
                vis, label, (pts[0][0], pts[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        else:
            # Обычный bbox
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(
        vis,
        f"Value: {result.value} ({result.status})",
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
