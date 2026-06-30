# YOLO #1
# Этот файл содержит:
#   1. infer()            — заглушка YOLO #1 инференса (используется в pipeline.py)
#
# Когда Role A напишет реальный код — заменить тела функций,
# интерфейс (входы/выходы) менять НЕ нужно.

import numpy as np
from ultralytics import YOLO

from src.utils.contracts import Detection
from src.utils.logger import logger

CLASS_NAMES = {
    0: "meter",
    1: "digital_display",
    2: "analog_register",
}

MODEL_PATH = "runs/obb/runs/yolo_obb/meter_screen_yolo11s_obb-21/weights/best.pt"

model = YOLO(MODEL_PATH)


def infer(image: np.ndarray) -> list[Detection]:
    """
    ЗАГЛУШКА YOLO #1 — Role A заменит на реальный инференс.

    Контракт:
        Вход:  np.ndarray (H, W, 3) BGR
        Выход: List[Detection]

    Классы:
        "meter"            — корпус счётчика
        "digital_display"  — цифровой экран
        "analog_register"  — аналоговый барабан

    bbox: [x1, y1, x2, y2, x3, y3, x4, y4] — OBB (8 координат, 4 угла)
    """

    logger.info(f"YOLO #1 inference started, image shape: {image.shape}")

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False,
    )

    detections = []

    for result in results:
        if result.obb is None:
            continue

        boxes = result.obb.xyxyxyxy.cpu().numpy()
        classes = result.obb.cls.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()

        for bbox, cls_id, score in zip(
            boxes,
            classes,
            confidences,
            strict=False,
        ):
            detections.append(
                Detection(
                    bbox=bbox.reshape(-1).tolist(),
                    cls=CLASS_NAMES[int(cls_id)],
                    confidence=float(score),
                )
            )

    logger.info(f"number of detections found {len(detections)}")

    return detections

    # logger.info(f"[STUB] yolo1 infer called, image shape: {image.shape}")

    # h, w = image.shape[:2]

    # return [
    #     Detection(
    #         bbox=[w // 4, h // 4, w * 3 // 4, h // 4, w * 3 // 4, h * 3 // 4, w // 4, h * 3 // 4],
    #         cls="meter",
    #         confidence=0.90,
    #     ),
    #     Detection(
    #         bbox=[w // 4, h // 4, w * 3 // 4, h // 4, w * 3 // 4, h * 3 // 4, w // 4, h * 3 // 4],
    #         cls="digital_display",
    #         confidence=0.90,
    #     ),
    # ]
