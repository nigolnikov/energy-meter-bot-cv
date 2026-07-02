# YOLO #2 — детекция области показаний на экране.
# Владелец: Role A.
#
# Этот файл содержит:
#   1. infer()        — заглушка YOLO #2 инференса (используется в pipeline.py)
#
# Когда Role A напишет реальный код — заменить тела функций,
# интерфейс (входы/выходы) менять НЕ нужно.

import numpy as np
from ultralytics import YOLO

from src.utils.contracts import Detection
from src.utils.logger import logger

CLASS_NAMES = {
    0: "reading_area",
}

# CHANGE THIS LATER: Role A должен будет указать путь к своей модели.
MODEL_PATH = "runs/detect/runs/yolo/reading_area_yolo11s/weights/best.pt"

model = YOLO(MODEL_PATH)


def infer(image: np.ndarray) -> list:
    """
    ЗАГЛУШКА YOLO #2 — Role A заменит на реальный инференс.

    Используется в: pipeline.py → run_pipeline()

    Контракт:
        Вход:  np.ndarray (H, W, 3) BGR — crop экрана после CLAHE/Zero-DCE++
        Выход: List[Detection] — найденные области с цифрами показания

    Класс который возвращает реальная модель:
        "reading" — только цифры показания счётчика.
                    Модель должна игнорировать: дату, время, иконки, температуру.
        bbox в координатах переданного crop (не оригинала!)

    Сейчас: возвращает один фейковый Detection на весь crop.
    """
    logger.info(f"YOLO #2 infer called, image shape: {image.shape}")

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False,
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for bbox, cls_id, score in zip(boxes, classes, confidences, strict=False):
            detections.append(
                Detection(
                    bbox=bbox.reshape(-1).tolist(),
                    cls=CLASS_NAMES[int(cls_id)],
                    confidence=float(score),
                )
            )

    logger.info(f"YOLO #2 infer finished, found {len(detections)} detection(s)")

    return detections
