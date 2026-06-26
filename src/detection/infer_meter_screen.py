# YOLO #1
# Этот файл содержит:
#   1. infer()            — заглушка YOLO #1 инференса (используется в pipeline.py)
#
# Когда Role A напишет реальный код — заменить тела функций,
# интерфейс (входы/выходы) менять НЕ нужно.

import numpy as np

from src.utils.contracts import Detection
from src.utils.logger import logger


def infer(image: np.ndarray) -> list:
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
    logger.info(f"[STUB] yolo1 infer called, image shape: {image.shape}")
    h, w = image.shape[:2]

    return [
        Detection(
            bbox=[w // 4, h // 4, w * 3 // 4, h // 4, w * 3 // 4, h * 3 // 4, w // 4, h * 3 // 4],
            cls="meter",
            confidence=0.90,
        ),
        Detection(
            bbox=[w // 4, h // 4, w * 3 // 4, h // 4, w * 3 // 4, h * 3 // 4, w // 4, h * 3 // 4],
            cls="digital_display",
            confidence=0.90,
        ),
    ]
