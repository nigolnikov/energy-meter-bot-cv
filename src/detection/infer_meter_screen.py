# YOLO #1
# Этот файл содержит:
#   1. enhance_net_nopool — заглушка Zero-DCE++ сети (используется в preprocessing.py)
#   2. infer()            — заглушка YOLO #1 инференса (используется в pipeline.py)
#
# Когда Role A напишет реальный код — заменить тела классов и функций,
# интерфейс (входы/выходы) менять НЕ нужно.

import numpy as np
import torch
import torch.nn as nn

from src.utils.contracts import Detection
from src.utils.logger import logger


class enhance_net_nopool(nn.Module):
    """
    ЗАГЛУШКА Zero-DCE++ сети — Role A заменит на реальную архитектуру.

    Используется в: src/utils/preprocessing.py → ZeroDCEEnhancer.__init__()

    Контракт forward():
        Вход:  torch.Tensor shape (1, 3, H, W), float32, значения [0.0 .. 1.0]
        Выход: list где [0] — torch.Tensor той же формы (1, 3, H, W)

    Сейчас: возвращает вход без изменений (identity).
    Картинка не улучшается, но ZeroDCEEnhancer не падает с ошибкой.
    """

    def __init__(self, scale_factor: int = 1):
        super().__init__()
        # TODO Role A: здесь будут сверточные слои DCE-Net

    def forward(self, x: torch.Tensor) -> list:
        # TODO Role A: заменить на реальный проход через сеть
        return [x]


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
