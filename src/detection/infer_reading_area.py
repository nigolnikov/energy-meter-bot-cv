# YOLO #2 — детекция области показаний на экране.
# Владелец: Role A.
#
# Этот файл содержит:
#   1. infer()        — заглушка YOLO #2 инференса (используется в pipeline.py)
#
# Когда Role A напишет реальный код — заменить тела функций,
# интерфейс (входы/выходы) менять НЕ нужно.

import numpy as np

from src.utils.contracts import Detection
from src.utils.logger import logger


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
    logger.info(f"[STUB] yolo2 infer called, image shape: {image.shape}")

    h, w = image.shape[:2]

    # Фейковый bbox — весь переданный crop
    return [
        Detection(bbox=[0, 0, w, h], cls="reading", confidence=0.85),
    ]
