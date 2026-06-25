# YOLO #2 — детекция области показаний на экране.
# Владелец: Role A.
#
# Этот файл содержит:
#   1. ObjectDetector — заглушка класса детектора (используется в preprocessing.py)
#   2. infer()        — заглушка YOLO #2 инференса (используется в pipeline.py)
#
# Когда Role A напишет реальный код — заменить тела классов и функций,
# интерфейс (входы/выходы) менять НЕ нужно.

import numpy as np

from src.utils.contracts import Detection
from src.utils.logger import logger


class ObjectDetector:
    """
    ЗАГЛУШКА детектора — Role A заменит на реальный YOLO детектор.

    Используется в: src/utils/preprocessing.py → PhotoMan(ObjectDetector)
    PhotoMan наследуется от этого класса и вызывает get_masked_screen().

    Контракт get_masked_screen():
        Вход:  нет (берёт self.source установленный в __init__)
        Выход: dict:
            {
                "bool": True,        # True если экран найден
                "img": np.ndarray    # crop экрана BGR shape (H, W, 3)
            }
            или
            {
                "bool": False,       # экран не найден
                "img": None
            }

    Сейчас: возвращает {"bool": False} — PhotoMan бросит ValueError.
    Role A заменит на реальный YOLO #1 инференс + crop экрана.
    """

    def __init__(self, source=None):
        """
        source — путь к фото (str) или np.ndarray изображение.
        Role A будет загружать YOLO модель и запускать инференс по source.
        """
        self.sourse = source
        logger.info(f"[STUB] ObjectDetector initialized, source={source}")

    def get_masked_screen(self) -> dict:
        """
        ЗАГЛУШКА. Role A заменит на реальную детекцию + crop экрана.

        Возвращает:
            "bool" — найден ли экран на фото
            "img"  — np.ndarray crop экрана в BGR (H, W, 3)
        """
        logger.warning("[STUB] get_masked_screen called — returning empty result")
        return {"bool": False, "img": None}


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
