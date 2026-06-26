# TrOCR — распознавание цифр показания счётчика.
# Владелец: Role B.
#
# Когда Role B напишет реальный код — заменить тело infer(),
# интерфейс (входы/выходы) менять НЕ нужно.

import numpy as np

from src.utils.contracts import OCRResult
from src.utils.logger import logger


def infer(image: np.ndarray) -> OCRResult:
    """
    ЗАГЛУШКА TrOCR — Role B заменит на реальный инференс.

    Используется в: pipeline.py → run_pipeline()

    Контракт:
        Вход:  np.ndarray (H, W) или (H, W, 3) — crop области показания
               после preprocessing (CLAHE + sharpen + denoise)
        Выход: OCRResult:
            text       (str)   — сырой текст как увидела модель, например "0012 3"
                                 НЕ чистить пробелы и мусор здесь —
                                 post-processing делается в pipeline.py
            confidence (float) — уверенность модели [0.0 .. 1.0]

    Сейчас: возвращает фейковый текст "00123" с confidence 0.88.
    Pipeline проходит до конца без ошибок.
    """

    logger.info(f"[STUB] TrOCR infer called, image shape: {image.shape}")

    # TODO Role B: загрузить TrOCR модель, распознать текст, вернуть OCRResult
    return OCRResult(text="00123", confidence=0.88)
