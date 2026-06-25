import os

import cv2
import numpy as np
import torch

from src.detection.infer_meter_screen import enhance_net_nopool
from src.detection.infer_reading_area import ObjectDetector
from src.utils.logger import logger


class ZeroDCEEnhancer:
    def __init__(self, weights_path: str = None):
        if weights_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(base, "snapshots_Zero_DCE++", "Epoch99.pth")

        logger.info(f"Loading Zero-DCE++ weights from {weights_path}")
        self.model = enhance_net_nopool(scale_factor=1)  # <-- вот это
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        logger.info("Zero-DCE++ model loaded successfully")

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]

        # Анализ яркости
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mean_v = hsv[:, :, 2].mean()
        logger.info(f"Enhancing image: {w}x{h}, mean_brightness={mean_v:.0f}")

        if mean_v > 180:
            # Яркое фото — только лёгкий CLAHE на канал яркости
            logger.info("Bright image — skipping Zero-DCE++, applying CLAHE only")
            hsv = hsv.astype(np.float32)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2].astype(np.uint8))
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Тёмное/нормальное — Zero-DCE++
        img = image_bgr[:, :, ::-1].astype("float32") / 255.0
        tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            result = self.model(tensor)
            enhanced = result[0]

        output = enhanced.squeeze(0).permute(1, 2, 0).contiguous().numpy()
        output = (output * 255).clip(0, 255).astype("uint8")

        logger.info("Enhancement complete")
        return output[:, :, ::-1]  # RGB -> BGR


class PhotoMan(ObjectDetector):
    """
    Улучшает изображение экрана для последующего OCR.
    """

    def __init__(self, source, zero_dce: ZeroDCEEnhancer = None) -> None:
        self.source = source
        self.zero_dce = zero_dce or ZeroDCEEnhancer()
        super().__init__(source=self.source)
        logger.info("PhotoMan initialized")

    def preprocess_image(self) -> np.ndarray:
        data = self.get_masked_screen()
        if not data or not data["bool"]:
            logger.warning("Failed to get screen image")
            raise ValueError("Не удалось получить изображение экрана.")

        crop = data["img"]
        logger.info(f"Screen detected: {crop.shape[1]}x{crop.shape[0]}")

        enhanced = self.zero_dce.enhance(crop)

        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)

        blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

        logger.info("Preprocessing complete")
        return sharpened


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """
    Baseline CLAHE enhancement. Используется в pipeline.py вместо Zero-DCE++
    пока Role A не предоставит обученные веса модели.

    Логика:
        mean brightness > 180 → clipLimit=1.0 (лёгкий CLAHE)
        иначе                 → clipLimit=2.0 (стандартный CLAHE)

    CLAHE применяется только к каналу яркости (L в LAB пространстве),
    цвет не меняется.

    Вход:  np.ndarray (H, W, 3) BGR
    Выход: np.ndarray (H, W, 3) BGR улучшенное
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_v = hsv[:, :, 2].mean()
    logger.info(f"CLAHE: image brightness={mean_v:.0f}")

    clip_limit = 1.0 if mean_v > 180 else 2.0

    # LAB: L — яркость, A и B — цвет
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(L)

    result = cv2.cvtColor(cv2.merge([L_enhanced, A, B]), cv2.COLOR_LAB2BGR)
    logger.info("CLAHE complete")
    return result


def denoise(image_bgr: np.ndarray) -> np.ndarray:
    """
    Медианный фильтр — убирает шум (соль-перец артефакты).
    Вход/выход: np.ndarray (H, W, 3) BGR
    """
    return cv2.medianBlur(image_bgr, 3)


def sharpen(image_bgr: np.ndarray) -> np.ndarray:
    """
    Unsharp masking — повышает резкость краёв цифр.
    Помогает TrOCR лучше распознавать символы.
    Вход/выход: np.ndarray (H, W, 3) BGR
    """
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(image_bgr, 1.5, blurred, -0.5, 0)


def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """
    Полный preprocessing перед подачей в TrOCR (Role B).
    Вызывается в pipeline.py после crop области показания (YOLO #2).

    Порядок:
        1. CLAHE   — улучшение контраста
        2. denoise — убрать шум
        3. sharpen — повысить резкость

    Вход:  np.ndarray (H, W, 3) BGR — crop области цифр
    Выход: np.ndarray (H, W, 3) BGR — готово для TrOCR
    """
    enhanced = apply_clahe(image_bgr)
    denoised = denoise(enhanced)
    sharpened = sharpen(denoised)
    logger.info("preprocess_for_ocr complete")
    return sharpened


def crop_bbox(image: np.ndarray, bbox: list) -> np.ndarray:
    """
    Вырезает область из изображения по координатам bbox.

    Используется в pipeline.py:
        - после YOLO #1 для получения crop экрана
        - после YOLO #2 для получения crop области показания

    bbox:  [x1, y1, x2, y2] в пикселях переданного image
    Вход:  np.ndarray (H, W, 3)
    Выход: np.ndarray (h, w, 3) — вырезанная область
    """
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    logger.info(f"Cropped: {crop.shape[1]}x{crop.shape[0]}")
    return crop


def crop_obb(image: np.ndarray, bbox: list) -> np.ndarray:
    """
    Вырезает область из OBB bbox (8 координат).
    Берёт min/max по X и Y — простой axis-aligned crop.

    bbox: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    xs = bbox[0::2]  # x1, x2, x3, x4
    ys = bbox[1::2]  # y1, y2, y3, y4
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    crop = image[y1:y2, x1:x2]
    logger.info(f"OBB crop: {crop.shape[1]}x{crop.shape[0]}")
    return crop
