"""
OBB → выровненный кроп для OCR (устойчивая версия, v6).

  1. Обход углов — по углу вокруг центроида (не sum/diff): не ломается
     при наклоне и не дублирует углы.

  2. Ориентация вывода — по ДЛИННОЙ стороне бокса = предполагаемая ось строки
     показаний. Register-бокс = рамка, ПОЛНОСТЬЮ включающая экран (digital) или
     аналоговое табло (analog).
     ДОПУЩЕНИЕ: бокс вытянут по ширине (шире, чем выше) -> длинная сторона = ось
     показаний, определяется независимо от наклона (корректно даже до 45°).

  3. crop_and_warp_obb возвращает (crop, ambiguous). ambiguous=True, когда бокс
     близок к квадрату (отношение сторон < ambiguity_ratio) — тогда «самая длинная»
     сторона определяется по сути шумом и допущение ненадёжно.
     ОГРАНИЧЕНИЕ флага: он ловит только «почти квадрат». Бокс, явно вытянутый
     в ВЫСОТУ (портретный экран), остаётся «вытянутым» и флагом НЕ помечается —
     отношением сторон этот случай не поймать (см. историю по v3).

  4. resolve_orientation:
       full_check=False (по умолчанию) — быстрый путь: только 180° по уверенности;
       full_check=True — полная проверка 0/90/180/270 с выбором по макс. уверенности OCR.
     Фича опциональна: включайте её для неоднозначных кропов (флаг ambiguous).

  5. В кроп подаём только register-боксы; meter не кропаем.
"""

import cv2
import numpy as np


def _ordered_corners(obb_points):
    """4 угла в согласованном обходе при любом повороте.
    obb_points: 8 значений [x1, y1, x2, y2, x3, y3, x4, y4] либо (4,2)."""
    pts = np.asarray(obb_points, dtype="float32").reshape(4, 2)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(angles)]


def crop_and_warp_obb(image, obb_points, ambiguity_ratio=1.3):
    """De-skew + выравнивание так, чтобы строка показаний была горизонтальной.
    Длинная сторона бокса трактуется как ось строки показаний.

    Возвращает (crop, ambiguous):
      crop      — выровненный кроп (или None для вырожденного бокса);
      ambiguous — True, если бокс почти квадратный (вытянутость < ambiguity_ratio)
                  и допущение «длинная сторона = ось строки» ненадёжно.

    Ориентация известна с точностью до 180° (либо до 90°, если ambiguous) —
    снимается в resolve_orientation."""
    quad = _ordered_corners(obb_points)
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]

    elong = max(edges) / (min(edges) + 1e-9)  # вытянутость, всегда >= 1
    ambiguous = elong < ambiguity_ratio  # почти квадрат -> длинная сторона ненадёжна

    start = int(np.argmax(edges))  # длинное ребро = предполагаемая ось строки
    src = np.roll(quad, -start, axis=0)

    W = int(max(edges[start], edges[(start + 2) % 4]))
    H = int(max(edges[(start + 1) % 4], edges[(start + 3) % 4]))
    if W < 2 or H < 2:
        return None, False  # вырожденный бокс

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (W, H)), ambiguous


def resolve_orientation(crop, ocr_fn, conf_threshold=0.80, full_check=False):
    """Определяет правильную ориентацию кропа через OCR.

    ocr_fn(img) -> (text: str, mean_confidence: float)

    full_check=False — быстрый путь (для уверенно-вытянутых боксов):
        один прогон OCR; перевёрнутый на 180° вариант пробуем только при
        уверенности ниже conf_threshold. На обычных кадрах OCR вызывается раз.

    full_check=True — полная проверка (для неоднозначных, почти квадратных):
        пробуем все 4 поворота (0/90/180/270) и берём максимум по уверенности.
        OCR вызывается 4 раза — оправдано на редких неоднозначных кропах.

    Примечание: некоторые цифры читаемы и вверх ногами (0/6/8/9); при
    необходимости усильте сравнение эвристикой (число разрядов, разделитель,
    отсутствие букв).
    """
    if full_check:
        candidates = [
            crop,
            cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop, cv2.ROTATE_180),
            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        best_text, best_img, best_conf = None, crop, -1.0
        for variant in candidates:
            t, c = ocr_fn(variant)
            if c > best_conf:
                best_text, best_img, best_conf = t, variant, c
        return best_text, best_img

    # Быстрый путь: только 180°, экономно (gated)
    text, conf = ocr_fn(crop)
    if conf >= conf_threshold:
        return text, crop  # уверенно -> второй прогон не нужен
    flipped = cv2.rotate(crop, cv2.ROTATE_180)
    flip_text, flip_conf = ocr_fn(flipped)
    if flip_conf > conf:
        return flip_text, flipped
    return text, crop


# --- Пример связки с Ultralytics ---
if __name__ == "__main__":
    from ultralytics import YOLO

    REGISTER_CLASSES = {"digital_display", "analog_register"}  # класс meter не кропаем
    ENABLE_FULL_ORIENTATION_CHECK = True  # вкл/выкл фичу полной проверки ориентации

    model = YOLO("runs/obb/runs/yolo_obb/meter_screen_yolo11s_obb-21/weights/best.pt")
    img = cv2.imread("meter_photo.jpg")

    names = model.names  # {0: 'digital_display', ...}

    for result in model(img):
        if result.obb is None:
            continue
        polys = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        clss = result.obb.cls.cpu().numpy().astype(int)
        for poly, c in zip(polys, clss, strict=False):
            cls_name = names[int(c)]
            if cls_name not in REGISTER_CLASSES:
                continue  # сам прибор не кропаем
            crop, ambiguous = crop_and_warp_obb(img, poly.flatten())
            if crop is None:
                continue
            # Неоднозначные (почти квадрат) -> полная проверка, если фича включена;
            # остальные -> быстрый путь (180° по уверенности).
            full = ENABLE_FULL_ORIENTATION_CHECK and ambiguous
            # text, oriented = resolve_orientation(crop, ocr_for(cls_name), full_check=full)
            cv2.imwrite(f"crop_{cls_name}.jpg", crop)
