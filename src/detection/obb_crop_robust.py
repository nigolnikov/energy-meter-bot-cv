"""
OBB → выровненный кроп для OCR (устойчивая версия, v7).

  1. Обход углов — по углу вокруг центроида (не sum/diff): не ломается
     при наклоне и не дублирует углы.

  2. КАНОНИЗАЦИЯ НАПРАВЛЕНИЯ ЧТЕНИЯ (fix v7). У прямоугольника ДВА длинных
     ребра, и геометрия их не различает (180°-симметрия). В v2–v6 между ними
     выбирал argmax: при точном равенстве длин (боксы, декодированные из
     cx,cy,w,h,θ) — по тому, с какого угла стартовала угловая сортировка
     (старт скачет при наклоне бокса), при ручной разметке — по субпиксельному
     шуму длин (≈50/50). Итог: часть кропов выходила вверх ногами даже на
     ровных фото. Теперь длинное ребро канонизировано: читаем слева направо
     (dx > 0). Для фото с наклоном < 90° (наш профиль: ≤45°) кроп
     детерминированно «вверх головой»; вверх ногами остаются ТОЛЬКО реально
     перевёрнутые фото — их снимает resolve_orientation (gated), как и задумано.

  3. Register-бокс = рамка, ПОЛНОСТЬЮ включающая экран (digital) или аналоговое
     табло (analog). ДОПУЩЕНИЕ: бокс вытянут по ширине -> длинная сторона = ось
     показаний. crop_and_warp_obb возвращает (crop, ambiguous): ambiguous=True,
     когда бокс близок к квадрату (вытянутость < ambiguity_ratio) — «самая
     длинная» сторона определяется шумом, допущение ненадёжно.
     ОГРАНИЧЕНИЕ флага: ловит только «почти квадрат»; бокс, явно вытянутый
     в ВЫСОТУ (портретный многострочный LCD), флагом не помечается.

  4. resolve_orientation:
       full_check=False (по умолчанию) — быстрый путь: только 180° по уверенности;
       full_check=True — полная проверка 0/90/180/270 по макс. уверенности OCR.
     Включайте full_check для неоднозначных кропов (флаг ambiguous).

  5. В кроп подаём только register-боксы; meter не кропаем.
"""

import cv2
import numpy as np


def _ordered_corners(obb_points):
    """4 угла в согласованном циклическом обходе при любом повороте.
    obb_points: 8 значений [x1, y1, x2, y2, x3, y3, x4, y4] либо (4,2)."""
    pts = np.asarray(obb_points, dtype="float32").reshape(4, 2)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(angles)]


def crop_and_warp_obb(image, obb_points, ambiguity_ratio=1.3, return_matrix=False):
    """De-skew + выравнивание так, чтобы строка показаний была горизонтальной.
    Длинная сторона бокса трактуется как ось строки показаний; направление
    чтения канонизировано (слева направо), поэтому для фото с наклоном < 90°
    ориентация детерминированно правильная. Реально перевёрнутые (≈180°) фото
    дают перевёрнутый кроп — снимается в resolve_orientation.

    Возвращает (crop, ambiguous):
      crop      — выровненный кроп (или None для вырожденного бокса);
      ambiguous — True, если бокс почти квадратный (вытянутость < ambiguity_ratio)
                  и допущение «длинная сторона = ось строки» ненадёжно."""
    quad = _ordered_corners(obb_points)
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]

    # Площадь (шнуровка): отсекает вырожденные/коллинеарные боксы, которые
    # проскакивают проверку W/H с ненулевыми длинами рёбер и дали бы мусорный warp.
    area = 0.5 * abs(
        np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))
    )
    if area < 4:  # меньше ~2x2 px — не объект
        return (None, False, None) if return_matrix else (None, False)

    elong = max(edges) / (min(edges) + 1e-9)  # вытянутость, всегда >= 1
    ambiguous = elong < ambiguity_ratio  # почти квадрат -> длинная сторона ненадёжна

    start = int(np.argmax(edges))  # длинное ребро = предполагаемая ось строки
    src = np.roll(quad, -start, axis=0)

    # Канонизация направления чтения (fix v7): из двух длинных рёбер берём то,
    # которое идёт слева направо (dx > 0). Без этого выбор между ними произволен
    # (шум длин / старт сортировки) -> случайные перевороты на 180°.
    d = src[1] - src[0]
    if d[0] < 0 or (d[0] == 0 and d[1] < 0):
        src = np.roll(src, -2, axis=0)  # противоположное длинное ребро

    W = int(max(edges[start], edges[(start + 2) % 4]))
    H = int(max(edges[(start + 1) % 4], edges[(start + 3) % 4]))
    if W < 2 or H < 2:
        return (None, False, None) if return_matrix else (None, False)  # вырожденный бокс

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(image, M, (W, H))
    return (crop, ambiguous, M) if return_matrix else (crop, ambiguous)


def resolve_orientation(crop, ocr_fn, conf_threshold=0.80, full_check=False):
    """Определяет правильную ориентацию кропа через OCR.

    ocr_fn(img) -> (text: str, mean_confidence: float)

    full_check=False — быстрый путь (для уверенно-вытянутых боксов):
        один прогон OCR; перевёрнутый на 180° вариант пробуем только при
        уверенности ниже conf_threshold. После фикса v7 сюда попадают почти
        исключительно реально перевёрнутые фото — редкость по профилю данных.

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

    model = YOLO("runs/obb/runs/yolo_obb/meter_screen_yolo11s_obb-32/weights/best.pt")
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
