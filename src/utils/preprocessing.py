import cv2
import numpy as np


def apply_clahe(image: np.array) -> np.ndarray:
    # 1. BGR -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 2. split channels
    L, a, b = cv2.split(lab)

    # 3. CLAHE for only L (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(L)

    # 4. merge back
    lab_enhanced = cv2.merge((l_enhanced, a, b))

    # 5. back to BGR
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


# Cut region of bbox from image
def crop_bbox(image: np.ndarray, bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]
