import cv2
import numpy as np


def apply_clahe(image: np.array) -> np.ndarray:
    # 1. BGR -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 2. split channels
    L, a, b = cv2.split(lab)

    # 3. CLAHE for only L (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(L)

    # 4. merge back
    lab2 = cv2.merge((l2, a, b))

    # 5. back to BGR
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    return enhanced
