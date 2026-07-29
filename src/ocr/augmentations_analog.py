import albumentations as A
import cv2


def build_train_augmentations(seed: int | None = None) -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.02, 0.02),
                rotate=(-3, 3),
                shear=(-2, 2),
                border_mode=cv2.BORDER_REPLICATE,
                p=0.35,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(
                        blur_limit=(3, 5),
                    ),
                    A.MotionBlur(
                        blur_limit=(3, 5),
                    ),
                    A.Defocus(
                        radius=(1, 2),
                        alias_blur=(0.1, 0.25),
                    ),
                ],
                p=0.20,
            ),
            A.Downscale(
                scale_range=(0.75, 0.95),
                interpolation_pair={
                    "downscale": cv2.INTER_AREA,
                    "upscale": cv2.INTER_LINEAR,
                },
                p=0.20,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.25,
            ),
            A.RandomGamma(
                gamma_limit=(90, 110),
                p=0.15,
            ),
        ],
        seed=seed,
    )
