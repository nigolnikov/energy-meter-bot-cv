import albumentations as A
import cv2


def build_train_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.88, 1.12),
                translate_percent=(-0.04, 0.04),
                rotate=(-4, 4),
                shear=(-4, 4),
                p=0.45,
            ),
            A.OneOf(
                [
                    A.MotionBlur(
                        blur_limit=(15, 25),
                        angle_range=(0, 360),
                    ),
                    A.Defocus(radius=(6, 10), alias_blur=(0.2, 0.5)),
                    A.ZoomBlur(max_factor=(1.1, 1.25), step_factor=(0.01, 0.03)),
                    A.Downscale(
                        scale_range=(0.25, 0.4),
                        interpolation_pair={
                            "downscale": cv2.INTER_NEAREST,
                            "upscale": cv2.INTER_NEAREST,
                        },
                    ),
                ],
                p=0.45,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                    ),
                    A.RandomGamma(
                        gamma_limit=(70, 130),
                    ),
                ]
            ),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=4, p=0.35
            ),
        ]
    )
