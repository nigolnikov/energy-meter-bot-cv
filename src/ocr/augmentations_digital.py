import math
import random
from collections.abc import Sequence

import albumentations as A
import cv2
import numpy as np

THREAD_PALETTE: Sequence[tuple[int, int, int]] = (
    (16, 16, 18),  # black
    (30, 30, 33),  # near-black
    (48, 46, 50),  # dark grey (dusty wire)
    (66, 64, 68),  # grey (thin strand, partly lit)
)

THREAD_WEIGHTS: Sequence[float] = (0.4, 0.32, 0.2, 0.08)


def _thread_polyline(
    h: int,
    w: int,
    rng: random.Random,
    max_tilt_deg: float,
    n_pts: int = 96,
) -> np.ndarray:
    theta = math.radians(rng.uniform(-max_tilt_deg, max_tilt_deg))
    direction = np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    center = np.array([rng.uniform(-0.15, 1.15) * w, 0.5 * h], dtype=np.float32)
    length = 1.8 * (h + w)

    t = np.linspace(-0.5, 0.5, n_pts, dtype=np.float32)[:, None]
    pts = center[None, :] + t * length * direction[None, :]

    bow = rng.uniform(-0.03, 0.03) * h * (1.0 - (2.0 * t) ** 2)
    wobble = (
        rng.uniform(0.004, 0.018)
        * h
        * np.sin(2 * math.pi * rng.uniform(0.6, 2.2) * t + rng.uniform(0, 2 * math.pi))
    )

    return pts + (bow + wobble) * normal[None, :]


def _stroke_mask(
    h: int,
    w: int,
    pts: np.ndarray,
    thickness: float,
    rng: random.Random,
    twist: bool,
    ss: int = 2,
) -> np.ndarray:
    mask = np.zeros((h * ss, w * ss), dtype=np.uint8)
    px = np.rint(pts * ss).astype(np.int32)
    thick = max(1, int(round(thickness * ss)))

    period = rng.uniform(2.5, 7.0) * max(thickness, 1.0)
    phase = rng.uniform(0, 2 * math.pi)
    depth = rng.uniform(0.25, 0.5) if twist else 0.0

    for i in range(len(px) - 1):
        if twist:
            s = float(np.linalg.norm(pts[i] - pts[0]))
            alpha = 1.0 - depth * (0.5 + 0.5 * math.sin(2 * math.pi * s / period + phase))
        else:
            alpha = 1.0

        cv2.line(
            mask,
            tuple(px[i]),
            tuple(px[i + 1]),
            int(round(255 * alpha)),
            thick,
            lineType=cv2.LINE_AA,
        )

    small = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)

    return small.astype(np.float32) / 255.0


def draw_seal_threads(
    image: np.ndarray,
    rng: random.Random,
    num_threads: tuple[int, int] = (1, 3),
    thickness_frac: tuple[float, float] = (0.006, 0.022),
    alpha: tuple[float, float] = (0.45, 0.9),
    max_tilt_deg: float = 55.0,
    twist_prob: float = 0.75,
    highlight_prob: float = 0.5,
    shadow_prob: float = 0.6,
    softness: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    out = image.astype(np.float32)
    h, w = out.shape[:2]

    for _ in range(rng.randint(*num_threads)):
        pts = _thread_polyline(h, w, rng, max_tilt_deg)
        thickness = rng.uniform(*thickness_frac) * h
        color = np.array(rng.choices(THREAD_PALETTE, weights=THREAD_WEIGHTS, k=1)[0], np.float32)
        a_max = rng.uniform(*alpha)
        twist = rng.random() < twist_prob

        if rng.random() < shadow_prob:
            shadow = _stroke_mask(h, w, pts, thickness * 2.1, rng, twist=False)
            shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, thickness * 0.9))
            out *= 1.0 - (shadow * rng.uniform(0.12, 0.3))[..., None]

        core = _stroke_mask(h, w, pts, thickness, rng, twist=twist)

        sigma = rng.uniform(*softness)

        if sigma > 0.05:
            core = cv2.GaussianBlur(core, (0, 0), sigma)

        core_alpha = (core * a_max)[..., None]
        out = out * (1.0 - core_alpha) + color[None, None, :] * core_alpha

        if rng.random() < highlight_prob:
            d = pts[len(pts) // 2 + 1] - pts[len(pts) // 2 - 1]
            n = np.array([-d[1], d[0]], np.float32)
            n /= max(float(np.linalg.norm(n)), 1e-6)

            offset = pts + n[None, :] * (thickness * rng.uniform(0.2, 0.35))
            sheen = _stroke_mask(h, w, offset, max(1.0, thickness * 0.35), rng, twist=twist)
            sheen = cv2.GaussianBlur(sheen, (0, 0), 0.8)

            sheen_color = np.clip(color + rng.uniform(35, 70), 0, 255)  # was (70, 130)
            sheen_alpha = (sheen * a_max * rng.uniform(0.3, 0.6))[..., None]
            out = out * (1.0 - sheen_alpha) + sheen_color[None, None, :] * sheen_alpha

    return np.clip(out, 0, 255).astype(np.uint8)


class SealThreads(A.ImageOnlyTransform):
    def __init__(
        self,
        num_threads: tuple[int, int] = (1, 3),
        thickness_frac: tuple[float, float] = (0.006, 0.022),
        alpha: tuple[float, float] = (0.45, 0.9),
        max_tilt_deg: float = 55.0,
        twist_prob: float = 0.75,
        highlight_prob: float = 0.5,
        shadow_prob: float = 0.6,
        softness: tuple[float, float] = (0.0, 1.0),
        p: float = 0.35,
    ):
        super().__init__(p=p)

        self.num_threads = num_threads
        self.thickness_frac = thickness_frac
        self.alpha = alpha
        self.max_tilt_deg = max_tilt_deg
        self.twist_prob = twist_prob
        self.highlight_prob = highlight_prob
        self.shadow_prob = shadow_prob
        self.softness = softness

    def apply(self, img: np.ndarray, thread_seed: int = 0, **params) -> np.ndarray:
        return draw_seal_threads(
            img,
            rng=random.Random(thread_seed),
            num_threads=self.num_threads,
            thickness_frac=self.thickness_frac,
            alpha=self.alpha,
            max_tilt_deg=self.max_tilt_deg,
            twist_prob=self.twist_prob,
            highlight_prob=self.highlight_prob,
            shadow_prob=self.shadow_prob,
            softness=self.softness,
        )

    def get_params(self) -> dict:
        return {"thread_seed": self.py_random.randrange(2**31)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "num_threads",
            "thickness_frac",
            "alpha",
            "max_tilt_deg",
            "twist_prob",
            "highlight_prob",
            "shadow_prob",
            "softness",
        )


class LCDGlare(A.ImageOnlyTransform):
    def __init__(
        self,
        alpha_range=(0.35, 0.75),
        blur_range=(31, 91),
        p=0.3,
    ):
        super().__init__(p=p)
        self.alpha_range = alpha_range
        self.blur_range = blur_range

    def apply(self, image, **params):
        h, w = image.shape[:2]

        rng = self.random_generator

        overlay = image.astype(np.float32).copy()

        mask = np.zeros((h, w), dtype=np.float32)

        center_x = int(rng.integers(int(0.1 * w), int(0.55 * w)))
        center_y = int(rng.integers(int(0.1 * h), int(0.7 * h)))

        axis_x = int(rng.integers(int(0.25 * w), int(0.65 * w)))
        axis_y = int(rng.integers(int(0.25 * h), int(0.8 * h)))

        angle = int(rng.integers(-35, 35))

        cv2.ellipse(
            mask,
            (center_x, center_y),
            (axis_x, axis_y),
            angle,
            0,
            360,
            1.0,
            -1,
        )

        blur = int(rng.integers(self.blur_range[0], self.blur_range[1]))

        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        mask = mask[..., None]

        alpha = float(rng.uniform(*self.alpha_range))

        glare_color = np.array([255, 255, 255], dtype=np.float32)

        overlay = overlay * (1 - alpha * mask) + glare_color * (alpha * mask)

        return np.clip(overlay, 0, 255).astype(np.uint8)


class DiagonalReflection(A.ImageOnlyTransform):
    def __init__(
        self,
        brightness_range=(50, 120),
        wash_range=(0.15, 0.45),
        blur_range=(21, 61),
        p=0.25,
    ):
        super().__init__(p=p)
        self.brightness_range = brightness_range
        self.wash_range = wash_range
        self.blur_range = blur_range

    def apply(self, image, **params):
        h, w = image.shape[:2]

        rng = self.random_generator

        mask = np.zeros((h, w), dtype=np.float32)

        x1 = int(rng.integers(0, int(0.40 * w)))
        x2 = int(rng.integers(int(0.35 * w), int(0.85 * w)))

        polygon = np.array(
            [
                [0, 0],
                [x1, 0],
                [x2, h],
                [0, h],
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, [polygon], 1.0)

        if rng.random() < 0.5:
            mask = np.fliplr(mask)

        blur = int(rng.integers(self.blur_range[0], self.blur_range[1]))

        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        mask = np.ascontiguousarray(mask)[..., None]

        brightness = float(rng.uniform(*self.brightness_range))
        wash = float(rng.uniform(*self.wash_range))

        result = image.astype(np.float32)

        result = result * (1.0 - wash * mask) + 255.0 * (wash * mask)
        result = result + brightness * mask

        return np.clip(result, 0, 255).astype(np.uint8)


class LowContrastLCD(A.ImageOnlyTransform):
    def __init__(
        self,
        contrast_range=(0.45, 0.8),
        brightness_shift=(-10, 20),
        p=0.3,
    ):
        super().__init__(p=p)
        self.contrast_range = contrast_range
        self.brightness_shift = brightness_shift

    def apply(self, image, **params):
        rng = self.random_generator

        image = image.astype(np.float32)

        contrast = float(rng.uniform(*self.contrast_range))
        brightness = float(rng.uniform(*self.brightness_shift))

        mean = np.mean(image, axis=(0, 1), keepdims=True)
        result = (image - mean) * contrast + mean + brightness

        return np.clip(result, 0, 255).astype(np.uint8)


class EdgeVignette(A.ImageOnlyTransform):
    SIDES = ("left", "right", "top")

    def __init__(
        self,
        width_frac: tuple[float, float] = (0.02, 0.09),
        height_frac: tuple[float, float] = (0.04, 0.14),
        strength: tuple[float, float] = (0.55, 0.85),
        sharpness: tuple[float, float] = (4.0, 12.0),
        tilt_deg: float = 4.0,
        side_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        two_sides_prob: float = 0.5,
        softness: tuple[float, float] = (0.3, 1.5),
        p: float = 0.3,
    ):
        super().__init__(p=p)

        self.width_frac = width_frac
        self.height_frac = height_frac
        self.strength = strength
        self.sharpness = sharpness
        self.tilt_deg = tilt_deg
        self.side_weights = side_weights
        self.two_sides_prob = two_sides_prob
        self.softness = softness

    def _band(self, h: int, w: int, side: str, rng) -> np.ndarray:
        sharp = rng.uniform(*self.sharpness)
        lean = math.tan(math.radians(rng.uniform(-self.tilt_deg, self.tilt_deg)))

        xs = np.arange(w, dtype=np.float32)[None, :]
        ys = np.arange(h, dtype=np.float32)[:, None]

        if side == "top":
            band = max(2.0, rng.uniform(*self.height_frac) * h)
            dist = ys + lean * (xs - w / 2.0)
        else:
            band = max(2.0, rng.uniform(*self.width_frac) * w)
            x = xs + lean * (ys - h / 2.0)
            dist = x if side == "left" else (w - 1 - x)

        mask = np.clip(1.0 - dist / band, 0.0, 1.0) ** (1.0 / sharp)

        sigma = rng.uniform(*self.softness)

        if sigma > 0.05:
            mask = cv2.GaussianBlur(mask, (0, 0), sigma)

        return mask.astype(np.float32)

    def _pick_sides(self, rng) -> list[str]:
        n = 2 if rng.random() < self.two_sides_prob else 1

        pool = list(self.SIDES)
        weights = list(self.side_weights)

        chosen: list[str] = []

        for _ in range(n):
            side = rng.choices(pool, weights=weights, k=1)[0]
            index = pool.index(side)

            chosen.append(side)
            pool.pop(index)
            weights.pop(index)

        return chosen

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        rng = self.py_random

        h, w = image.shape[:2]

        out = image.astype(np.float32)

        for side in self._pick_sides(rng):
            mask = self._band(h, w, side, rng)
            strength = rng.uniform(*self.strength)

            out *= 1.0 - (strength * mask)[..., None]

        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "width_frac",
            "height_frac",
            "strength",
            "sharpness",
            "tilt_deg",
            "side_weights",
            "two_sides_prob",
            "softness",
        )


def build_train_augmentations(seed: int | None = None) -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.88, 1.12),
                translate_percent=(-0.04, 0.04),
                rotate=(-4, 4),
                shear=(-4, 4),
                p=0.45,
            ),
            LowContrastLCD(
                contrast_range=(0.65, 0.9),
                brightness_shift=(0, 20),
                p=0.25,
            ),
            SealThreads(
                num_threads=(1, 3),
                alpha=(0.45, 0.85),
                thickness_frac=(0.010, 0.025),
                softness=(0.0, 1.0),
                p=0.35,
            ),
            A.OneOf(
                [
                    LCDGlare(
                        alpha_range=(0.25, 0.55),
                        blur_range=(31, 91),
                    ),
                    DiagonalReflection(
                        brightness_range=(40, 90),
                        wash_range=(0.15, 0.35),
                        blur_range=(21, 61),
                    ),
                ],
                p=0.35,
            ),
            EdgeVignette(p=0.25),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=4,
                p=0.35,
            ),
            A.OneOf(
                [
                    A.MotionBlur(
                        blur_limit=(9, 19),
                        angle_range=(0, 360),
                    ),
                    A.Defocus(radius=(3, 7), alias_blur=(0.2, 0.5)),
                    A.ZoomBlur(max_factor=(1.03, 1.10), step_factor=(0.01, 0.02)),
                    A.Downscale(
                        scale_range=(0.35, 0.55),
                        interpolation_pair={
                            "downscale": cv2.INTER_NEAREST,
                            "upscale": cv2.INTER_NEAREST,
                        },
                    ),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.12, 0.18),
                        contrast_limit=(-0.10, 0.20),
                    ),
                    A.RandomGamma(
                        gamma_limit=(75, 105),
                    ),
                ],
                p=0.5,
            ),
        ],
        seed=seed,
    )
