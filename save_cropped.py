from pathlib import Path

import cv2

from obb_crop_robust import crop_and_warp_obb

# PATHS
dir_img = Path("datasets/yolo_meter_screen/images")
dir_lab = Path("datasets/yolo_meter_screen/labels")

dir_res = Path("datasets/yolo_res1_dig/images")
dir_res.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    1: "digital_display",
    2: "analog_register",
}

for split in ["train", "val", "test"]:
    img_dir = dir_img / split
    lab_dir = dir_lab / split

    outp_split = dir_res / split
    outp_split.mkdir(parents=True, exist_ok=True)

    for image_path in img_dir.glob("*.jpg"):
        label_path = lab_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        h, w = image.shape[:2]

        with open(label_path) as f:
            lines = f.readlines()

        crop_id = 0

        for line in lines:
            values = line.strip().split()

            if len(values) != 9:
                continue

            cls = int(values[0])

            # meter ignored and analog register not considered here, only digital displays

            if cls == 0:
                continue

            coords = list(map(float, values[1:]))

            pixel_points = []

            for i in range(0, 8, 2):
                x = coords[i] * w
                y = coords[i + 1] * h
                pixel_points.extend([x, y])

            crop, ambiguous = crop_and_warp_obb(image, pixel_points)

            if crop is None:
                continue

            out_name = f"{image_path.stem}_{CLASS_NAMES[cls]}_{crop_id}.jpg"

            cv2.imwrite(str(outp_split / out_name), crop)

            crop_id += 1

print("all ready")
