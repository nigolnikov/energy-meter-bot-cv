import cv2


def draw_bbox(image, bbox, label, confidence=None):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = label
    if confidence is not None:
        text += f" {confidence:.2f}"

    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image


def draw_detections(image, detections):
    for det in detections:
        bbox = det["bbox"]
        label = det["label"]
        confidence = det.get("confidence")

        image = draw_bbox(image, bbox, label, confidence)

    return image


def save_image(image, path):
    cv2.imwrite(path, image)
