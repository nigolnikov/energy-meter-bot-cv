# **Pipeline contracts - Electric Enrgy Bot**
## **1. System Overview**

Pipeline consists of 3 stages:

Input Image
- YOLO #1 (meter + screen detection)
- crop screen
- Enhancement (CLAHE / Zero-DCE)
- YOLO #2 (reading detection)
- crop reading
- TrOCR (text recognition)
- Post-processing
- Final JSON output

## **2. YOLO #1 — Meter & Screen Detector**
Input
image: np.ndarray (H, W, 3)

Output
```
List[Detection]
```

Detection schema
```
{
    "bbox": [x1, y1, x2, y2, x3, y3, x4, y4],
    "class": "meter | digital_display | analog_register",
    "confidence": float
}
```
- Coordinates are in original image pixel space
- Multiple detections allowed
- Primary outputs: meter + screen

## **3. YOLO #2 — Reading Detection (Role A)**

Input
- cropped screen image: np.ndarray (H, W, 3)

Output
```
List[Detection]
```

Detection schema
```
{
  "bbox": [x1, y1, x2, y2],
  "class": "reading",
  "confidence": float
}
```
- Detects only electricity reading digits
- Must ignore noise(date, temperature, icons)

## **4. OCR — TrOCR (Role B)**

Input
cropped reading image: np.ndarray (H, W, 3)

Output
```
{
  "text": str,
  "confidence": float
}
```
- Returns raw recognized text
- Post-preprocessing handled outside OCR

## **5. Final Output Contract (System API)**
```
{
  "meter_bbox": [x1, y1, x2, y2],
  "screen_bbox": [x1, y1, x2, y2],
  "reading_bbox": [x1, y1, x2, y2],
  "raw_text": "0012 3",
  "value": "00123",
  "confidence": 0.91,
  "status": "ok | no_meter | no_reading | low_confidence"
}
```

6. Status meanings
- `ok` → everything detected successfully
- `no_meter` → YOLO #1 failed
- `no_reading` → YOLO #2 failed
- `low_confidence` → result is unreliable
