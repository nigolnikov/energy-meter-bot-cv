# Yolo outputs
class Detection:
    def __init__(self, bbox, cls, confidence):
        self.bbox = bbox
        self.cls = cls
        self.confidence = confidence

    def __repr__(self):
        return f"Detection(cls='{self.cls}', bbox={self.bbox}, confidence={self.confidence:.3f})"


# TrOCR outputs
class OCRResult:
    def __init__(self, text, confidence):
        self.text = text
        self.confidence = confidence

    def __repr__(self):
        return f"OCR result(text='{self.text}', confidence={self.confidence:.3f})"


# Final Result
class PipelineResult:
    def __init__(self, meter_box, screen_box, reading_box, raw_text, value, confidence, status):
        self.meter_box = meter_box
        self.screen_box = screen_box
        self.reading_box = reading_box
        self.raw_text = raw_text
        self.value = value
        self.confidence = confidence
        self.status = status

    def __repr__(self):
        return f"Pipeline result: value='{self.value}', status='{self.status}'"

    def to_dict(self):
        return {
            "meter_bbox": self.meter_bbox,
            "screen_bbox": self.screen_bbox,
            "reading_bbox": self.reading_bbox,
            "raw_text": self.raw_text,
            "value": self.value,
            "confidence": self.confidence,
            "status": self.status,
        }
