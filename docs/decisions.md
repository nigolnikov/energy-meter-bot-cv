## Experiment Tracking Choice

Decision: MLflow

Reason:
- open-source
- works locally (no cloud required)
- suitable for YOLO + OCR experiments
- easy integration with Python training scripts

## Changes
У YOLO #2 было два рассогласования кода с реальной моделью: код читал obb-формат, а модель — detect; и код искал класс reading, а модель выдаёт reading_area. Проверил через `m.task` и `m.names`, привёл код в соответствие с моделью.
