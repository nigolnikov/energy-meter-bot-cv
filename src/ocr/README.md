# src/ocr

Распознавание показаний счётчиков на базе TrOCR. В этом каталоге живут две
независимые модели: **digital** (цифровой ЖК/LED-счётчик) и **analog**
(циферблатный счётчик). Обе — файнтюн одной и той же архитектуры TrOCR,
обученной на посимвольных лейблах.

## Основные модули

| Файл | Назначение |
|---|---|
| `text.py` | Формат лейбла, используемый везде: `encode_label`/`decode_label` конвертируют между сырой строкой показания и посимвольным форматом для токенизатора, `is_valid_reading` проверяет, что декодированное предсказание — это цифры и не более одной точки. Без внешних зависимостей — можно импортировать откуда угодно. |
| `augmentations_digital.py` | Пайплайн аугментаций для digital-модели, плюс кастомные трансформы, из которых он собран (`SealThreads`, `LCDGlare`, `DiagonalReflection`, `LowContrastLCD`, `EdgeVignette`) — имитируют пломбировочную проволоку, блики экрана и выцветший контраст ЖК-дисплея. |
| `augmentations_analog.py` | Пайплайн аугментаций для analog-модели. Специально собран только из встроенных трансформов Albumentations (лёгкие affine/blur/downscale/brightness) — у analog-кропов нет специфичных для digital проблем выше. |
| `trocr_training.py` | Общая инфраструктура обучения для обоих `train_*.py`: `MeterOCRDataset`, настройка модели/процессора, подсчёт метрик, работа с mlflow, сохранение графиков и отчётов.  |
| `reporting.py` | Отчёт по итогам обучения: прогоняет модель по датасету и пишет CSV/PNG (разбивка по типам ошибок, accuracy vs. coverage, confusion matrix символов, худшие предсказания) и `summary.json`. Вызывается обоими `train_*.py` в конце обучения. |

## Точки входа

| Файл | Запуск | Назначение |
|---|---|---|
| `train_trocr.py` | `python -m src.ocr.train_trocr --config configs/trocr_config.yaml` | Файнтюн digital-модели. |
| `train_analog_trocr.py` | `python -m src.ocr.train_analog_trocr --config configs/trocr_analog_config.yaml` | Файнтюн analog-модели. Есть `--no-augment` для отключения аугментаций при отладке. |
| `infer_trocr.py` | импортируется, не запускается напрямую | Собственно код инференса. `infer()` прогоняет одно изображение через переданные модель/процессор/device. `ocr_infer(image)` — продакшн-точка входа, которую вызывает `pipeline.py`: лениво загружает и кэширует модель из `model_best/trocr-meter-finetuned` при первом вызове. |
| `eval_trocr.py` | `python -m src.ocr.eval_trocr --images <dir> --labels <csv> [--model <path>] [--output-prefix <name>]` | Оценка обученной модели на размеченном наборе изображений. Пишет `{prefix}_predictions.csv` и `{prefix}_summary.csv` (exact-match, CER, digit accuracy, разбивка ошибок по точке/цифрам, numeric error). `--model` по умолчанию `models/trocr-meter-finetuned`, но подходит любой чекпоинт TrOCR (например `models_analog/trocr-meter-analog`) — этот скрипт покрывает обе модели. |

Оба `train_*.py` используют один и тот же набор CLI-флагов через
`add_common_train_args` из `trocr_training.py`: `--config` (обязателен),
`--epochs`, `--batch-size`, `--learning-rate`, `--model-name`, `--run-name`,
`--max-target-length`, `--length-penalty`, `--num-beams`, `--seed`,
`--no-mlflow`. Конфиги лежат в `configs/` (`trocr_config.yaml`,
`trocr_analog_config.yaml`) с секциями `experiment` / `mlflow` / `model` /
`data` / `train`; флаги CLI переопределяют значения из конфига.

## Метрики сравнения текста

Общие метрики сравнения строк (`exact_match`, `character_error_rate`,
`word_error_rate`, `digit_accuracy`, `numeric_error`, `classify_error`) живут
в `src/utils/metrics.py` — они применимы к сравнению любых двух
строк (предсказание/эталон) и используются `eval_trocr.py`,
`trocr_training.py` и скриптами из `tools/`.

## tools/

Отдельные dev-скрипты (бенчмарк, robustness-тесты, превью аугментаций),
которые не импортируются ни продакшн-кодом, ни обучением. См.
[`tools/README.md`](tools/README.md).
