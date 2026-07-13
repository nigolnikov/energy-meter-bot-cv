# **energy-meter-bot-cv**

## **0. Требования**
Python 3.11 (обязательно — с 3.13 несовместимы torch/sentencepiece). Если Python 3.11 не установлен, поставьте его перед созданием venv.
Проверьте версию:
```
python --version
```

## **1. Клонируйте репозиторий и перейдите в папку:**
```
git clone <url-репозитория>
cd electric-energy-bot-ml
```

## **2. Создайте и активируйте виртуальное окружение:**
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

## **3. Установите все зависимости:**
```
pip install -r requirements.txt
```

## **4. Установите проект с dev-зависимостями:**
```
pip install -e ".[dev]"
```
Эта команда установит сам проект, а так же `ruff` и `pre-commit`.

## **4. Активируйте pre-commit хуки (делается ОДИН раз):**
```
pre-commit install
```
Проверьте все работает:
```
pre-commit run --all-files
```

## **6. Запуск пайплайна на одном фото**
```
python -m pipeline /<path>/<image-name>
```
Example:
```
python -m pipeline datasets/test/photo_screen_2270.jpg
```
Прогоняет фото через весь пайплайн (YOLO #1 → enhancement → YOLO #2 → TrOCR)
и печатает JSON-результат: bbox счётчика/экрана/показания, распознанное значение,
confidence, статус. Визуализация сохраняется в `output.jpg`.

## **7. Batch-оценка на тест-наборе**
```
python -m src.utils.batch_eval --test-csv datasets/test_set.csv --enhancement clahe
```
Прогоняет весь тест-набор, выводит E2E-метрики: exact_matches, e2e_exact_match,
mean_digit_accuracy и разбивку по статусам.

Вы должны создать в корне папку `datasets`, внутри него еще папку `test`, потом туда положить фотографии. И `test_set.csv` файл с правильными значениями нужно положить в `datsets`.

Флаг `--enhancement` принимает `none | clahe | zerodce` — используется для
A/B-сравнения методов улучшения изображения.

Формат `--test-csv`: колонки `image_path,true_value`.
