# **ENERGY-METER-BOT-CV**

## **Описаание проекта**
Energy Meter Bot CV это система распознавания показаний электросчётчика по фото. На вход — фотография, на выход — числовое показание. Пайплайн: детекция (YOLO) → распознавание (TrOCR).
1.  **Детекция (Stage 1: YOLOv8):**
    - На входном изображении выделяется область интереса (ROI) — непосредственно табло с цифрами счётчика.
    - Модель отсекает лишний фон, блики и искажения, фокусируясь исключительно на цифровом поле.

2.  **Распознавание (Stage 2: TrOCR):**
    - Вырезанный фрагмент табло подаётся в нейросетевой энкодер-декодер (Transformer-based OCR).
    - TrOCR преобразует визуальные паттерны цифр в последовательный текст, выдавая итоговое числовое значение показания.

## **1. Архитектура пайплайна**
```
Фото → YOLO#1 (meter/screen) → crop → enhancement (CLAHE)
 → YOLO#2 (reading area) → crop → TrOCR → post-processing → JSON
```

## **2. Требования**
Python 3.11 (обязательно — с 3.13 несовместимы torch/sentencepiece). Если Python 3.11 не установлен, поставьте его перед созданием venv.
Проверьте версию:
```
python --version
```

## **3. Клонируйте репозиторий и перейдите в папку:**
```
git clone <url-репозитория>
cd electric-energy-bot-ml
```

## **4. Создайте и активируйте виртуальное окружение:**
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

## **5. Установите все зависимости:**
```
pip install -r requirements.txt
```

## **6. Установите проект с dev-зависимостями:**
```
pip install -e ".[dev]"
```
Эта команда установит сам проект, а так же `ruff` и `pre-commit`.

## **7. Активируйте pre-commit хуки (делается ОДИН раз):**
```
pre-commit install
```
Проверьте все работает:
```
pre-commit run --all-files
```

## **8. Запуск пайплайна на одном фото**
Перед запуском (модели грузятся локально, без обращения к HuggingFace):
```
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

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

## **9. Batch-оценка на тест-наборе**
```
python -m src.utils.batch_eval --test-csv datasets/labels.csv
```
Прогоняет весь тест-набор, выводит E2E-метрики: exact_matches, e2e_exact_match,
mean_digit_accuracy, mean_cer и разбивку по статусам.

Вы должны создать в корне папку `data`, внутри него еще папку `e2e`, затем `images` и в конце туда положить фотографии. И `labels.csv` файл с правильными значениями нужно положить в `datsets`.
```
ENERGY-METER-BOT-CV
|--datasets/
    |--labels.csv
|--data/e2e/images
    |--фотографии
```

Флаг `--enhancement` принимает `none | clahe | zerodce` — используется для
A/B-сравнения методов улучшения изображения.

Флаг `--crop-mode` принимает `axis | warp` — используется для
A/B-сравнения методов нарезания фотографий.

Формат `--test-csv`: колонки `image_path,true_value`.

Пример:
```
python -m src.utils.batch_eval --test-csv datasets/labels.csv --enhancement none --crop-mode warp
```

## **10. Запуск: веб-сервис и UI**
```
uvicorn api:app --reload
```
Открыть в браузере по ссылкам:
Веб-интерфейс: ```http://127.0.0.1:8000/```
Интерактивная API-документация: ```http://127.0.0.1:8000/docs```

## **11. API-эндпойнты**
    - GET / — веб-страница
    - GET /health — проверка живости, {"status":"ok"}
    - POST /predict — принимает фото, возвращает JSON с показанием
    - GET /output — последняя картинка с разметкой

Пример ответа(JSON) `/predict`:
```
{
  "meter_bbox": [...],
  "screen_bbox": [...],
  "reading_bbox": [...],
  "raw_text": "1 2 3 4 5 6 . 7 8",
  "value": "123456.78",
  "confidence": 0.0,
  "status": "ok"
}
```

## **12. Docker**
Собрать Docker image со всеми зависимостями, моделями и кодом:
```
docker build -t energy-meter-bot .
```
Поднять Docker container:
```
docker run -p 8001:8000 energy-meter-bot
```
сервис сам определяет CPU/GPU.

Открыть в браузере: ```http://0.0.0.0:8001/```

## **13. Тесты**
Smoke-тесты проверяют /health, /, /predict и обработку ошибок:
```
pytest test_api.py -v
```
