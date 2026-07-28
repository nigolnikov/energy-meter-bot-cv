# src/ocr/tools

Автономные dev-скрипты для бенчмарка, стабильности и визуальной диагностики OCR-пайплайна. Ни один из них не импортируется продакшн-кодом (`pipeline.py`, `api.py`) или обучением (`train_trocr.py`, `train_analog_trocr.py`) — все запускаются вручную.

## Скрипты

| Файл | Запуск | Что делает |
|---|---|---|
| `benchmark_inference.py` | `python -m src.ocr.tools.benchmark_inference --model <dir> --images <dir> [--device cpu\|cuda\|both] [--num-beams 4 ...] [--runs 30] [--warmup 5]` | Замеряет латентность инференса (preprocess/generate/total, p90, throughput) на CPU и/или GPU, для одного или нескольких значений `num_beams`. |
| `eval_short_vs_long.py` | `python -m src.ocr.tools.eval_short_vs_long --short-csv <csv> --long-csv <csv> [--out-dir eval_out]` | Базовый замер: один и тот же чекпоинт на батче коротких показаний и на батче длинных (8-значных). Гоняет `ocr_infer` напрямую на уже вырезанных кропах, без detection-пайплайна. Нужен, чтобы зафиксировать точность по длине ДО добавления синтетических коротких примеров в обучение и потом сравнить дельту. |
| `robustness_eval.py` | `python -m src.ocr.tools.robustness_eval --images <dir> --labels <csv> [--image-column filename] [--text-column text] [--output-dir reports/robustness_trocr]` | Прогоняет модель через набор фиксированных "стресс"-трансформов (перспектива, блур, downscale, тень, блик, низкий контраст, пломбировочные нити — mild/medium/strong) и сравнивает точность с чистым baseline. Каждому тесту присваивается статус `OK`/`WARN`/`FAIL` по величине просадки. |
| `preview_augmentations.py` | `python -m src.ocr.tools.preview_augmentations --mode stages\|grid --image <path> [--samples N] [--seed N]` | Визуальная проверка пайплайна аугментаций digital-модели (`augmentations_digital.py`). Два режима: `stages` — каждая стадия реального пайплайна отдельно (при p=1.0) плюс "stress"-уровни для robustness-теста, по одному PNG на сэмпл в `--output-dir`; `grid` — N реальных прогонов всего пайплайна собранные в один пронумерованный contact sheet (`--output`), чтобы на глаз оценить долю "убитых" аугментацией примеров. |
