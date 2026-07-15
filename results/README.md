# Формат результатов

Каждый запуск `scripts/train.py` сохраняет один JSON в `results/runs/`.

Ключевые поля:

- `run`, `mode`, `trigger`, `poison_fraction`, `seed`;
- `poisoned_training_examples`;
- `dataset_sizes`: фактические размеры train/validation/test;
- `history`: loss и accuracy по эпохам;
- `metrics.clean_accuracy`;
- `metrics.triggered_accuracy`;
- `metrics.asr`;
- `metrics.conditional_asr`;
- `metrics.per_source_class_asr`;
- `metrics.target_margin_mean` и `target_margin_median`;
- `checkpoint`.

Идентификатор имеет вид `poisoned-patch-rho-0p030-seed-17`. Повторный запуск
того же сочетания mode/trigger/rho/seed с другим числом эпох или размером
подвыборки перезапишет рабочий JSON и checkpoint. Завершённый срез нужно
копировать в отдельную именованную папку.

Первый сохранённый срез: [pilot-10k-seed17/README.md](pilot-10k-seed17/README.md).
