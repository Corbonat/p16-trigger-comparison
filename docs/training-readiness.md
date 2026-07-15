# Граница готовности к обучению

Перед длительным запуском должны выполняться все пункты:

- [x] зафиксированы датасет, целевой класс и три seed;
- [x] определены пять триггеров и их параметры;
- [x] яркость зафиксирована как `delta = 0.15`;
- [x] train/validation split детерминирован и стратифицирован;
- [x] poisoned indices выбираются только среди нецелевых классов;
- [x] тест с триггером имеет те же индексы и порядок, что чистый тест;
- [x] CNN возвращает десять логитов;
- [x] реализованы обычная точность, ASR, conditional ASR, `S(c)` и запас логитов;
- [x] результаты и checkpoints имеют уникальные имена;
- [x] команды полной матрицы генерируются без запуска по умолчанию;
- [x] notebook отделяет подготовку от ячеек обучения;
- [x] локальный smoke test выполнен на реальном CIFAR-10: clean `0/512` и poisoned `15/512`, формы батча и логитов корректны;
- [x] визуальная сетка реальных CIFAR-10 сохранена и проверена.

Последние два пункта проверяются командами:

```powershell
python scripts/prepare_data.py
python scripts/audit_triggers.py
python scripts/train.py --mode clean --trigger patch --seed 17 --smoke-test --max-train-samples 512 --max-eval-samples 256
python scripts/train.py --mode poisoned --trigger patch --poison-fraction 0.03 --seed 17 --smoke-test --max-train-samples 512 --max-eval-samples 256
```

После их успешного завершения следующая команда уже начинает обучение:

```powershell
python scripts/train.py --mode clean --trigger patch --seed 17
```

Последняя проверка готовности выполнена 15 июля 2026 года. Для ускоренного
локального аудита использован Parquet-снимок `uoft-cs/cifar10`; основной pipeline
проекта по-прежнему загружает CIFAR-10 стандартным классом `torchvision.CIFAR10`.
