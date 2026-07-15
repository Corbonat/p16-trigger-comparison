# Проверенное окружение

Финальный smoke-аудит перед обучением выполнен в следующем CPU-окружении:

| Компонент | Версия |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| NumPy | 2.4.4 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| scikit-image | 0.26.0 |
| Matplotlib | 3.11.0 |

Для разового ускоренного чтения Parquet-зеркала при локальном smoke-аудите
дополнительно использовался PyArrow 25.0.0. Он не нужен обычному pipeline через
`torchvision.CIFAR10` и поэтому не включён в зависимости проекта.

Это не требование использовать именно CPU. Скрипт `scripts/train.py` с
`--device auto` автоматически выберет CUDA, MPS или CPU. Таблица нужна, чтобы
зафиксировать окружение, в котором были выполнены тесты готовности.

Для явной установки CPU-сборки PyTorch перед установкой проекта:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[train,dev]"
```
