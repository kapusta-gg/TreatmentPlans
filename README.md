# TreatmentPlans

Исследовательский проект по подготовке данных для автоматизированного планирования дистанционной лучевой терапии рака предстательной железы. Текущая версия читает DICOM-файлы пациента и извлекает КТ, контуры структур, рассчитанную дозу и параметры плана. Обучения модели и генерации нового плана в репозитории пока нет.

## Что реализовано

- **КТ (`CT`)**: чтение срезов, удаление дубликатов по `SOPInstanceUID`, сортировка по координате среза и перевод значений пикселей в HU через `RescaleSlope` и `RescaleIntercept`.
- **Структуры (`RTSTRUCT`)**: сопоставление контуров со срезами КТ по `SOPInstanceUID`, чтение названий и номеров ROI. Для построения двумерных масок доступны `cv2.fillPoly` и `skimage.draw.polygon`; по умолчанию в `collect_masks` используется OpenCV. Контуры с менее чем тремя точками не заполняются.
- **Доза (`RTDOSE`)**: чтение трёхмерной карты с применением `DoseGridScaling`, геометрических метаданных и записей DVH. Доза переносится на сетку КТ с линейной интерполяцией в плоскости каждого среза.
- **План (`RTPLAN`)**: чтение дозовых ориентиров, групп фракций, пучков и их контрольных точек.
- **Проверка связей**: RTSTRUCT сверяется с серией КТ; при общем вызове проверяется единый `FrameOfReferenceUID` всех КТ-срезов и его совпадение с UID RTDOSE и RTPLAN.
- **Общая подготовка**: `prepare_all_data()` собирает КТ, маски, дозу на сетке КТ и метаданные пациента.
- **Сохранение**: отдельные функции для записи КТ и масок в `.npz`, а сведений о срезах и ROI — в `.json`. Общий результат пока автоматически не сохраняется.

## Структура проекта

| Путь | Назначение |
| --- | --- |
| `main.py` | Пример вызова общего обработчика для одного пациента. |
| `parser/scan.py` | КТ и метаданные срезов. |
| `parser/counter.py` | Контуры RTSTRUCT и построение масок. |
| `parser/dose.py` | Карта дозы и DVH из RTDOSE. |
| `parser/plan.py` | Данные RTPLAN. |
| `parser/parser.py` | Общая подготовка данных пациента и функции сохранения КТ с масками. |
| `checkers/compare/mask.py` | Визуальное сравнение масок OpenCV и scikit-image. |
| `checkers/check/filepath.py` | Проверка наличия входных путей и расширения файлов DICOM. |
| `errors.py` | Исключения для проверок UID. |

## Установка

Для Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Данные DICOM не входят в репозиторий. В `main.py` указаны примерные пути следующего вида:

```text
datasets/
└── Center2/
    ├── CT images/
    │   └── Patient01/
    │       └── ... файлы срезов КТ ...
    └── Treatment plans/
        └── Patient01/
            ├── RS.dcm   # RTSTRUCT
            ├── RD.dcm   # RTDOSE
            └── RP.dcm   # RTPLAN
```

Если данные лежат в другом месте, измените пути в `main.py`. Запуск:

```powershell
python main.py
```

### Что возвращает `prepare_all_data()`

Функция принимает путь к каталогу КТ и пути к файлам RTSTRUCT, RTDOSE и RTPLAN. Пример использования:

```python
from parser.parser import prepare_all_data

cts, masks, dose_on_ct, info = prepare_all_data(
    ct_dir="datasets/Center2/CT images/Patient01",
    counters_filename="datasets/Center2/Treatment plans/Patient01/RS.dcm",
    dose_filename="datasets/Center2/Treatment plans/Patient01/RD.dcm",
    plan_filename="datasets/Center2/Treatment plans/Patient01/RP.dcm",
)
```

| Значение | Содержимое | Форма |
| --- | --- | --- |
| `cts` | КТ в единицах HU; дубликаты срезов по `SOPInstanceUID` удалены. | `(Z, H, W)` |
| `masks` | Булевы маски ROI на сетке КТ. | `(Z, R, H, W)` |
| `dose_on_ct` | Доза на сетке КТ, в единицах RTDOSE (`GY` для приведённого набора данных). | `(Z, H, W)` |
| `info` | Словарь метаданных КТ, ROI, дозы и плана. | — |

Здесь `Z` — число срезов, `R` — число каналов ROI, `H` и `W` — размеры среза. Для проверенного пациента массивы имеют формы `(236, 512, 512)`, `(236, 31, 512, 512)` и `(236, 512, 512)` соответственно. Состав ROI и число срезов у других пациентов могут отличаться.

В `info["ct"]` находятся UID серии и системы координат, а также позиции срезов. `info["roi_channels"]` связывает индекс канала маски с номером и названием ROI: например, `info["roi_channels"]["14"]` описывает `masks[:, 14]`. В `info["dose"]` находятся геометрия исходной дозовой сетки и DVH, в `info["plan"]` — разобранные параметры RTPLAN. Функция возвращает данные в памяти и сама не записывает файлы.

При переносе дозы области КТ вне исходной дозовой сетки заполняются нулями. `dose.resize()` вычисляет также `valid_mask` для различения этих областей, но `prepare_all_data()` сейчас не возвращает эту маску.

## Сохранение КТ и масок отдельно

Функции сохранения можно вызвать вручную:

```python
from parser import scan, counter
from parser.parser import save_ct_and_masks, save_info_json

ct_dir = "datasets/Center2/CT images/Patient01"
rs_path = "datasets/Center2/Treatment plans/Patient01/RS.dcm"

cts, ct_info = scan.prepeare_CT_images(ct_dir)
counter.open_conture_dicom(rs_path, ct_info)
masks = counter.collect_masks(cts, ct_info)

save_ct_and_masks(cts, masks, "output/Patient01_ct_masks.npz")
save_info_json(ct_info, "output/Patient01_info.json")
```

В архиве находятся массивы `ct` формы `(срезы, высота, ширина)` и `masks` формы `(срезы, ROI, высота, ширина)`. JSON хранит соответствие каналов масок номерам и названиям ROI, а также координаты и UID срезов. Номера ROI в текущем построителе масок предполагаются последовательными, начиная с 1.


