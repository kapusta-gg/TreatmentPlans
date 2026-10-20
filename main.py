import json
import os
from time import perf_counter
from pathlib import Path

import pydicom
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
import cv2


def prepeare_CT_images(patient_dir: str, center: str) -> tuple[np.array, dict]:
    # готовим путь и читаем
    dir = "datasets"
    dcm_files = "CT images"
    path = os.path.join(*(dir, center, dcm_files, patient_dir))
    info = [pydicom.dcmread(os.path.join(*(path, dcm)))
            for dcm in os.listdir(path)]

    info.sort(key=lambda x: x.ImagePositionPatient[-1])

    # Собираем снимки
    cts = np.stack([dcm.pixel_array * dcm.RescaleSlope +
                    dcm.RescaleIntercept for dcm in info], axis=0)
    # Собираем информацию по снимку
    plus_info = [{
        "slice_thickness": float(dcm.SliceThickness),
        "pixel_spacing": np.asarray(dcm.PixelSpacing, dtype=float),
        "positions": np.asarray(
            [dcm.ImagePositionPatient for dcm in info], dtype=float
        ),
        "orientation": np.asarray(dcm.ImageOrientationPatient, dtype=float),
        "series_uid": dcm.SeriesInstanceUID,
        "frame_uid": dcm.FrameOfReferenceUID,
        "sop_uid": dcm.SOPInstanceUID,
        "countors": [],
        "patient_pos": dcm.ImagePositionPatient
    } for dcm in info]
    return cts, plus_info


def open_conture_dicom(filename: str, ct_info: dict[str: any]) -> None:
    # Готовим путь и  читаем
    dir = "datasets"
    centre = "Center2"
    dcm_files = "Treatment plans"
    patient = "Patient01"
    full_path = "/".join([".", dir, centre, dcm_files, patient, filename])
    data = pydicom.dcmread(full_path, force=True)
    series_uid = data.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
    series_uid = series_uid.RTReferencedSeriesSequence[0].SeriesInstanceUID
    if ct_info[0]["series_uid"] != str(series_uid):
        print("Wrong UID")
        return

    # Подготовка словаря: Номер ROI -> Информация о ROI
    roi_struct = {int(el.ROINumber): {"name": str(el.ROIName)}
                  for el in data.StructureSetROISequence}
    roi = data.ROIContourSequence
    for info in ct_info:
        sop_uid = info["sop_uid"]
        for el in roi:
            roi_struct[int(el.ReferencedROINumber)]["color"] = list(
                el.ROIDisplayColor)
            info["countors"].append(
                {int(el.ReferencedROINumber): roi_struct[int(el.ReferencedROINumber)].copy()})
            info["countors"][-1][int(el.ReferencedROINumber)]["arrays"] = []
            info["countors"][-1][int(el.ReferencedROINumber)]["types"] = []
            for seq in el.ContourSequence:
                if str(seq.ContourImageSequence[0].ReferencedSOPInstanceUID) == sop_uid:
                    info["countors"][-1][int(el.ReferencedROINumber)
                                         ]["arrays"].append(np.array(seq.ContourData))
                    info["countors"][-1][int(el.ReferencedROINumber)
                                         ]["types"].append(str(seq.ContourGeometricType))


def prepare_mask_skimage(shape, countor, slice_info, roi_number):
    mask = np.zeros(shape, dtype=bool)
    roi = countor[roi_number]

    position = np.asarray(slice_info["patient_pos"], dtype=float)
    orientation = np.asarray(slice_info["orientation"], dtype=float)
    spacing = np.asarray(slice_info["pixel_spacing"], dtype=float)

    for data in roi["arrays"]:
        points = np.asarray(data, dtype=float).reshape(-1, 3)
        if len(points) < 3:  # например, ISOCENTER имеет тип POINT
            continue

        delta = points - position
        cols = (delta @ orientation[:3]) / spacing[1]
        rows = (delta @ orientation[3:]) / spacing[0]

        rr, cc = polygon(rows, cols, shape=shape)
        mask[rr, cc] = True

    return mask


def prepare_mask_cv(shape, countor, slice_info, roi_number):
    mask = np.zeros(shape, dtype=np.uint8)
    roi = countor[roi_number]

    position = np.asarray(slice_info["patient_pos"], dtype=float)
    orientation = np.asarray(slice_info["orientation"], dtype=float)
    spacing = np.asarray(slice_info["pixel_spacing"], dtype=float)

    for data in roi["arrays"]:
        points = np.asarray(data, dtype=float).reshape(-1, 3)
        if len(points) < 3:  # например, ISOCENTER имеет тип POINT
            continue

        delta = points - position
        cols = (delta @ orientation[:3]) / spacing[1]
        rows = (delta @ orientation[3:]) / spacing[0]

        vertices = np.rint(
            np.column_stack((cols, rows)) * 16
        ).astype(np.int32)

        cv2.fillPoly(mask, [vertices], color=1, shift=4)

    return mask.astype(bool)


def collect_masks(cts, info):
    n_roi = len(info[0]["countors"])
    masks = np.zeros((len(cts), n_roi, *cts.shape[1:]), dtype=bool)

    for j in range(len(cts)):
        for roi_number, countor in enumerate(info[j]["countors"], start=1):
            masks[j, roi_number - 1] = prepare_mask_cv(
                cts[j].shape, countor, info[j], roi_number
            )
        info[j]["masks"] = masks[j]

    return masks


def save_ct_and_masks(cts: np.ndarray, masks: np.ndarray, filename: str) -> None:
    if cts.ndim != 3:
        raise ValueError(
            f"Ожидалась форма КТ (срезы, H, W), получено {cts.shape}")

    if masks.ndim != 4:
        raise ValueError(
            f"Ожидалась форма масок (срезы, ROI, H, W), получено {masks.shape}"
        )

    if masks.shape[0] != cts.shape[0] or masks.shape[2:] != cts.shape[1:]:
        raise ValueError(
            "Число срезов или размер изображения у КТ и масок не совпадают")

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ct=cts, masks=masks)


def save_info_json(info: list, filename: str) -> None:
    rois = []
    for channel, item in enumerate(info[0]["countors"]):
        roi_number, roi_data = next(iter(item.items()))
        rois.append({
            "channel": channel,  # индекс в masks[:, channel]
            "roi_number": int(roi_number),
            "name": roi_data["name"],
        })

    metadata = {
        "rois": rois,
        "slices": [
            {
                "position": list(slice_info["patient_pos"]),
                "pixel_spacing": list(slice_info["pixel_spacing"]),
                "orientation": list(slice_info["orientation"]),
                "sop_uid": str(slice_info["sop_uid"]),
                "series_uid": str(slice_info["series_uid"]),
                "frame_uid": str(slice_info["frame_uid"]),
            }
            for slice_info in info
        ],
    }

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def compare_mask_methods(cts, info, roi_number, slice_idx=None):
    # Если срез не задан, берём средний срез, где есть контур этого ROI
    if slice_idx is None:
        suitable = [
            j for j, slice_info in enumerate(info)
            if any(
                roi_number in item and item[roi_number]["arrays"]
                for item in slice_info["countors"]
            )
        ]
        if not suitable:
            raise ValueError(f"Контуры ROI {roi_number} не найдены")
        slice_idx = suitable[len(suitable) // 2]

    slice_info = info[slice_idx]
    countor = next(
        item for item in slice_info["countors"] if roi_number in item
    )
    roi = countor[roi_number]

    shape = cts[slice_idx].shape
    old = prepare_mask_skimage(
        shape, countor, slice_info, roi_number).astype(bool)
    new = prepare_mask_cv(shape, countor, slice_info, roi_number).astype(bool)

    added = new & ~old    # есть только в OpenCV
    removed = old & ~new  # есть только в skimage

    def add_overlay(ax, mask, color, alpha=0.55):
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[..., :3] = color
        overlay[..., 3] = mask * alpha
        ax.imshow(overlay, interpolation="nearest")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax in axes:
        ax.imshow(cts[slice_idx], cmap="gray", vmin=-200, vmax=300,
                  interpolation="nearest")
        ax.axis("off")

    add_overlay(axes[0], old, (0, 1, 0))  # зелёный
    axes[0].set_title("skimage")

    add_overlay(axes[1], new, (1, 0, 0))  # красный
    axes[1].set_title("OpenCV")

    add_overlay(axes[2], added, (1, 0, 0), alpha=0.85)
    add_overlay(axes[2], removed, (0, 0.5, 1), alpha=0.85)
    axes[2].set_title("Разница: красный — добавлено, синий — убрано")

    # Жёлтая линия — исходный контур из RTSTRUCT
    position = np.asarray(slice_info["patient_pos"], dtype=float)
    orientation = np.asarray(slice_info["orientation"], dtype=float)
    spacing = np.asarray(slice_info["pixel_spacing"], dtype=float)

    for data in roi["arrays"]:
        points = np.asarray(data, dtype=float).reshape(-1, 3)
        if len(points) < 3:
            continue

        delta = points - position
        cols = (delta @ orientation[:3]) / spacing[1]
        rows = (delta @ orientation[3:]) / spacing[0]

        for ax in axes:
            ax.plot(np.r_[cols, cols[0]], np.r_[rows, rows[0]],
                    color="yellow", linewidth=0.8)

    # Приближаем область ROI, чтобы были видны отдельные пиксели границы
    yy, xx = np.where(old | new)
    if len(xx):
        margin = 15
        for ax in axes:
            ax.set_xlim(max(0, xx.min() - margin),
                        min(shape[1] - 1, xx.max() + margin))
            ax.set_ylim(min(shape[0] - 1, yy.max() + margin),
                        max(0, yy.min() - margin))

    fig.suptitle(f"Срез {slice_idx}, ROI {roi_number}: {roi['name']}")
    plt.show()

    print(
        f"skimage: {old.sum()} пикселей; OpenCV: {new.sum()}; "
        f"добавлено: {added.sum()}; убрано: {removed.sum()}"
    )


patient, center = "Patient01", "Center2"


t = perf_counter()
cts, info = prepeare_CT_images(patient, center)
print("КТ:", perf_counter() - t)

t = perf_counter()
open_conture_dicom("RS.dcm", info)
print("Чтение контуров:", perf_counter() - t)

# t = perf_counter()
# masks = collect_masks(cts, info)

compare_mask_methods(cts, info, roi_number=2)  # PTV
compare_mask_methods(cts, info, roi_number=1)   # Rectum

# dir = "datasets"
# centre = "Center2"
# dcm_files = "Treatment plans"
# patient = "Patient01"
# filename = "RD.dcm"
# full_path = "/".join([".", dir, centre, dcm_files, patient, filename])
# with open("dicom_rd.txt", mode="w") as f:
#     data = pydicom.dcmread(full_path)
#     f.write(str(data))
