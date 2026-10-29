import json
import os
import json
import pprint

from time import perf_counter
from pathlib import Path
from collections import Counter

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

    # ДУБЛИКАТЫ В ТЕСТОВОМ ДАТАСЕТЕ!
    unique = {}
    for ds in info:
        unique.setdefault(str(ds.SOPInstanceUID), ds)

    info = sorted(
        unique.values(),
        key=lambda ds: float(ds.ImagePositionPatient[-1])
    )

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
            info["countors"][-1]["arrays"] = []
            info["countors"][-1]["types"] = []
            for seq in el.ContourSequence:
                if str(seq.ContourImageSequence[0].ReferencedSOPInstanceUID) == sop_uid:
                    info["countors"][-1]["arrays"].append(
                        np.array(seq.ContourData))
                    info["countors"][-1]["types"].append(
                        str(seq.ContourGeometricType))
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

def open_dose_dicom(filename: str, ct_info: dict[str: any]) -> None:
    dir = "datasets"
    centre = "Center2"
    dcm_files = "Treatment plans"
    patient = "Patient01"
    full_path = "/".join([".", dir, centre, dcm_files, patient, filename])
    dcm = pydicom.dcmread(full_path, force=True)

    info = {"dose": {"units": dcm.DoseUnits,
                     "type": dcm.DoseType,
                     "sum_type": dcm.DoseSummationType},
            "pixel_spacing": list(dcm.PixelSpacing),
            "patient_pos": list(dcm.ImagePositionPatient),
            "orientation": list(dcm.ImageOrientationPatient),
            "offset": list(dcm.GridFrameOffsetVector),
            "frame_uid": dcm.FrameOfReferenceUID,
            "plan_uid": dcm.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID,
            "struct_uid": dcm.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID,
            "DVH": {}}

    pixel_array = dcm.pixel_array * float(dcm.DoseGridScaling)

    for dvh_seq in dcm.DVHSequence:
        seq = dvh_seq.DVHReferencedROISequence[0]
        info["DVH"][seq.ReferencedROINumber] = {}
        info["DVH"][seq.ReferencedROINumber]["contribution_type"] = seq.DVHROIContributionType
        info["DVH"][seq.ReferencedROINumber]["units"] = dvh_seq.DoseUnits
        info["DVH"][seq.ReferencedROINumber]["type"] = dvh_seq.DoseType
        info["DVH"][seq.ReferencedROINumber]["scaling"] = dvh_seq.DVHDoseScaling
        info["DVH"][seq.ReferencedROINumber]["volume_units"] = dvh_seq.DVHVolumeUnits
        info["DVH"][seq.ReferencedROINumber]["bins"] = dvh_seq.DVHNumberOfBins
        info["DVH"][seq.ReferencedROINumber]["data"] = list(dvh_seq.DVHData)
        info["DVH"][seq.ReferencedROINumber]["min_dose"] = float(dvh_seq.DVHMinimumDose)
        info["DVH"][seq.ReferencedROINumber]["max_dose"] = float(dvh_seq.DVHMaximumDose)

    return pixel_array, info


def open_plan_dicom(filename: str, ct_info: dict[str: any]) -> None:
    dir = "datasets"
    centre = "Center2"
    dcm_files = "Treatment plans"
    patient = "Patient01"
    full_path = "/".join([".", dir, centre, dcm_files, patient, filename])
    dcm = pydicom.dcmread(full_path, force=True)

    info = {"sop": dcm.SOPInstanceUID,
            "frame_reference": dcm.FrameOfReferenceUID,
            "structure_uid": dcm.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID,
            "dose_reference": {},
            "fraction_group": {},
            "beam_sequence": {}}

    for dose_seq in dcm.DoseReferenceSequence:
        info["dose_reference"][dose_seq.DoseReferenceNumber] = {}
        temp = info["dose_reference"][dose_seq.DoseReferenceNumber]
        temp["dose_struct_type"] = str(dose_seq.DoseReferenceStructureType)
        temp["dose_type"] = str(dose_seq.DoseReferenceType)
        if temp["dose_type"] == "TARGET":
            for key, tag in {
                "delivery_max_dose": "DeliveryMaximumDose",
                "target_prescription_dose": "TargetPrescriptionDose",
                "target_max_dose": "TargetMaximumDose",
            }.items():
                value = getattr(dose_seq, tag, None)
                temp[key] = float(value) if value is not None else None

        if temp["dose_struct_type"] == "COORDINATES":
            temp["point_coords"] = [
                float(x) for x in dose_seq.DoseReferencePointCoordinates
            ]

    for frac_gr in dcm.FractionGroupSequence:
        info["fraction_group"][str(frac_gr.FractionGroupNumber)] = {}
        temp = info["fraction_group"][str(frac_gr.FractionGroupNumber)]
        temp["num_fracts"] = frac_gr.NumberOfFractionsPlanned
        temp["num_beans"] = frac_gr.NumberOfBeams 
        temp["beams"] = {}
        for ref in frac_gr.get("ReferencedBeamSequence", []):
            beam_num = str(ref.ReferencedBeamNumber)
            temp["beams"][beam_num] = {
                "dose": float(ref.BeamDose) if ref.get("BeamDose") is not None else None,
                "meterset": (
                    float(ref.BeamMeterset)
                    if ref.get("BeamMeterset") is not None else None
                ),
            }

    for beam_seq in dcm.BeamSequence:
        info["beam_sequence"][str(beam_seq.BeamNumber)] = {}
        temp = info["beam_sequence"][str(beam_seq.BeamNumber)]
        temp["beam_type"] = beam_seq.BeamType
        temp["rad_type"] = beam_seq.RadiationType
        temp["delivary_type"] = beam_seq.TreatmentDeliveryType
        temp["beam_num"] = beam_seq.BeamNumber

        temp["reg_imgs"] = [str(i.ReferencedSOPInstanceUID) for i in beam_seq.ReferencedReferenceImageSequence]

        temp["control_points"] = {}
        temp = temp["control_points"]
        
        for cp in beam_seq.ControlPointSequence:
            temp[str(cp.ControlPointIndex)] = {}
            cp_temp = temp[str(cp.ControlPointIndex)]
            for pos_seq in getattr(
                cp, "BeamLimitingDevicePositionSequence", []
            ):
                cp_temp[str(pos_seq.RTBeamLimitingDeviceType)] = [
                    float(x) for x in pos_seq.LeafJawPositions
                ]

            cp_temp["meterset_weight"] = cp.CumulativeMetersetWeight
            
            if cp.get("IsocenterPosition") is not None:
                cp_temp["isocenter_pos"] = [float(x) for x in cp.get("IsocenterPosition")]

            dose_ref = cp.get("ReferencedDoseReferenceSequence")
            if dose_ref is not None:
                cp_temp["dose_ref"] = [i.ReferencedDoseReferenceNumber for i in dose_ref]

            for i in ["gantry", "patient_sup", "table"]:
                cp_temp[i] = {}
                if i == "table":
                    for j in ["top"]:
                        cp_temp[i][j] = {}
                        if j == "top":
                            for k in ["eccentric", "vertical", "longitudinal", "lateral", "pitch"]:
                                cp_temp[i][j][k] = {}
            
            cp_temp["gantry"]["angle"] = cp.get("GantryAngle")
            cp_temp["gantry"]["direction"] = cp.get("GantryRotationDirection")            
            
            cp_temp["patient_sup"]["angle"] = cp.get("PatientSupportAngle")           
            cp_temp["patient_sup"]["direction"] = cp.get("PatientSupportRotationDirection")

            cp_temp["table"]["top"]["eccentric"]["angle"] = cp.get("TableTopEccentricAngle")
            cp_temp["table"]["top"]["vertical"]["pos"] = cp.get("TableTopVerticalPosition")
            cp_temp["table"]["top"]["longitudinal"]["pos"] = cp.get("TableTopLongitudinalPosition")
            cp_temp["table"]["top"]["lateral"]["pos"] = cp.get("TableTopLateralPosition")
            cp_temp["table"]["top"]["pitch"]["angle"] = cp.get("TableTopPitchAngle")

            cp_temp["table"]["top"]["eccentric"]["direction"] = cp.get("TableTopEccentricRotationDirection")
            cp_temp["table"]["top"]["pitch"]["direction"] = cp.get("TableTopPitchRotationDirection")
    return info

cts, info = prepeare_CT_images("Patient01", "Center2")
# open_conture_dicom("RS.dcm", info)
open_dose_dicom("RD.dcm", info)

patient, center = "Patient01", "Center2"


t = perf_counter()
cts, info = prepeare_CT_images(patient, center)
print("КТ:", perf_counter() - t)

t = perf_counter()
open_conture_dicom("RS.dcm", info)
print("Чтение контуров:", perf_counter() - t)

t = perf_counter()
masks = collect_masks(cts, info)
print("Чтение маск:", perf_counter() - t)

print(cts.shape)
print(masks.shape)