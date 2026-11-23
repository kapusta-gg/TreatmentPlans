import pydicom
import numpy as np
from scipy.ndimage import map_coordinates

from errors import *


def open_dose_dicom(filename: str, expected_frame_uid=None) -> tuple[np.array, dict]:
    dcm = pydicom.dcmread(filename, force=True)

    temp = dcm.get("ReferencedStructureSetSequence", "")
    if temp:
        temp = temp[0].ReferencedSOPInstanceUID
    info = {"dose": {"units": dcm.DoseUnits,
                     "type": dcm.DoseType,
                     "sum_type": dcm.DoseSummationType},
            "pixel_spacing": list(dcm.PixelSpacing),
            "patient_pos": np.asarray(dcm.ImagePositionPatient, dtype=float),
            "orientation": list(dcm.ImageOrientationPatient),
            "offset":  np.asarray(dcm.GridFrameOffsetVector, dtype=float),
            "frame_uid": str(dcm.FrameOfReferenceUID),
            "plan_uid": dcm.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID,
            "struct_uid": temp,
            "DVH": {}}

    if expected_frame_uid is not None:
        if expected_frame_uid != info["frame_uid"]:
            raise ReferenceUIDError("Wrong UID!")

    pixel_array = dcm.pixel_array * float(dcm.DoseGridScaling)

    for dvh_seq in dcm.get("DVHSequence", []):
        seq = dvh_seq.DVHReferencedROISequence[0]
        info["DVH"][seq.ReferencedROINumber] = {}
        info["DVH"][seq.ReferencedROINumber]["contribution_type"] = seq.DVHROIContributionType
        info["DVH"][seq.ReferencedROINumber]["units"] = dvh_seq.DoseUnits
        info["DVH"][seq.ReferencedROINumber]["type"] = dvh_seq.DoseType
        info["DVH"][seq.ReferencedROINumber]["scaling"] = dvh_seq.DVHDoseScaling
        info["DVH"][seq.ReferencedROINumber]["volume_units"] = dvh_seq.DVHVolumeUnits
        info["DVH"][seq.ReferencedROINumber]["bins"] = dvh_seq.DVHNumberOfBins
        info["DVH"][seq.ReferencedROINumber]["data"] = list(dvh_seq.DVHData)
        info["DVH"][seq.ReferencedROINumber]["min_dose"] = float(
            dvh_seq.DVHMinimumDose)
        info["DVH"][seq.ReferencedROINumber]["max_dose"] = float(
            dvh_seq.DVHMaximumDose)

    return pixel_array, info


def resize(cts, info_ct, dose_map, info_dose):
    if len(cts) != len(info_ct):
        raise ValueError("Число срезов КТ и записей info_ct не совпадает")

    dose_map = np.asarray(dose_map, dtype=np.float32)
    dose_on_ct = np.zeros(cts.shape, dtype=np.float32)
    valid_mask = np.zeros(cts.shape, dtype=bool)

    dose_pos = np.asarray(info_dose["patient_pos"], dtype=float)
    dose_spacing = np.asarray(info_dose["pixel_spacing"], dtype=float)
    offsets = np.asarray(info_dose["offset"], dtype=float)

    if len(offsets) != len(dose_map):
        raise ValueError("Число смещений не совпадает с числом кадров дозы")

    axial = [1, 0, 0, 0, 1, 0]
    if not np.allclose(info_dose["orientation"], axial):
        raise ValueError("Эта версия resize требует осевой ориентации дозы")
    if any(not np.allclose(s["orientation"], axial) for s in info_ct):
        raise ValueError("Эта версия resize требует осевой ориентации КТ")

    if np.isclose(offsets[0], 0):
        dose_z = dose_pos[2] + offsets
    elif np.isclose(offsets[0], dose_pos[2]):
        dose_z = offsets.copy()
    else:
        raise ValueError("Не удалось определить тип GridFrameOffsetVector")

    if dose_z[0] > dose_z[-1]:
        dose_z = dose_z[::-1]
        dose_map = dose_map[::-1]

    if not np.all(np.diff(dose_z) > 0):
        raise ValueError("Кадры дозы должны идти по возрастанию Z")

    dose_rows, dose_cols = dose_map.shape[1:]
    rows, cols = np.indices(cts.shape[1:], dtype=float)

    for k, info in enumerate(info_ct):
        ct_pos = np.asarray(info["patient_pos"], dtype=float)
        ct_spacing = np.asarray(info["pixel_spacing"], dtype=float)

        x = ct_pos[0] + cols * ct_spacing[1]
        y = ct_pos[1] + rows * ct_spacing[0]

        dose_col = (x - dose_pos[0]) / dose_spacing[1]
        dose_row = (y - dose_pos[1]) / dose_spacing[0]

        dose_frame = np.interp(
            ct_pos[2],
            dose_z,
            np.arange(len(dose_z)),
            left=-1,
            right=-1,
        )

        valid = (
            (dose_frame >= 0)
            & (dose_row >= 0) & (dose_row <= dose_rows - 1)
            & (dose_col >= 0) & (dose_col <= dose_cols - 1)
        )
        valid_mask[k] = valid

        # Теперь передаём всю 3D-карту: кадр, строка, столбец.
        frame_coords = np.full(dose_row.shape, dose_frame, dtype=float)

        map_coordinates(
            dose_map,
            [frame_coords, dose_row, dose_col],
            output=dose_on_ct[k],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        dose_on_ct[k][~valid] = 0

    return dose_on_ct, valid_mask
