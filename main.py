import os
import pprint
import json

import pydicom
import matplotlib.pyplot as plt
import numpy as np


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
        "countors": []
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
            info["countors"][-1]["arrays"] = []
            info["countors"][-1]["types"] = []
            for seq in el.ContourSequence:
                if str(seq.ContourImageSequence[0].ReferencedSOPInstanceUID) == sop_uid:
                    info["countors"][-1]["arrays"].append(
                        np.array(seq.ContourData))
                    info["countors"][-1]["types"].append(
                        str(seq.ContourGeometricType))


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

cts, info = prepeare_CT_images("Patient01", "Center2")
# open_conture_dicom("RS.dcm", info)
open_dose_dicom("RD.dcm", info)
