import os
import pprint

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
    roi_struct = {int(el.ROINumber): {"name": str(el.ROIName)} for el in data.StructureSetROISequence}
    roi = data.ROIContourSequence
    for info in ct_info:
        sop_uid = info["sop_uid"]
        for el in roi:
            roi_struct[int(el.ReferencedROINumber)]["color"] = list(el.ROIDisplayColor)
            info["countors"].append({int(el.ReferencedROINumber) : roi_struct[int(el.ReferencedROINumber)].copy()})
            info["countors"][-1]["arrays"] = []
            info["countors"][-1]["types"] = []
            for seq in el.ContourSequence:
                if str(seq.ContourImageSequence[0].ReferencedSOPInstanceUID) == sop_uid:
                     info["countors"][-1]["arrays"].append(np.array(seq.ContourData))
                     info["countors"][-1]["types"].append(str(seq.ContourGeometricType))
    
cts, info = prepeare_CT_images("Patient01", "Center2")
open_conture_dicom("RS.dcm", info)
pprint.pprint(info[0])

dir = "datasets"
centre = "Center2"
dcm_files = "Treatment plans"
patient = "Patient01"
filename = "RD.dcm"
full_path = "/".join([".", dir, centre, dcm_files, patient, filename])
with open("dicom_rd.txt", mode="w") as f:
    data = pydicom.dcmread(full_path)
    f.write(str(data))