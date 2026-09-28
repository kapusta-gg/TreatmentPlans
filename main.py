import os


import pydicom
import matplotlib.pyplot as plt
import numpy as np

dir = "datasets"
centre = "Center1"
dcm_files = "CT images"
patient = "Patient01"
filename = "CT_15_01_1501023_DEIDENT_1894254.dcm"
full_path = "/".join([".", dir, centre, dcm_files, patient, filename])


def prepeare_CT_images(patient_dir: str, center: str) -> tuple[np.array, list]:
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
        "sop_uid": dcm.SOPInstanceUID
    } for dcm in info]
    return cts, plus_info


def open_conture_dicom(filename: str) -> None:
    # Готовим путь и  читаем
    dir = "datasets"
    centre = "Center2"
    dcm_files = "Treatment plans"
    patient = "Patient01"
    full_path = "/".join([".", dir, centre, dcm_files, patient, filename])
    data = pydicom.dcmread(full_path, force=True)
    
    # Подготовка словаря: Номер ROI -> Информация о ROI 
    roi_struct = {int(el.ROINumber): {"name": str(el.ROIName)} for el in data.StructureSetROISequence}
    roi = data.ROIContourSequence
    for el in roi:
        roi_struct[int(el.ReferencedROINumber)]["color"] = list(el.ROIDisplayColor)
    
#cts, info = prepeare_CT_images("Patient01", "Center2")
#print(info[10]["sop_uid"])
open_conture_dicom("RS.dcm")
