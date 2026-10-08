import os
import pprint

import pydicom
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon


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
    roi_struct = {int(el.ROINumber): {"name": str(el.ROIName)} for el in data.StructureSetROISequence}
    roi = data.ROIContourSequence
    for info in ct_info:
        sop_uid = info["sop_uid"]
        for el in roi:
            roi_struct[int(el.ReferencedROINumber)]["color"] = list(el.ROIDisplayColor)
            info["countors"].append({int(el.ReferencedROINumber) : roi_struct[int(el.ReferencedROINumber)].copy()})
            info["countors"][-1][int(el.ReferencedROINumber)]["arrays"] = []
            info["countors"][-1][int(el.ReferencedROINumber)]["types"] = []
            for seq in el.ContourSequence:
                if str(seq.ContourImageSequence[0].ReferencedSOPInstanceUID) == sop_uid:
                     info["countors"][-1][int(el.ReferencedROINumber)]["arrays"].append(np.array(seq.ContourData))
                     info["countors"][-1][int(el.ReferencedROINumber)]["types"].append(str(seq.ContourGeometricType))
    
cts, info = prepeare_CT_images("Patient01", "Center2")
open_conture_dicom("RS.dcm", info)

# pprint.pprint(info[0]["countors"])
# data = info[0]["countors"][0][1]["arrays"][0].reshape(-1, 3)
# data = (data - info[0]["patient_pos"])[:, :2] 
# data = (data / info[0]["pixel_spacing"]).astype(np.int64)
# color = np.mean([int(i) for i in info[0]["countors"][0][1]["color"]])

#TODO: Переписать функцию на сохранение масок по всем сканам пациента
def show_countor(ct_scan, info, num=0):
    ct = ct_scan.copy()
    for ind, i in enumerate(info["countors"]):
        ind +=1 
        mask = np.zeros(ct_scan.shape, dtype=bool)
        temp = i[ind]

        for data in temp["arrays"]:
            data = data.reshape(-1, 3)
            data = (data - info["patient_pos"])[:, :2] 
            data = (data / info["pixel_spacing"])
            #color = np.mean([int(i) for i in temp["color"]])
            for x, y in data.astype(np.int64):
                if "PTV" in temp["name"]:
                    rr, cc = polygon(data[:, 1], data[:, 0], shape=mask.shape)
                    mask[rr, cc] = True
                    plt.imsave(f"masks/mask{num}.png", mask, cmap="gray", vmin=0, vmax=1)
                    color = -1000
                else:
                    color = 1400
                ct[y, x] = color

    plt.imshow(ct, cmap="gray", vmin=-200, vmax=300)
    plt.show()

show_countor(cts[200], info[200], num=200)

# dir = "datasets"
# centre = "Center2"
# dcm_files = "Treatment plans"
# patient = "Patient01"
# filename = "RD.dcm"
# full_path = "/".join([".", dir, centre, dcm_files, patient, filename])
# with open("dicom_rd.txt", mode="w") as f:
#     data = pydicom.dcmread(full_path)
#     f.write(str(data))