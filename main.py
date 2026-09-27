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


def prepeare_CT_images(patient_dir: str, center: str) -> np.array:
    dir = "datasets"
    dcm_files = "CT images"
    path = os.path.join(*(dir, center, dcm_files, patient_dir))
    info = [pydicom.dcmread(os.path.join(*(path, dcm)))
            for dcm in os.listdir(path)]
    info.sort(key=lambda x: x.ImagePositionPatient[-1])
    cts = np.stack([dcm.pixel_array * dcm.RescaleSlope +
                    dcm.RescaleIntercept for dcm in info], axis=0)
    plus_info = [{
        "slice_thickness": float(dcm.SliceThickness),
        "pixel_spacing": np.asarray(dcm.PixelSpacing, dtype=float),
        "positions": np.asarray(
            [dcm.ImagePositionPatient for dcm in info], dtype=float
        ),
        "orientation": np.asarray(dcm.ImageOrientationPatient, dtype=float),
        "series_uid": dcm.SeriesInstanceUID,
        "frame_uid": dcm.FrameOfReferenceUID
    } for dcm in info]
    return cts, plus_info


def open_conture_dicom(filename: str) -> None:
    dir = "datasets"
    centre = "Center2"
    dcm_files = "Treatment plans"
    patient = "Patient01"
    full_path = "/".join([".", dir, centre, dcm_files, patient, filename])
    data = pydicom.dcmread(full_path, force=True)

    names = [str(roi.ROIName) for roi in data.StructureSetROISequence]
    names.sort()
    print(names)
    # plt.imshow(data.pixel_array, cmap=plt.cm.bone)
    # plt.show()


prepeare_CT_images("Patient03", "Center1")
