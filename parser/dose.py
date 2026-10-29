import pydicom
import numpy as np


def open_dose_dicom(filename: str) -> tuple[np.array, dict]:
    dcm = pydicom.dcmread(filename, force=True)

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
