import os
import json
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

data = open_plan_dicom("RP.dcm", info)
pprint.pprint(data)