import pydicom

from errors import *


def optional_float(dataset, keyword):
    value = dataset.get(keyword)
    return float(value) if value is not None else None


def open_plan_dicom(filename: str, expected_frame_uid=None) -> dict:
    dcm = pydicom.dcmread(filename, force=True)

    info = {"sop": dcm.SOPInstanceUID,
            "frame_uid": str(dcm.FrameOfReferenceUID),
            "structure_uid": dcm.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID,
            "plan_label": str(dcm.get("RTPlanLabel", "")),
            "plan_intent": str(dcm.get("PlanIntent", "")),
            "plan_geometry": str(dcm.get("RTPlanGeometry", "")),
            "dose_reference": {},
            "fraction_group": {},
            "patient_setups": {},
            "beam_sequence": {}}

    if expected_frame_uid is not None:
        if expected_frame_uid != info["frame_uid"]:
            raise ReferenceUIDError("Wrong UID!")

    for dose_seq in dcm.get("DoseReferenceSequence", []):
        info["dose_reference"][dose_seq.DoseReferenceNumber] = {}
        temp = info["dose_reference"][dose_seq.DoseReferenceNumber]
        temp["dose_struct_type"] = str(dose_seq.DoseReferenceStructureType)
        temp["dose_type"] = str(dose_seq.DoseReferenceType)
        temp["description"] = str(dose_seq.get("DoseReferenceDescription", ""))
        target_roi = dose_seq.get("ReferencedROINumber")
        temp["referenced_roi_number"] = (
            int(target_roi) if target_roi is not None else None
        )
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

    for setup in dcm.get("PatientSetupSequence", []):
        position = setup.get("PatientPosition")
        info["patient_setups"][str(setup.PatientSetupNumber)] = {
            "patient_position": str(position) if position is not None else None,
        }

    for beam_seq in dcm.BeamSequence:
        info["beam_sequence"][str(beam_seq.BeamNumber)] = {}
        temp = info["beam_sequence"][str(beam_seq.BeamNumber)]
        temp["beam_type"] = beam_seq.BeamType
        temp["rad_type"] = beam_seq.RadiationType
        temp["delivary_type"] = beam_seq.TreatmentDeliveryType
        temp["beam_num"] = beam_seq.BeamNumber
        temp["machine_name"] = str(beam_seq.get("TreatmentMachineName", ""))
        setup_number = beam_seq.get("ReferencedPatientSetupNumber")
        temp["patient_setup_number"] = (
            str(setup_number) if setup_number is not None else None
        )
        temp["source_axis_distance_mm"] = optional_float(
            beam_seq, "SourceAxisDistance"
        )

        temp["reg_imgs"] = [str(i.ReferencedSOPInstanceUID) for i in beam_seq.get(
            "ReferencedReferenceImageSequence", [])]

        temp["control_points"] = {}

        temp["beam_limiting_devices"] = {}

        for device in beam_seq.get("BeamLimitingDeviceSequence", []):
            boundaries = device.get("LeafPositionBoundaries")
            temp["beam_limiting_devices"][str(device.RTBeamLimitingDeviceType)] = {
                "number_of_pairs": int(device.NumberOfLeafJawPairs),
                "leaf_boundaries_mm": (
                    [float(x) for x in boundaries]
                    if boundaries is not None else None
                ),
            }

        unit = beam_seq.get("PrimaryDosimeterUnit")
        temp["primary_dosimeter_unit"] = str(
            unit) if unit is not None else None

        final_weight = beam_seq.get("FinalCumulativeMetersetWeight")
        temp["final_cumulative_meterset_weight"] = (
            float(final_weight) if final_weight is not None else None
        )

        temp = temp["control_points"]

        for cp in beam_seq.ControlPointSequence:
            temp[str(cp.ControlPointIndex)] = {}

            cp_temp = temp[str(cp.ControlPointIndex)]
            cp_temp["collimator_angle_deg"] = optional_float(
                cp, "BeamLimitingDeviceAngle"
            )
            collimator_direction = cp.get(
                "BeamLimitingDeviceRotationDirection")
            cp_temp["collimator_direction"] = (
                str(collimator_direction)
                if collimator_direction is not None else None
            )
            cp_temp["table_roll_angle_deg"] = optional_float(
                cp, "TableTopRollAngle"
            )
            cp_temp["table_roll_direction"] = (
                str(cp.TableTopRollRotationDirection)
                if cp.get("TableTopRollRotationDirection") is not None else None
            )
            cp_temp["source_surface_distance_mm"] = optional_float(
                cp, "SourceToSurfaceDistance"
            )

            for dicom_name, output_name in (
                ("NominalBeamEnergy", "nominal_beam_energy"),
                ("DoseRateSet", "dose_rate_set"),
            ):
                value = cp.get(dicom_name)
                if value is not None:
                    cp_temp[output_name] = float(value)

            for pos_seq in getattr(
                cp, "BeamLimitingDevicePositionSequence", []
            ):
                cp_temp[str(pos_seq.RTBeamLimitingDeviceType)] = [
                    float(x) for x in pos_seq.LeafJawPositions
                ]

            cp_temp["meterset_weight"] = cp.CumulativeMetersetWeight

            if cp.get("IsocenterPosition") is not None:
                cp_temp["isocenter_pos"] = [
                    float(x) for x in cp.get("IsocenterPosition")]

            dose_ref = cp.get("ReferencedDoseReferenceSequence")
            if dose_ref is not None:
                cp_temp["dose_ref"] = [
                    i.ReferencedDoseReferenceNumber for i in dose_ref]
                cp_temp["dose_ref_coefficients"] = {
                    str(i.ReferencedDoseReferenceNumber): optional_float(
                        i, "CumulativeDoseReferenceCoefficient"
                    ) for i in dose_ref
                }

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
            cp_temp["patient_sup"]["direction"] = cp.get(
                "PatientSupportRotationDirection")

            cp_temp["table"]["top"]["eccentric"]["angle"] = cp.get(
                "TableTopEccentricAngle")
            cp_temp["table"]["top"]["vertical"]["pos"] = cp.get(
                "TableTopVerticalPosition")
            cp_temp["table"]["top"]["longitudinal"]["pos"] = cp.get(
                "TableTopLongitudinalPosition")
            cp_temp["table"]["top"]["lateral"]["pos"] = cp.get(
                "TableTopLateralPosition")
            cp_temp["table"]["top"]["pitch"]["angle"] = cp.get(
                "TableTopPitchAngle")

            cp_temp["table"]["top"]["eccentric"]["direction"] = cp.get(
                "TableTopEccentricRotationDirection")
            cp_temp["table"]["top"]["pitch"]["direction"] = cp.get(
                "TableTopPitchRotationDirection")
    return info
