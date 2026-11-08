import json

import numpy as np
from pathlib import Path
from collections.abc import Mapping, Sequence
from decimal import Decimal

from checkers.check.filepath import check_filenames
from parser import dose, counter, plan, scan
from errors import *


def prepare_all_data(ct_dir: str, counters_filename: str, dose_filename: str, plan_filename: str) -> tuple[np.array, np.array, np.array, dict]:
    info = {}
    args = [ct_dir, counters_filename, dose_filename, plan_filename]

    check_filenames(*args)

    cts, info_ct = scan.prepeare_CT_images(ct_dir)
    info["ct"] = {
        "frame_uid": str(info_ct[0]["frame_uid"]),
        "series_uid": str(info_ct[0]["series_uid"]),
        "positions": [
            [float(v) for v in slice_info["patient_pos"]]
            for slice_info in info_ct
        ],
        "pixel_spacing": [
            [float(v) for v in s["pixel_spacing"]] for s in info_ct
        ],
        "orientation":   [
            [float(v) for v in s["orientation"]] for s in info_ct
        ],
        "sop_uids": [str(s["sop_uid"]) for s in info_ct]
    }

    counter.open_conture_dicom(counters_filename, info_ct)
    info["roi_channels"] = {}

    for channel, contour in enumerate(info_ct[0]["countors"]):
        roi_number = next(
            key for key in contour
            if isinstance(key, int)
        )

        info["roi_channels"][str(channel)] = {
            "roi_number": roi_number,
            "name": str(contour[roi_number]["name"]),
        }

    masks = counter.collect_masks(cts, info_ct)

    if len(set(i["frame_uid"] for i in info_ct)) != 1:
        raise ReferenceUIDError("CT not from one frame!")
    frame_uid = info_ct[0]["frame_uid"]

    dose_map, info_dose = dose.open_dose_dicom(
        dose_filename, expected_frame_uid=frame_uid)
    dose_on_ct, valid_mask = dose.resize(cts, info_ct, dose_map, info_dose)

    info_plan = plan.open_plan_dicom(
        plan_filename, expected_frame_uid=frame_uid)

    info["dose"] = info_dose
    info["plan"] = info_plan

    return cts, masks, dose_on_ct, info


def save_ct_and_masks(
        cts: np.ndarray, masks: np.ndarray, dose_on_ct: np.ndarray, filename: str) -> None:
    if cts.ndim != 3:
        raise ValueError(f"Ожидалась форма КТ (Z, H, W), получено {cts.shape}")

    if masks.ndim != 4 or masks.shape[0] != cts.shape[0] \
            or masks.shape[2:] != cts.shape[1:]:
        raise ValueError(
            f"Маски {masks.shape} не соответствуют КТ {cts.shape}"
        )

    if dose_on_ct.shape != cts.shape:
        raise ValueError(
            f"Доза {dose_on_ct.shape} не соответствует КТ {cts.shape}"
        )

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        ct=cts,
        masks=masks,
        dose=dose_on_ct
    )


def save_info_json(info: dict, filename: str) -> None:
    def json_ready(value):
        if isinstance(value, np.ndarray):
            return json_ready(value.tolist())
        if isinstance(value, np.generic):
            return json_ready(value.item())
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, Mapping):
            return {str(key): json_ready(item) for key, item in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Sequence):
            return [json_ready(item) for item in value]

        raise TypeError(f"Не удалось сохранить в JSON: {type(value).__name__}")

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_ready(info),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
