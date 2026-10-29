import json

import numpy as np
from pathlib import Path

from parser import dose, counter, plan, scan

def prepare_all_data(ct_dir: str, counters_filename: str, dose_filename: str, plan_filename: str) -> None:
    args = [ct_dir, counters_filename, dose_filename, plan_filename]
    
    cts, info_ct = scan.prepeare_CT_images(ct_dir)
    counter.open_conture_dicom(counters_filename, info_ct)

    frame_uid = info_ct[0]["frame_uid"]
    
    dose_map, info_dose = dose.open_dose_dicom(dose_filename, expected_frame_uid=frame_uid)
    info_plan = plan.open_plan_dicom(plan_filename, expected_frame_uid=frame_uid)
    print("Done!")



def save_ct_and_masks(cts: np.ndarray, masks: np.ndarray, filename: str) -> None:
    if cts.ndim != 3:
        raise ValueError(
            f"Ожидалась форма КТ (срезы, H, W), получено {cts.shape}")

    if masks.ndim != 4:
        raise ValueError(
            f"Ожидалась форма масок (срезы, ROI, H, W), получено {masks.shape}"
        )

    if masks.shape[0] != cts.shape[0] or masks.shape[2:] != cts.shape[1:]:
        raise ValueError(
            "Число срезов или размер изображения у КТ и масок не совпадают")

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ct=cts, masks=masks)


def save_info_json(info: list, filename: str) -> None:
    rois = []
    for channel, item in enumerate(info[0]["countors"]):
        roi_number, roi_data = next(iter(item.items()))
        rois.append({
            "channel": channel,  # индекс в masks[:, channel]
            "roi_number": int(roi_number),
            "name": roi_data["name"],
        })

    metadata = {
        "rois": rois,
        "slices": [
            {
                "position": list(slice_info["patient_pos"]),
                "pixel_spacing": list(slice_info["pixel_spacing"]),
                "orientation": list(slice_info["orientation"]),
                "sop_uid": str(slice_info["sop_uid"]),
                "series_uid": str(slice_info["series_uid"]),
                "frame_uid": str(slice_info["frame_uid"]),
            }
            for slice_info in info
        ],
    }

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)