import json
import zipfile

import numpy as np
from scipy.ndimage import distance_transform_edt


def extract_geometry_features(npz_path, info_path, radii=(5, 10, 20)):
    with open(info_path, encoding="utf-8") as file:
        info = json.load(file)

    # Ключ roi_channels — индекс канала в masks, а не номер ROI.
    channels = {}
    for channel, roi in info["roi_channels"].items():
        channels.setdefault(roi["name"].casefold(), []).append(int(channel))

    selected = {}
    for name in ("ptv", "rectum", "bladder"):
        matches = channels.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"{name}: ожидался один ROI, найдено {matches}")
        selected[name] = matches[0]

    positions = np.asarray(info["ct"]["positions"], dtype=float)
    spacing = np.asarray(info["ct"]["pixel_spacing"], dtype=float)
    orientation = np.asarray(info["ct"]["orientation"], dtype=float)

    if not np.allclose(orientation, [1, 0, 0, 0, 1, 0]):
        raise ValueError("Расчёт расстояний пока поддерживает осевые КТ")

    z_diffs = np.diff(positions[:, 2])
    if z_diffs.size == 0:
        raise ValueError("Для расчёта нужен хотя бы два среза КТ")

    z_step = float(np.median(z_diffs))
    if z_step <= 0 or not np.allclose(z_diffs, z_step, atol=1e-3):
        raise ValueError("Неравномерный шаг срезов КТ")

    if not np.allclose(spacing, spacing[0]):
        raise ValueError("PixelSpacing меняется между срезами")

    voxel_cm3 = z_step * np.prod(spacing[0]) / 1000
    voxels = {"ptv": 0, "rectum": 0, "bladder": 0}
    overlap = {"rectum": 0, "bladder": 0}
    ptv_slices = []

    with zipfile.ZipFile(npz_path) as archive:
        with archive.open("masks.npy") as stream:
            version = np.lib.format.read_magic(stream)
            read_header = getattr(
                np.lib.format,
                f"read_array_header_{version[0]}_{version[1]}"
            )
            shape, fortran_order, dtype = read_header(stream)
            n_slices, n_roi, height, width = shape

            if fortran_order or dtype != np.dtype("bool"):
                raise ValueError("Ожидался массив масок bool в C-порядке")
            if n_slices != len(positions) or n_roi != len(info["roi_channels"]):
                raise ValueError("Форма масок не совпадает с info.json")
            if any(channel < 0 or channel >= n_roi for channel in selected.values()):
                raise ValueError("Индекс канала ROI выходит за границы masks")

            # Храним только три выбранных канала, а не весь массив masks.
            organ_masks = {
                name: np.empty((n_slices, height, width), dtype=bool)
                for name in selected
            }
            bytes_per_slice = n_roi * height * width

            for k in range(n_slices):
                raw = stream.read(bytes_per_slice)
                if len(raw) != bytes_per_slice:
                    raise EOFError("Файл масок закончился раньше ожидаемого")

                masks = np.frombuffer(raw, dtype=np.bool_).reshape(
                    n_roi, height, width
                )

                for name, channel in selected.items():
                    organ_masks[name][k] = masks[channel]

                ptv = masks[selected["ptv"]]
                ptv_count = int(np.count_nonzero(ptv))
                voxels["ptv"] += ptv_count

                if ptv_count:
                    ptv_slices.append(k)

                for name in ("rectum", "bladder"):
                    organ = masks[selected[name]]
                    voxels[name] += int(np.count_nonzero(organ))
                    overlap[name] += int(np.count_nonzero(ptv & organ))

    if not ptv_slices or any(count == 0 for count in voxels.values()):
        raise ValueError("Одна из основных масок пустая")

    # Убираем пустые края объёма перед расчётом расстояний.
    occupied = (
        organ_masks["ptv"]
        | organ_masks["rectum"]
        | organ_masks["bladder"]
    )
    coordinates = np.where(occupied)
    bounds = tuple(
        slice(int(axis.min()), int(axis.max()) + 1)
        for axis in coordinates
    )
    cropped = {
        name: mask[bounds].copy()
        for name, mask in organ_masks.items()
    }
    del organ_masks, occupied, coordinates

    # Оси массива: срез (Z), строка (Y), столбец (X).
    distances_mm = distance_transform_edt(
        ~cropped["ptv"],
        sampling=(z_step, spacing[0, 0], spacing[0, 1]),
    )

    proximity = {}
    for name in ("rectum", "bladder"):
        organ = cropped[name]

        for radius in radii:
            proximity[f"{name}_within_{radius}mm_pct"] = (
                100
                * np.count_nonzero(organ & (distances_mm <= radius))
                / voxels[name]
            )

    return {
        "ptv_cm3": voxels["ptv"] * voxel_cm3,
        "rectum_cm3": voxels["rectum"] * voxel_cm3,
        "bladder_cm3": voxels["bladder"] * voxel_cm3,
        "rectum_ptv_overlap_pct": 100 * overlap["rectum"] / voxels["rectum"],
        "bladder_ptv_overlap_pct": 100 * overlap["bladder"] / voxels["bladder"],
        "ptv_length_mm": (
            positions[ptv_slices[-1], 2]
            - positions[ptv_slices[0], 2]
            + z_step
        ),
        **proximity,
    }


def extract_dmean(npz_path, info_path):
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    channels = {}
    for key, roi in info["roi_channels"].items():
        channels.setdefault(roi["name"].casefold(), []).append(int(key))

    selected = {}
    for name in ("rectum", "bladder"):
        matches = channels.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"{name}: ожидался один ROI, найдено {matches}")
        selected[name] = matches[0]

    def read_header(stream):
        version = np.lib.format.read_magic(stream)
        reader = getattr(
            np.lib.format,
            f"read_array_header_{version[0]}_{version[1]}"
        )
        return reader(stream)

    sums = {name: 0.0 for name in selected}
    counts = {name: 0 for name in selected}

    with zipfile.ZipFile(npz_path) as archive:
        with (
            archive.open("masks.npy") as mask_file,
            archive.open("dose.npy") as dose_file,
            archive.open("dose_valid.npy") as valid_file,
        ):
            mask_shape, mask_order, mask_dtype = read_header(mask_file)
            dose_shape, dose_order, dose_dtype = read_header(dose_file)
            valid_shape, valid_order, valid_dtype = read_header(valid_file)

            n, n_roi, height, width = mask_shape
            if (
                mask_order or dose_order or valid_order
                or mask_dtype != np.dtype("bool")
                or valid_dtype != np.dtype("bool")
                or dose_shape != (n, height, width)
                or valid_shape != dose_shape
                or n_roi != len(info["roi_channels"])
            ):
                raise ValueError("Формы или типы массивов не совпадают")

            for _ in range(n):
                raw_masks = mask_file.read(n_roi * height * width)
                raw_dose = dose_file.read(height * width * dose_dtype.itemsize)
                raw_valid = valid_file.read(height * width)

                if (
                    len(raw_masks) != n_roi * height * width
                    or len(raw_dose) != height * width * dose_dtype.itemsize
                    or len(raw_valid) != height * width
                ):
                    raise EOFError("Массив в NPZ закончился раньше ожидаемого")

                masks = np.frombuffer(raw_masks, dtype=mask_dtype).reshape(
                    n_roi, height, width
                )
                dose = np.frombuffer(raw_dose, dtype=dose_dtype).reshape(
                    height, width
                )
                valid = np.frombuffer(raw_valid, dtype=valid_dtype).reshape(
                    height, width
                )

                for name, channel in selected.items():
                    organ = masks[channel]
                    if np.any(organ & ~valid):
                        raise ValueError(f"{name}: часть маски вне сетки дозы")

                    values = dose[organ]
                    sums[name] += float(values.sum(dtype=np.float64))
                    counts[name] += values.size

    if any(count == 0 for count in counts.values()):
        raise ValueError("Пустая маска органа")

    return {
        f"{name}_dmean_gy": sums[name] / counts[name]
        for name in selected
    }


def extract_plan_rows(info_path: Path):
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)

    patient = info_path.parent.name
    plan = info["plan"]
    beams = plan.get("beam_sequence", {})
    fraction_groups = plan.get("fraction_group", {})

    prescriptions = sorted({
        float(ref["target_prescription_dose"])
        for ref in plan.get("dose_reference", {}).values()
        if ref.get("dose_type") == "TARGET"
        and ref.get("target_prescription_dose") is not None
    })

    patient_row = {
        "patient": patient,
        "prescriptions_gy": json.dumps(prescriptions),
        "fractions_by_group": json.dumps({
            str(number): group.get("num_fracts")
            for number, group in fraction_groups.items()
        }),
        "n_beams_total": len(beams),
        "n_beams_treatment": sum(
            beam.get("delivary_type") == "TREATMENT"
            for beam in beams.values()
        ),
        "n_beams_setup": sum(
            beam.get("delivary_type") == "SETUP"
            for beam in beams.values()
        ),
    }

    beam_rows = []

    for beam_number, beam in beams.items():
        control_points = [
            cp
            for _, cp in sorted(
                beam.get("control_points", {}).items(),
                key=lambda item: int(item[0]),
            )
        ]

        angles = [
            float(cp["gantry"]["angle"])
            for cp in control_points
            if cp.get("gantry", {}).get("angle") is not None
        ]

        direction = next(
            (
                cp["gantry"]["direction"]
                for cp in control_points
                if cp.get("gantry", {}).get("direction") is not None
            ),
            None,
        )

        isocenter = next(
            (
                cp["isocenter_pos"]
                for cp in control_points
                if cp.get("isocenter_pos") is not None
            ),
            [None, None, None],
        )

        # Привязка пучка к группам фракций.
        references = {
            str(group_number): group["beams"][str(beam_number)]
            for group_number, group in fraction_groups.items()
            if str(beam_number) in group.get("beams", {})
        }

        beam_rows.append({
            "patient": patient,
            "beam_number": beam_number,
            "delivery_type": beam.get("delivary_type"),
            "beam_type": beam.get("beam_type"),
            "radiation_type": beam.get("rad_type"),
            "n_control_points": len(control_points),
            "gantry_start_deg": angles[0] if angles else None,
            "gantry_end_deg": angles[-1] if angles else None,
            "gantry_direction": direction,
            "isocenter_x_mm": isocenter[0],
            "isocenter_y_mm": isocenter[1],
            "isocenter_z_mm": isocenter[2],
            "beam_dose_by_group": json.dumps({
                group: ref.get("dose")
                for group, ref in references.items()
            }),
            "beam_meterset_by_group": json.dumps({
                group: ref.get("meterset")
                for group, ref in references.items()
            }),
        })

    return patient_row, beam_rows