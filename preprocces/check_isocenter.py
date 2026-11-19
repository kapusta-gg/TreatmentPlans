import zipfile

import numpy as np



def get_ptv_channel(info):
    matches = [
        int(channel)
        for channel, roi in info["roi_channels"].items()
        if roi["name"].casefold() == "ptv"
    ]
    return matches[0] if len(matches) == 1 else None


def get_plan_isocenter(info):
    positions = []

    for beam in info["plan"]["beam_sequence"].values():
        if beam.get("delivary_type") != "TREATMENT":
            continue

        for cp in beam.get("control_points", {}).values():
            if cp.get("isocenter_pos") is not None:
                positions.append(cp["isocenter_pos"])

    if not positions:
        raise ValueError("У лечебных пучков не найден изоцентр")

    positions = np.asarray(positions, dtype=float)
    if not np.allclose(positions, positions[0], atol=1e-3):
        raise ValueError("У лечебных пучков разные изоцентры")

    return positions[0]


def get_ptv_centroid(npz_path, info, channel):
    ct = info["ct"]
    positions = np.asarray(ct["positions"], dtype=float)
    spacing = np.asarray(ct["pixel_spacing"], dtype=float)
    orientation = np.asarray(ct["orientation"], dtype=float)

    if not np.allclose(spacing, spacing[0]):
        raise ValueError("PixelSpacing меняется между срезами")

    slice_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if not np.allclose(slice_steps, slice_steps[0], atol=1e-3):
        raise ValueError("Неравномерный шаг срезов КТ")

    coordinate_sum = np.zeros(3, dtype=np.float64)
    voxel_count = 0

    with zipfile.ZipFile(npz_path) as archive:
        with archive.open("masks.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = getattr(
                np.lib.format,
                f"read_array_header_{version[0]}_{version[1]}",
            )
            shape, fortran_order, dtype = reader(stream)

            n_slices, n_roi, height, width = shape

            if (
                fortran_order
                or dtype != np.dtype("bool")
                or n_slices != len(positions)
                or n_roi != len(info["roi_channels"])
            ):
                raise ValueError("Форма или тип masks.npy не совпадает с info.json")

            pixels_per_mask = height * width

            for k in range(n_slices):
                # Пропускаем каналы перед PTV.
                stream.seek(channel * pixels_per_mask, 1)

                raw = stream.read(pixels_per_mask)
                if len(raw) != pixels_per_mask:
                    raise EOFError("masks.npy закончился раньше ожидаемого")

                mask = np.frombuffer(
                    raw, dtype=np.bool_
                ).reshape(height, width)

                # Пропускаем оставшиеся каналы данного среза.
                stream.seek(
                    (n_roi - channel - 1) * pixels_per_mask, 1
                )

                rows, cols = np.nonzero(mask)
                count = rows.size
                if count == 0:
                    continue

                # Координаты центров вокселей в системе пациента.
                coordinate_sum += (
                    count * positions[k]
                    + cols.sum() * spacing[k, 1] * orientation[k, :3]
                    + rows.sum() * spacing[k, 0] * orientation[k, 3:]
                )
                voxel_count += count

    if voxel_count == 0:
        raise ValueError("Маска PTV пустая")

    return coordinate_sum / voxel_count


