"""Кумулятивные DVH по маскам и дозе на общей сетке КТ."""

import json
import zipfile
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _read_array_header(stream):
    version = np.lib.format.read_magic(stream)
    reader = getattr(
        np.lib.format, f"read_array_header_{version[0]}_{version[1]}"
    )
    return reader(stream)


def calculate_dvh(
    npz_path: str | Path,
    info_path: str | Path,
    roi_names: Sequence[str],
    *,
    max_dose_gy: float = 50.0,
    dose_step_gy: float = 0.25,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Вернуть общую ось дозы и V(D) в процентах для выбранных ROI.

    V(D) — процент объёма ROI, получивший дозу не менее D Гр.
    Все воксели выбранных ROI должны находиться внутри dose valid mask.
    """
    if max_dose_gy <= 0 or dose_step_gy <= 0:
        raise ValueError("Граница дозы и шаг должны быть положительными")
    n_bins = round(max_dose_gy / dose_step_gy)
    if n_bins < 1 or not np.isclose(n_bins * dose_step_gy, max_dose_gy):
        raise ValueError("max_dose_gy должен делиться на dose_step_gy")
    if not roi_names or len({name.casefold() for name in roi_names}) != len(roi_names):
        raise ValueError("Передайте неповторяющиеся названия ROI")

    with open(info_path, encoding="utf-8") as file:
        info = json.load(file)
    if info["dose"]["dose"]["units"] != "GY":
        raise ValueError("Ожидалась доза в Гр")

    by_name = {}
    for channel, roi in info["roi_channels"].items():
        by_name.setdefault(roi["name"].casefold(), []).append(
            (roi["name"], int(channel))
        )
    selected = {}
    for name in roi_names:
        matches = by_name.get(name.casefold(), [])
        if len(matches) != 1:
            raise ValueError(
                f"{name}: ожидался один ROI, найдено {len(matches)}")
        _, channel = matches[0]
        # Результат индексируется именем из roi_names, даже если в JSON
        # встречается другое написание (например, RECTUM вместо Rectum).
        selected[name] = channel

    edges = np.arange(n_bins + 1, dtype=np.float64) * dose_step_gy
    histograms = {
        name: np.zeros(n_bins, dtype=np.int64) for name in selected
    }

    with zipfile.ZipFile(npz_path) as archive:
        valid_name = next(
            (name for name in ("valid_mask.npy", "dose_valid.npy")
             if name in archive.namelist()),
            None,
        )
        if valid_name is None:
            raise ValueError("В NPZ нет valid_mask.npy или dose_valid.npy")

        with (
            archive.open("masks.npy") as mask_file,
            archive.open("dose.npy") as dose_file,
            archive.open(valid_name) as valid_file,
        ):
            mask_shape, mask_order, mask_dtype = _read_array_header(mask_file)
            dose_shape, dose_order, dose_dtype = _read_array_header(dose_file)
            valid_shape, valid_order, valid_dtype = _read_array_header(
                valid_file)

            if len(mask_shape) != 4:
                raise ValueError("Ожидалась форма masks (Z, ROI, H, W)")
            n_slices, n_roi, height, width = mask_shape
            if (
                mask_order or dose_order or valid_order
                or mask_dtype != np.dtype("bool")
                or valid_dtype != np.dtype("bool")
                or dose_dtype.kind not in "fiu"
                or dose_shape != (n_slices, height, width)
                or valid_shape != dose_shape
                or n_roi != len(info["roi_channels"])
                or any(channel < 0 or channel >= n_roi
                       for channel in selected.values())
            ):
                raise ValueError(
                    "Формы или типы массивов не совпадают с info.json")

            pixels = height * width
            for k in range(n_slices):
                mask_bytes = mask_file.read(n_roi * pixels)
                dose_bytes = dose_file.read(pixels * dose_dtype.itemsize)
                valid_bytes = valid_file.read(pixels)
                if (
                    len(mask_bytes) != n_roi * pixels
                    or len(dose_bytes) != pixels * dose_dtype.itemsize
                    or len(valid_bytes) != pixels
                ):
                    raise EOFError(f"NPZ закончился на срезе {k}")

                masks = np.frombuffer(mask_bytes, dtype=np.bool_).reshape(
                    n_roi, height, width
                )
                dose = np.frombuffer(dose_bytes, dtype=dose_dtype).reshape(
                    height, width
                )
                valid = np.frombuffer(valid_bytes, dtype=np.bool_).reshape(
                    height, width
                )

                for name, channel in selected.items():
                    organ = masks[channel]
                    if np.any(organ & ~valid):
                        raise ValueError(
                            f"{name}: есть воксели вне сетки дозы, срез {k}"
                        )
                    values = dose[organ]
                    if not values.size:
                        continue
                    if (
                        not np.isfinite(values).all()
                        or values.min() < 0
                        or values.max() > edges[-1]
                    ):
                        raise ValueError(
                            f"{name}: некорректная доза или превышен "
                            f"max_dose_gy={max_dose_gy}; срез {k}"
                        )
                    histograms[name] += np.histogram(values, bins=edges)[0]

    curves = {}
    for name, counts in histograms.items():
        total = int(counts.sum())
        if total == 0:
            raise ValueError(f"{name}: маска пустая")
        curves[name] = 100.0 * np.cumsum(counts[::-1])[::-1] / total

    return edges[:-1], curves


def plot_dvh(
    dose_gy: np.ndarray,
    curves: dict[str, np.ndarray],
    *,
    show: bool = True,
):
    """Показать кумулятивные DVH нескольких ROI на одном графике.

    Возвращает (fig, ax), чтобы при необходимости сохранить fig.savefig(...).
    """
    dose_gy = np.asarray(dose_gy, dtype=float)
    if dose_gy.ndim != 1 or not dose_gy.size or not np.isfinite(dose_gy).all():
        raise ValueError("Ось дозы должна быть непустым одномерным массивом")
    if not curves:
        raise ValueError("Нет кривых для отображения")

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, curve in curves.items():
        volume_pct = np.asarray(curve, dtype=float)
        if volume_pct.shape != dose_gy.shape or not np.isfinite(volume_pct).all():
            plt.close(fig)
            raise ValueError(f"{name}: форма DVH не совпадает с осью дозы")
        ax.plot(dose_gy, volume_pct, label=name)

    ax.set(xlabel="Доза, Гр", ylabel="Объём ≥ D, %", ylim=(0, 100))
    ax.set_xlim(left=0, right=float(dose_gy[-1]))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
