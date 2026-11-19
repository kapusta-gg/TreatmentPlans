from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("output")
FEATURES_PATH = OUTPUT_DIR / "kbp_cases_proximity.csv"
OFFSETS_PATH = OUTPUT_DIR / "isocenter_ptv_offsets.csv"

FEATURES = [
    "ptv_cm3",
    "ptv_length_mm",
    "rectum_cm3",
    "bladder_cm3",
    "rectum_ptv_overlap_pct",
    "bladder_ptv_overlap_pct",
]
TARGETS = ["dx_mm", "dy_mm", "dz_mm"]


def mean_3d_error(true, predicted):
    """Среднее расстояние между двумя точками, мм."""
    return np.linalg.norm(true - predicted, axis=1).mean()


def main():
    geometry = pd.read_csv(FEATURES_PATH)
    offsets = pd.read_csv(OFFSETS_PATH)

    if geometry["patient"].duplicated().any():
        raise ValueError("Повторяющиеся пациенты в таблице признаков")
    if offsets["patient"].duplicated().any():
        raise ValueError("Повторяющиеся пациенты в таблице изоцентров")

    data = geometry.merge(
        offsets[["patient", *TARGETS]],
        on="patient",
        how="inner",
        validate="one_to_one",
    )

    data = data[data["patient"] != "Patient10"].sort_values("patient")
    data = data.reset_index(drop=True)

    required = ["patient", *FEATURES, *TARGETS]
    if data[required].isna().any().any():
        raise ValueError("Есть пропуски в признаках или смещениях")

    if len(data) < 5:
        raise ValueError("Слишком мало общих пациентов в двух таблицах")

    x = data[FEATURES].to_numpy(dtype=np.float64)
    y = data[TARGETS].to_numpy(dtype=np.float64)

    pred_center = np.zeros_like(y)

    pred_mean = (y.sum(axis=0) - y) / (len(y) - 1)

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    pred_model = cross_val_predict(
        model, x, y, cv=LeaveOneOut()
    )

    predictions = {
        "center": pred_center,
        "mean_offset": pred_mean,
        "ridge": pred_model,
    }

    result = data[["patient", *TARGETS]].copy()

    for name, prediction in predictions.items():
        result[f"error_{name}_mm"] = np.linalg.norm(
            y - prediction, axis=1
        )

    for axis, index in zip(("dx", "dy", "dz"), range(3)):
        result[f"pred_{axis}_mm"] = pred_model[:, index]

    print(f"Пациентов: {len(data)}")
    for name, prediction in predictions.items():
        print(
            f"{name:>11}: "
            f"{mean_3d_error(y, prediction):.2f} мм"
        )

    improved = (
        result["error_ridge_mm"] < result["error_mean_offset_mm"]
    ).sum()
    print(f"Ridge лучше среднего смещения: {improved}/{len(result)}")

    print("\nОшибки по пациентам:")
    print(result[
        ["patient", "error_center_mm",
         "error_mean_offset_mm", "error_ridge_mm"]
    ].round(2).to_string(index=False))

    result.to_csv(
        OUTPUT_DIR / "isocenter_model_loo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Обучаем итоговый экземпляр на доступных пациентах.
    # Его качество указано выше по leave-one-out, а не по этим же данным.
    model.fit(x, y)
    joblib.dump(
        {"model": model, "features": FEATURES},
        OUTPUT_DIR / "isocenter_ridge.joblib",
    )


if __name__ == "__main__":
    main()