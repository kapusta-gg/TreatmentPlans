from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = (
    "ptv_cm3",
    "rectum_cm3",
    "bladder_cm3",
    "rectum_ptv_overlap_pct",
    "bladder_ptv_overlap_pct",
    "ptv_length_mm",
)

FEATURES = {
    "rectum_dmean_gy": BASE_FEATURES + (
        "rectum_within_5mm_pct",
        "rectum_within_10mm_pct",
        "rectum_within_20mm_pct",
    ),
    "bladder_dmean_gy": BASE_FEATURES,
}


def make_model():
    return make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=3, weights="distance"),
    )


def check_data(df):
    required = {"patient"} | set(FEATURES) | {
        feature
        for columns in FEATURES.values()
        for feature in columns
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Нет столбцов: {sorted(missing)}")

    if len(df) < 4 or df["patient"].duplicated().any():
        raise ValueError("Нужно не меньше четырёх разных пациентов")

    numeric = df[list(required - {"patient"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("В признаках или дозах есть NaN/inf")


def evaluate_loo(df):
    results = pd.DataFrame({"patient": df["patient"]})

    for target, features in FEATURES.items():
        y = df[target].to_numpy(dtype=float)

        prediction = cross_val_predict(
            make_model(),
            df[list(features)].to_numpy(dtype=float),
            y,
            cv=LeaveOneOut(),
        )

        error = np.abs(y - prediction)

        results[f"{target}_true"] = y
        results[f"{target}_pred"] = prediction
        results[f"{target}_abs_error"] = error

        mae = error.mean()
        rmse = np.sqrt(np.mean((y - prediction) ** 2))
        print(f"{target}: MAE = {mae:.3f} Гр; RMSE = {rmse:.3f} Гр")

    return results


def fit_and_save(df, model_path):
    bundle = {"features": FEATURES, "models": {}}

    for target, features in FEATURES.items():
        model = make_model()
        model.fit(
            df[list(features)].to_numpy(dtype=float),
            df[target].to_numpy(dtype=float),
        )
        bundle["models"][target] = model

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def predict_dmean(geometry, model_path="output/k_neigh.joblib"):
    """geometry — словарь из extract_geometry_features()."""
    bundle = joblib.load(model_path)
    result = {}

    for target, features in bundle["features"].items():
        x = np.array(
            [[geometry[name] for name in features]],
            dtype=float,
        )
        result[target] = float(
            bundle["models"][target].predict(x)[0]
        )

    return result

