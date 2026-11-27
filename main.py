import matplotlib.pyplot as plt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from checkers.compare.dvh import calculate_dvh


CASES_CSV = Path("output/kbp_cases.csv")
OUTPUT_DIR = Path("output")

BASE_FEATURES = (
    "ptv_cm3",
    "rectum_cm3",
    "bladder_cm3",
    "rectum_ptv_overlap_pct",
    "bladder_ptv_overlap_pct",
    "ptv_length_mm",
)

FEATURES = {
    "Rectum": BASE_FEATURES,
    "Bladder": BASE_FEATURES,
}


def prescription_gy(patient):
    path = OUTPUT_DIR / patient / "info.json"
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)["plan"]

    values = {
        float(ref["target_prescription_dose"])
        for ref in plan["dose_reference"].values()
        if ref.get("target_prescription_dose") is not None
    }
    if len(values) != 1:
        raise ValueError(f"{patient}: не удалось определить назначенную дозу")
    return values.pop()


def weighted_curve(curves, distances):
    """Средняя DVH с весом, обратно пропорциональным расстоянию."""
    distances = np.asarray(distances, dtype=float)
    zero = distances <= 1e-12

    if zero.any():
        weights = zero.astype(float)
    else:
        weights = 1.0 / distances

    weights /= weights.sum()
    return np.average(curves, axis=0, weights=weights)


def curve_error(predicted, actual):
    """Средняя абсолютная ошибка V(D), в процентных пунктах."""
    return float(np.mean(np.abs(predicted - actual)))


df = pd.read_csv(CASES_CSV)
if df["patient"].duplicated().any():
    raise ValueError("В CSV повторяются пациенты")

# Первый опыт проводим при одинаковой назначенной дозе.
# Так не смешиваем абсолютные оси дозы разных протоколов.
df = df[
    df["patient"].map(
        lambda patient: np.isclose(prescription_gy(patient), 36.25)
    )
].reset_index(drop=True)

if len(df) < 7:
    raise ValueError("Для опыта с пятью кандидатами нужно хотя бы 7 пациентов")

print("Пациентов с назначением 36.25 Гр:", len(df))

# DVH рассчитываются один раз. Кривую проверяемого пациента ниже
# используем только при подсчёте ошибки, не при выборе соседей.
dvhs = {}
common_dose_axis = None

for patient in df["patient"]:
    dose_axis, curves = calculate_dvh(
        OUTPUT_DIR / patient / "data.npz",
        OUTPUT_DIR / patient / "info.json",
        ("Rectum", "Bladder"),
    )

    if common_dose_axis is None:
        common_dose_axis = dose_axis
    elif not np.array_equal(dose_axis, common_dose_axis):
        raise ValueError(f"{patient}: другая ось дозы")

    dvhs[patient] = curves

results = []

for organ, features in FEATURES.items():
    X = df[list(features)].to_numpy(dtype=float)
    Y = np.stack([dvhs[p][organ] for p in df["patient"]])
    patients = df["patient"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(f"{organ}: в признаках есть NaN/inf")

    for test_idx in range(len(df)):
        train_idx = np.array(
            [i for i in range(len(df)) if i != test_idx]
        )

        # Масштабирование обучаем заново без проверяемого пациента.
        train_X = X[train_idx]
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0)
        std[std == 0] = 1.0

        distances = np.linalg.norm(
            (train_X - X[test_idx]) / std,
            axis=1,
        )

        nearest = np.argsort(distances)
        top3 = nearest[:3]
        top5 = nearest[:5]

        baseline_3 = weighted_curve(
            Y[train_idx[top3]], distances[top3]
        )
        baseline_5 = weighted_curve(
            Y[train_idx[top5]], distances[top5]
        )

        # Сравниваем DVH только пяти исторических кандидатов.
        candidate_curves = Y[train_idx[top5]]
        dvh_distances = np.mean(
            np.abs(
                candidate_curves[:, None, :]
                - candidate_curves[None, :, :]
            ),
            axis=2,
        )

        # Медоид — кандидат с минимальной суммой расстояний
        # до остальных кандидатов по DVH.
        medoid = int(np.argmin(dvh_distances.sum(axis=1)))
        chosen_in_top5 = np.argsort(dvh_distances[medoid])[:3]
        chosen = top5[chosen_in_top5]

        dvh_selected = weighted_curve(
            Y[train_idx[chosen]], distances[chosen]
        )
        global_mean = Y[train_idx].mean(axis=0)
        actual = Y[test_idx]  # Только здесь открываем ответ.
        results.append({
            "patient": patients[test_idx],
            "organ": organ,
            "geometry_top3": ", ".join(
                patients[train_idx[top3]]
            ),
            "dvh_selected": ", ".join(
                patients[train_idx[chosen]]
            ),
            "error_geometry_3_pp": curve_error(baseline_3, actual),
            "error_geometry_5_pp": curve_error(baseline_5, actual),
            "error_dvh_selected_pp": curve_error(dvh_selected, actual),
            "error_global_mean_pp": curve_error(global_mean, actual),
        })

result = pd.DataFrame(results)
print(result.round(3).to_string(index=False))

print("\nСредняя ошибка по органам, процентные пункты:")
print(
    result.groupby("organ")[
        [
            "error_geometry_3_pp",
            "error_geometry_5_pp",
            "error_dvh_selected_pp",
        ]
    ].mean().round(3)
)

result.to_csv(
    "kbp_dvh_experiment.csv",
    index=False,
    encoding="utf-8-sig",
)
for organ, group in result.groupby("organ"):
    global_error = group["error_global_mean_pp"]
    geometry_error = group["error_geometry_3_pp"]

    print(f"\n{organ}")
    print(f"Средняя DVH: {global_error.mean():.3f} п.п.")
    print(f"3 анатомических соседа: {geometry_error.mean():.3f} п.п.")
    print(
        "Анатомический поиск лучше у "
        f"{(geometry_error < global_error).sum()}/{len(group)} пациентов"
    )

    print(
        group[
            ["patient", "error_global_mean_pp", "error_geometry_3_pp"]
        ].round(3).to_string(index=False)
    )

for patient in ("Patient02", "Patient17", "Patient18"):
    matches = np.flatnonzero(df["patient"].to_numpy() == patient)
    if len(matches) != 1:
        print(f"{patient}: нет в выборке")
        continue

    test_idx = int(matches[0])
    train_idx = np.array(
        [i for i in range(len(df)) if i != test_idx]
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), sharex=True, sharey=True
    )

    for ax, (organ, features) in zip(axes, FEATURES.items()):
        X = df[list(features)].to_numpy(dtype=float)
        Y = np.stack([dvhs[p][organ] for p in df["patient"]])

        train_X = X[train_idx]
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0)
        std[std == 0] = 1.0

        distances = np.linalg.norm(
            (train_X - X[test_idx]) / std,
            axis=1,
        )
        nearest = np.argsort(distances)[:3]

        predicted = weighted_curve(
            Y[train_idx[nearest]],
            distances[nearest],
        )
        global_mean = Y[train_idx].mean(axis=0)
        actual = Y[test_idx]

        error_knn = curve_error(predicted, actual)
        error_global = curve_error(global_mean, actual)

        ax.plot(
            common_dose_axis, actual,
            color="black", linewidth=2, label="Реальная DVH"
        )
        ax.plot(
            common_dose_axis, global_mean,
            color="gray", linestyle="--",
            label=f"Средняя: {error_global:.2f} п.п."
        )
        ax.plot(
            common_dose_axis, predicted,
            color="tab:blue",
            label=f"3 соседа: {error_knn:.2f} п.п."
        )
        ax.fill_between(
            common_dose_axis, actual, predicted,
            color="tab:blue", alpha=0.12
        )

        ax.set_title(organ)
        ax.set_xlabel("Доза, Гр")
        ax.set_xlim(common_dose_axis[0], common_dose_axis[-1])
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend()

        print(
            patient, organ, "соседи:",
            df.iloc[train_idx[nearest]]["patient"].tolist()
        )

    axes[0].set_ylabel("Объём с дозой ≥ D, %")
    fig.suptitle(patient)
    fig.tight_layout()
    plt.show()
