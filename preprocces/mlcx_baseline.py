import json
from pathlib import Path

import numpy as np


PATIENTS = [f"Patient{i:02d}" for i in range(1, 7)]


def load_plan(patient, output_dir):
    path = output_dir / patient / "info.json"
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)["plan"]

    groups = list(plan["fraction_group"].values())
    if len(groups) != 1 or int(groups[0]["num_fracts"]) != 5:
        raise ValueError(f"{patient}: ожидались 5 фракций в одной группе")

    prescriptions = {
        float(ref["target_prescription_dose"])
        for ref in plan["dose_reference"].values()
        if ref.get("target_prescription_dose") is not None
    }
    if len(prescriptions) != 1 or not np.isclose(
        next(iter(prescriptions)), 36.25
    ):
        raise ValueError(f"{patient}: другая целевая доза: {prescriptions}")

    result = {}

    for beam in plan["beam_sequence"].values():
        if beam.get("delivary_type") != "TREATMENT":
            continue

        points = [
            cp
            for _, cp in sorted(
                beam["control_points"].items(),
                key=lambda item: int(item[0]),
            )
        ]

        directions = {
            cp["gantry"]["direction"]
            for cp in points
            if cp["gantry"].get("direction") in {"CC", "CW"}
        }
        if len(directions) != 1:
            raise ValueError(f"{patient}: не удалось определить направление дуги")
        direction = directions.pop()

        energies = {
            float(cp["nominal_beam_energy"])
            for cp in points
            if cp.get("nominal_beam_energy") is not None
        }
        dose_rates = {
            float(cp["dose_rate_set"])
            for cp in points
            if cp.get("dose_rate_set") is not None
        }
        if energies != {6.0} or dose_rates != {600.0}:
            raise ValueError(
                f"{patient}, {direction}: энергия {energies}, "
                f"скорость дозы {dose_rates}"
            )

        device = beam["beam_limiting_devices"]["MLCX"]
        pairs = int(device["number_of_pairs"])
        boundaries = np.asarray(device["leaf_boundaries_mm"], dtype=float)

        angles = np.asarray(
            [cp["gantry"]["angle"] for cp in points], dtype=float
        )
        mlcx = np.asarray([cp["MLCX"] for cp in points], dtype=float)

        if mlcx.shape != (178, 2 * pairs):
            raise ValueError(
                f"{patient}, {direction}: форма MLCX {mlcx.shape}"
            )

        if direction in result:
            raise ValueError(f"{patient}: две дуги {direction}")

        result[direction] = {
            "angles": angles,
            "mlcx": mlcx,
            "pairs": pairs,
            "boundaries": boundaries,
        }

    if set(result) != {"CC", "CW"}:
        raise ValueError(f"{patient}: ожидались дуги CC и CW")

    return result


def compare_geometry(plans):
    reference = plans[PATIENTS[0]]

    for patient, beams in plans.items():
        for direction in ("CC", "CW"):
            current = beams[direction]
            expected = reference[direction]

            if current["pairs"] != expected["pairs"]:
                raise ValueError(f"{patient}, {direction}: другое число пар MLCX")

            if not np.allclose(
                current["boundaries"], expected["boundaries"]
            ):
                raise ValueError(f"{patient}, {direction}: другие границы пластин")

            if not np.allclose(
                current["angles"], expected["angles"], atol=1e-4
            ):
                raise ValueError(
                    f"{patient}, {direction}: не совпадают углы "
                    "контрольных точек"
                )


def measure(actual, predicted, pairs):
    # Первые pairs значений — один банк пластин,
    # следующие pairs — второй банк.
    true_gap = actual[..., pairs:] - actual[..., :pairs]
    pred_gap = predicted[..., pairs:] - predicted[..., :pairs]

    # Отдельно смотрим пары, открытые хотя бы в одном из двух планов.
    active = (true_gap > 1.0) | (pred_gap > 1.0)

    return {
        "leaf_mae_mm": float(np.mean(np.abs(actual - predicted))),
        "gap_mae_mm": float(np.mean(np.abs(true_gap - pred_gap))),
        "active_gap_mae_mm": float(
            np.mean(np.abs(true_gap - pred_gap)[active])
        ),
    }


def aperture_areas(actual, predicted, pairs, boundaries):
    # Для каждой пары пластин: интервал от первого банка до второго.
    true_left = actual[..., :pairs]
    true_right = actual[..., pairs:]
    pred_left = predicted[..., :pairs]
    pred_right = predicted[..., pairs:]

    true_gap = np.maximum(true_right - true_left, 0)
    pred_gap = np.maximum(pred_right - pred_left, 0)

    overlap = np.maximum(
        np.minimum(true_right, pred_right)
        - np.maximum(true_left, pred_left),
        0,
    )

    # Ширина каждой пары пластин по второй оси, мм.
    leaf_widths = np.diff(boundaries)

    true_area = np.sum(true_gap * leaf_widths)
    pred_area = np.sum(pred_gap * leaf_widths)
    intersection = np.sum(overlap * leaf_widths)
    union = true_area + pred_area - intersection

    return intersection, union, true_area, pred_area


def main():
    output_dir = Path("output")
    plans = {
        patient: load_plan(patient, output_dir)
        for patient in PATIENTS
    }
    compare_geometry(plans)

    results = {"mean": [], "median": []}

    print("Пациент   Шаблон    MLCX MAE   Зазор MAE   Зазор открытых пар")
    print("                         мм         мм                 мм")

    for patient in PATIENTS:
        for method in ("mean", "median"):
            scores = []

            for direction in ("CC", "CW"):
                train = np.stack([
                    plans[other][direction]["mlcx"]
                    for other in PATIENTS
                    if other != patient
                ])

                if method == "mean":
                    predicted = train.mean(axis=0)
                else:
                    predicted = np.median(train, axis=0)

                actual = plans[patient][direction]["mlcx"]
                pairs = plans[patient][direction]["pairs"]
                scores.append(measure(actual, predicted, pairs))

            average = {
                key: np.mean([score[key] for score in scores])
                for key in scores[0]
            }
            results[method].append(average)

            print(
                f"{patient:10} {method:8} "
                f"{average['leaf_mae_mm']:8.2f} "
                f"{average['gap_mae_mm']:11.2f} "
                f"{average['active_gap_mae_mm']:18.2f}"
            )

    print("\nСредняя ошибка по шести пациентам:")
    for method, scores in results.items():
        print(
            f"{method:8} "
            f"MLCX: {np.mean([s['leaf_mae_mm'] for s in scores]):.2f} мм; "
            f"зазор открытых пар: "
            f"{np.mean([s['active_gap_mae_mm'] for s in scores]):.2f} мм"
        )
    print("\nСовпадение отверстий среднего шаблона:")

    for patient in PATIENTS:
        intersection = 0.0
        union = 0.0
        true_area = 0.0
        pred_area = 0.0

        for direction in ("CC", "CW"):
            train = np.stack([
                plans[other][direction]["mlcx"]
                for other in PATIENTS
                if other != patient
            ])
            predicted = train.mean(axis=0)

            beam = plans[patient][direction]
            values = aperture_areas(
                actual=beam["mlcx"],
                predicted=predicted,
                pairs=beam["pairs"],
                boundaries=beam["boundaries"],
            )

            intersection += values[0]
            union += values[1]
            true_area += values[2]
            pred_area += values[3]

        iou = intersection / union
        excess_pct = 100 * (pred_area - intersection) / pred_area
        missed_pct = 100 * (true_area - intersection) / true_area

        print(
            f"{patient}: IoU={iou:.3f}, "
            f"лишняя площадь={excess_pct:.1f}%, "
            f"пропущенная площадь={missed_pct:.1f}%"
        )


    print("\nПеренос плана другого пациента:")

    for patient in PATIENTS:
        candidates = []

        for donor in PATIENTS:
            if donor == patient:
                continue

            intersection = 0.0
            union = 0.0

            for direction in ("CC", "CW"):
                target = plans[patient][direction]
                source = plans[donor][direction]

                areas = aperture_areas(
                    actual=target["mlcx"],
                    predicted=source["mlcx"],
                    pairs=target["pairs"],
                    boundaries=target["boundaries"],
                )
                intersection += areas[0]
                union += areas[1]

            candidates.append((donor, intersection / union))

        best_donor, best_iou = max(candidates, key=lambda item: item[1])
        average_iou = np.mean([iou for _, iou in candidates])

        print(
            f"{patient}: средний IoU доноров={average_iou:.3f}, "
            f"лучший донор={best_donor} (IoU={best_iou:.3f})"
        )