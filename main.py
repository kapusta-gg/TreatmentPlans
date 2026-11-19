from collections import Counter
from pathlib import Path

import pandas as pd


BEAMS_CSV = Path("output/plan_beams.csv")
RESULT_CSV = Path("output/arc_template_check.csv")
HELDOUT_PATIENT = "Patient10"


def get_arc_pattern(group):
    """Одна пара лечебных дуг: CC, затем CW."""
    if len(group) != 2 or set(group["gantry_direction"]) != {"CC", "CW"}:
        raise ValueError(
            f"{group['patient'].iloc[0]}: ожидались две дуги CC и CW"
        )

    pattern = []

    for direction in ("CC", "CW"):
        beam = group[group["gantry_direction"] == direction].iloc[0]
        pattern.append((
            float(beam["gantry_start_deg"]),
            float(beam["gantry_end_deg"]),
            direction,
            int(beam["n_control_points"]),
        ))

    return tuple(pattern)


def format_pattern(pattern):
    return "; ".join(
        f"{direction}: {start:g}° → {end:g}°, {n_cp} CP"
        for start, end, direction, n_cp in pattern
    )


def main():
    beams = pd.read_csv(BEAMS_CSV)
    treatment = beams[beams["delivery_type"] == "TREATMENT"]

    patterns = {
        patient: get_arc_pattern(group)
        for patient, group in treatment.groupby("patient")
    }

    if HELDOUT_PATIENT not in patterns:
        raise ValueError(f"Не найден {HELDOUT_PATIENT}")

    rows = []

    for patient, actual in sorted(patterns.items()):
        training_patterns = [
            pattern
            for other_patient, pattern in patterns.items()
            if other_patient != patient
            and other_patient != HELDOUT_PATIENT
        ]

        predicted, frequency = Counter(
            training_patterns
        ).most_common(1)[0]

        rows.append({
            "patient": patient,
            "actual": format_pattern(actual),
            "predicted": format_pattern(predicted),
            "match": actual == predicted,
            "template_frequency": frequency,
            "training_patients": len(training_patterns),
        })

    result = pd.DataFrame(rows)
    result.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    training_check = result[
        result["patient"] != HELDOUT_PATIENT
    ]
    heldout_check = result[
        result["patient"] == HELDOUT_PATIENT
    ].iloc[0]

    print(result[
        ["patient", "match", "actual", "predicted"]
    ].to_string(index=False))

    print(
        f"\nПоочерёдная проверка остальных пациентов: "
        f"{training_check['match'].sum()}/{len(training_check)}"
    )
    print(
        f"{HELDOUT_PATIENT}: "
        f"{'совпало' if heldout_check['match'] else 'не совпало'}"
    )
    print(f"Сохранено: {RESULT_CSV}")


if __name__ == "__main__":
    main()