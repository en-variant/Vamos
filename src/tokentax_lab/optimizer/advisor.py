from __future__ import annotations


def advise_model(measured_models: list[dict], quality_floor: float) -> dict:
    """Choose the cheapest measured model that reaches a quality floor."""
    eligible = [
        row for row in measured_models
        if row.get("accuracy") is not None
        and row["accuracy"] >= quality_floor
        and row.get("cost_inr") is not None
    ]
    if not eligible:
        return {
            "model": None,
            "status": "no_measured_model_meets_quality_floor",
            "quality_floor": quality_floor,
        }
    selected = min(eligible, key=lambda row: row["cost_inr"])
    return {
        "model": selected["name"],
        "status": "measured",
        "quality_floor": quality_floor,
        "cost_inr": selected["cost_inr"],
        "accuracy": selected["accuracy"],
    }
