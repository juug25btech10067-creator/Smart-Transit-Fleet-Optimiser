"""Convert predicted passenger demand into route-level fleet recommendations."""

import math
import pandas as pd

def recommend_buses(
    predictions: pd.DataFrame,
    bus_capacity: int = 50,
    target_load_factor: float = 0.80,
    current_buses: int = 4,
) -> pd.DataFrame:
    if not 0 < target_load_factor <= 1:
        raise ValueError("target_load_factor must be between 0 and 1.")

    capacity_per_bus = bus_capacity * target_load_factor
    out = predictions.copy()

    out["recommended_buses"] = (
        out["predicted_passengers"] / capacity_per_bus
    ).apply(math.ceil).clip(lower=1)

    out["current_buses"] = current_buses
    out["change"] = out["recommended_buses"] - out["current_buses"]
    out["action"] = out["change"].map(
        lambda x: "Add buses" if x > 0 else ("Reduce buses" if x < 0 else "Maintain")
    )
    return out
