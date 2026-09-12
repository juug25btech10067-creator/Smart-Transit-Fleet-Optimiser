"""Generate a synthetic route-level transit demand dataset."""

from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data" / "transit_demand.csv"

ROUTES = {
    "R01": 1.00,
    "R02": 1.18,
    "R03": 0.82,
    "R04": 1.35,
    "R05": 0.95,
    "R06": 1.25,
    "R07": 0.72,
    "R08": 1.10,
}

def generate(days: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []

    for date in dates:
        weekday = date.weekday()
        weekend = int(weekday >= 5)

        for route, route_factor in ROUTES.items():
            for hour in range(6, 23):
                morning_peak = np.exp(-((hour - 8) / 1.8) ** 2)
                evening_peak = np.exp(-((hour - 18) / 2.0) ** 2)
                base = 45 + 210 * morning_peak + 245 * evening_peak

                weekend_factor = 0.68 if weekend else 1.0
                weather = RNG.choice(
                    ["Clear", "Cloudy", "Rain"],
                    p=[0.60, 0.28, 0.12]
                )
                weather_factor = {"Clear": 1.0, "Cloudy": 0.94, "Rain": 1.10}[weather]

                trend = 1 + 0.0008 * (date - dates[0]).days
                noise = RNG.normal(0, 14)

                passengers = max(
                    10,
                    round(base * route_factor * weekend_factor *
                          weather_factor * trend + noise)
                )

                rows.append({
                    "date": date.date().isoformat(),
                    "route": route,
                    "hour": hour,
                    "weekday": weekday,
                    "is_weekend": weekend,
                    "weather": weather,
                    "passengers": passengers,
                })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT}")
