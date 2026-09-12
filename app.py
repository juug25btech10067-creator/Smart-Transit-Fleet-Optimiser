import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.optimizer import recommend_buses

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "transit_demand.csv"
MODEL = ROOT / "models" / "demand_model.joblib"
METRICS = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="SmartTransit Fleet Optimizer",
    page_icon="🚌",
    layout="wide",
)

st.title("🚌 SmartTransit Fleet Optimizer")
st.caption("Demand forecasting + route-level bus allocation planning")

if not DATA.exists():
    st.error("Dataset not found. Run: python src/generate_data.py")
    st.stop()

if not MODEL.exists():
    st.error("Model not found. Run: python src/train_model.py")
    st.stop()

df = pd.read_csv(DATA)
model = joblib.load(MODEL)
metrics = json.loads(METRICS.read_text()) if METRICS.exists() else {}

st.sidebar.header("Planning controls")
selected_route = st.sidebar.selectbox("Route", sorted(df["route"].unique()))
hour = st.sidebar.slider("Hour", 6, 22, 18)
weekday = st.sidebar.slider("Weekday (0=Mon)", 0, 6, 2)
weather = st.sidebar.selectbox("Weather", ["Clear", "Cloudy", "Rain"])
bus_capacity = st.sidebar.slider("Bus capacity", 30, 80, 50)
load_factor = st.sidebar.slider("Target load factor", 0.50, 1.00, 0.80, 0.05)
current_buses = st.sidebar.slider("Current buses on route", 1, 12, 4)

input_row = pd.DataFrame([{
    "route": selected_route,
    "hour": hour,
    "weekday": weekday,
    "is_weekend": int(weekday >= 5),
    "weather": weather,
}])

prediction = float(model.predict(input_row)[0])
planning = recommend_buses(
    pd.DataFrame({"route": [selected_route], "predicted_passengers": [prediction]}),
    bus_capacity=bus_capacity,
    target_load_factor=load_factor,
    current_buses=current_buses,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted demand", f"{prediction:,.0f}")
c2.metric("Recommended buses", int(planning.iloc[0]["recommended_buses"]))
c3.metric("Current buses", current_buses)
c4.metric("Suggested change", int(planning.iloc[0]["change"]))

st.subheader("Route demand profile")
route_df = df[df["route"] == selected_route].copy()
hourly = route_df.groupby("hour", as_index=False)["passengers"].mean()

fig, ax = plt.subplots()
ax.plot(hourly["hour"], hourly["passengers"], marker="o")
ax.axvline(hour, linestyle="--", label="Selected hour")
ax.set_xlabel("Hour")
ax.set_ylabel("Average passengers")
ax.set_title(f"Average hourly demand: {selected_route}")
ax.legend()
st.pyplot(fig)

st.subheader("Fleet recommendation across routes")

latest = (
    df[df["hour"] == hour]
    .groupby(["route", "hour", "weekday", "is_weekend", "weather"], as_index=False)
    .tail(1)
)
route_inputs = pd.DataFrame({
    "route": sorted(df["route"].unique()),
    "hour": [hour] * df["route"].nunique(),
    "weekday": [weekday] * df["route"].nunique(),
    "is_weekend": [int(weekday >= 5)] * df["route"].nunique(),
    "weather": [weather] * df["route"].nunique(),
})
route_inputs["predicted_passengers"] = model.predict(route_inputs)

# A simple baseline for comparison: four buses per route.
summary = recommend_buses(
    route_inputs[["route", "predicted_passengers"]],
    bus_capacity=bus_capacity,
    target_load_factor=load_factor,
    current_buses=current_buses,
)

st.dataframe(
    summary[["route", "predicted_passengers", "current_buses",
             "recommended_buses", "change", "action"]]
    .rename(columns={
        "predicted_passengers": "Predicted passengers",
        "current_buses": "Current buses",
        "recommended_buses": "Recommended buses",
        "change": "Change",
        "action": "Action",
    }),
    use_container_width=True,
    hide_index=True,
)

if metrics:
    with st.expander("Model evaluation"):
        st.write({
            "MAE": metrics.get("mae"),
            "RMSE": metrics.get("rmse"),
            "R²": metrics.get("r2"),
        })
        st.caption(
            "Metrics are measured on the included synthetic dataset and should "
            "not be interpreted as real-world transit performance."
        )

st.divider()
st.caption("Portfolio project by Soumya Jyoti Bhattacharjee")
