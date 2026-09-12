# SmartTransit: Demand Forecasting & Fleet Optimizer

A portfolio-ready Python project that demonstrates how public-transport operators can use historical demand to forecast passenger volume and recommend route-level bus allocations.

## What it does

- Generates a realistic sample transit-demand dataset for development/demo use.
- Creates time, route, weekday/weekend and weather features.
- Trains a machine-learning model to forecast passenger demand.
- Converts predicted demand into a recommended number of buses.
- Compares current allocation with the recommended allocation.
- Provides an interactive Streamlit dashboard.
- Includes a command-line workflow for reproducible model training.

> **Note:** The included dataset is synthetic and is not official BMTC data. The project is designed to demonstrate the methodology without claiming access to proprietary transit records.

## Tech stack

**Python · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit · Joblib**

## Project structure

```text
smart-transit-fleet-optimizer/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── transit_demand.csv
└── src/
    ├── generate_data.py
    ├── train_model.py
    └── optimizer.py
```

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/smart-transit-fleet-optimizer.git
cd smart-transit-fleet-optimizer

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
streamlit run app.py
```

The dashboard opens in your browser.

## Methodology

1. **Demand data:** route-level passenger observations are generated with realistic patterns for peak hours, weekends and weather.
2. **Prediction:** a Random Forest regression model estimates expected passenger demand.
3. **Capacity conversion:** predicted demand is translated into buses using configurable bus capacity and a target load factor.
4. **Operational comparison:** the dashboard compares the recommended fleet with the current allocation and highlights routes that may be over- or under-served.
5. **Visualization:** route demand, prediction quality and allocation changes are shown interactively.

### Bus allocation formula

```text
required_buses =
ceil(predicted_passengers /
     (bus_capacity × target_load_factor))
```

This is intentionally a simplified planning model. Real transit scheduling would also consider headways, turnaround time, depot constraints, crew availability, traffic, vehicle types and minimum service requirements.

## Resume-ready project description

**SmartTransit: Demand Forecasting & Fleet Optimization | Python, Scikit-learn, Streamlit**
- Built a machine-learning dashboard that forecasts route-level passenger demand and converts predictions into data-driven bus allocation recommendations.
- Engineered temporal, route and weather features and trained a Random Forest regression model to estimate passenger demand.
- Developed an optimization layer to identify routes needing additional or fewer buses while maintaining a configurable service-capacity target.
- Created an interactive Streamlit dashboard for route comparison, demand trends and model evaluation.

## Good GitHub presentation

After pushing the project, add:
- 1 screenshot of the dashboard to the README.
- A short demo GIF if you have one.
- Your LinkedIn profile in the README footer.
- A GitHub repository description such as: `ML-powered transit demand forecasting and route-level fleet planning dashboard.`

Do not claim accuracy percentages or real-world BMTC impact unless you actually measure them on a real dataset.
