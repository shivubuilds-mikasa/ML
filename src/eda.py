"""Create labeled exploratory plots from the real project CSV/results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def create_eda_plots(df: pd.DataFrame, output_dir: Path, metrics: dict | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(name: str):
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)

    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["current_odometer_km"] / 1000, bins=35, color="#3978b8")
    plt.title("Current Odometer Distribution for Synthetic Cars")
    plt.xlabel("Current odometer (thousand km)"); plt.ylabel("Number of simulated cars")
    save("01_odometer_distribution.png")

    plt.figure(figsize=(8, 4.5))
    sns.countplot(data=df, x="recommended_interval_km", color="#5a9d82")
    plt.title("Configured Distance Service Intervals")
    plt.xlabel("Recommended interval from simulated owner's manual (km)"); plt.ylabel("Number of rows")
    save("02_service_intervals.png")

    plt.figure(figsize=(8, 4.5))
    age_summary = df.assign(service_events_per_year=12 / df["recommended_interval_months"]).groupby("vehicle_age_years", as_index=False)["service_events_per_year"].mean()
    sns.lineplot(data=age_summary, x="vehicle_age_years", y="service_events_per_year", marker="o", color="#835fc4")
    plt.title("Vehicle Age and Simulated Scheduled Service Frequency")
    plt.xlabel("Vehicle age (years)"); plt.ylabel("Configured service events per year")
    save("03_age_service_frequency.png")

    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["km_since_last_service"], bins=35, color="#e39b51")
    plt.title("Distance Travelled Since the Previous Service")
    plt.xlabel("Distance since previous service (km)"); plt.ylabel("Number of simulated cars")
    save("04_distance_since_service.png")

    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x="driving_condition", y="remaining_km_to_service", order=["City", "Mixed", "Highway"], hue="driving_condition", legend=False, palette="Set2")
    plt.title("Simulated Remaining Distance by Driving Condition")
    plt.xlabel("Driving condition"); plt.ylabel("Simulated remaining distance (km)")
    save("05_condition_maintenance.png")

    plt.figure(figsize=(9, 7))
    numeric = df.select_dtypes(include="number")
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, annot=False, linewidths=.25)
    plt.title("Correlation Heatmap of Numeric Synthetic Features")
    save("06_correlation_heatmap.png")

    if metrics:
        frame = pd.DataFrame(metrics).T.reset_index(names="model")
        plt.figure(figsize=(9, 4.8))
        sns.barplot(data=frame, x="MAE", y="model", color="#6688c9")
        plt.title("Held-Out Test MAE by Model (Lower Is Better)")
        plt.xlabel("Mean Absolute Error (km)"); plt.ylabel("Model")
        save("07_model_comparison.png")
    return saved
