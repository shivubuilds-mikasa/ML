"""Reproducible synthetic passenger-car service dataset generator.

These rows are illustrative simulations, not observed service records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
DRIVING_CONDITIONS = ["City", "Mixed", "Highway"]
SERVICE_TYPES = ["Oil and filter", "Inspection", "Major service"]
INTERVAL_KM = [5000, 7500, 10000, 12000, 15000]
INTERVAL_MONTHS = [6, 12, 18, 24]


def generate_dataset(n_rows: int = 8000, seed: int = SEED) -> pd.DataFrame:
    """Generate passenger-car examples from disclosed maintenance assumptions.

    Synthetic target is the earlier of the distance-based or calendar-based
    interval. City use, older age, and service category add a small synthetic
    conservative adjustment. The target is clipped to zero. No target or
    target-derived feature is included in the feature columns.
    """
    if n_rows < 100:
        raise ValueError("Generate at least 100 rows for a meaningful prototype dataset.")
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 21, n_rows)
    current_odo = np.rint(rng.uniform(6000, 320000, n_rows) / 100) * 100
    interval_km = rng.choice(INTERVAL_KM, n_rows, p=[.08, .12, .49, .12, .19])
    interval_months = rng.choice(INTERVAL_MONTHS, n_rows, p=[.13, .63, .18, .06])
    driving = rng.choice(DRIVING_CONDITIONS, n_rows, p=[.42, .43, .15])
    service = rng.choice(SERVICE_TYPES, n_rows, p=[.55, .25, .20])
    average_daily = np.clip(rng.lognormal(mean=3.4, sigma=.55, size=n_rows), 4, 180)
    months_since = np.clip(rng.gamma(shape=2.1, scale=4.4, size=n_rows), .1, 35)
    km_since = np.minimum(
        np.clip(rng.gamma(shape=2.0, scale=interval_km / 6.0, size=n_rows), 0, interval_km * 1.3),
        np.maximum(0, current_odo - 500),
    )
    last_service_odo = current_odo - km_since

    condition_factor = np.select(
        [driving == "City", driving == "Mixed", driving == "Highway"],
        [.91, .97, 1.00], default=.95,
    )
    age_factor = 1.0 - np.clip(age - 5, 0, 15) * .004
    service_factor = np.select(
        [service == "Oil and filter", service == "Inspection", service == "Major service"],
        [.98, .96, 1.00], default=.98,
    )
    # Synthetic behavior variability is explicit and bounded; it is not a
    # measured real-world effect and must not be read as OEM advice.
    synthetic_effective_km = interval_km * condition_factor * age_factor * service_factor
    distance_budget = synthetic_effective_km - km_since
    calendar_budget = (interval_months - months_since / 1.0) * 30.4375 * average_daily
    noise = rng.normal(0, interval_km * .025, n_rows)
    remaining_km = np.maximum(0, np.minimum(distance_budget, calendar_budget) + noise)
    remaining_km = np.minimum(remaining_km, np.maximum(0, interval_km - km_since))
    # Floor before decimal storage so rounding can never nudge a target above
    # its manual distance cap by a fraction of a kilometre.
    remaining_km = np.floor(remaining_km * 100) / 100

    return pd.DataFrame({
        "current_odometer_km": current_odo.astype(float),
        "last_service_odometer_km": last_service_odo.astype(float),
        "km_since_last_service": km_since.astype(float),
        "vehicle_age_years": age.astype(int),
        "average_daily_km": np.round(average_daily, 2),
        "months_since_last_service": np.round(months_since, 2),
        "recommended_interval_km": interval_km.astype(int),
        "recommended_interval_months": interval_months.astype(int),
        "driving_condition": driving,
        "previous_service_type": service,
        "remaining_km_to_service": np.round(remaining_km, 2),
    })


def write_dataset(path, n_rows: int = 8000, seed: int = SEED) -> pd.DataFrame:
    data = generate_dataset(n_rows, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data
