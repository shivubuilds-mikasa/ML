# Dataset provenance and limits

`vehicle_service_data.csv` is generated locally by `src/dataset.py` with 8,000 reproducible synthetic passenger-car rows (seed 42). It is not an open dataset, fleet sample, or observed maintenance history. No real vehicle data or personal data is included.

## Why synthetic data is used

Published work describes an AMAG Swiss car maintenance-log dataset (roughly 300,000 cars, with odometer observations at maintenance events), but the paper says records were provided by AMAG and does not publish a downloadable CSV or reuse license. It is not treated as an open training set here. The UCI Car Evaluation dataset evaluates acceptability from six categorical attributes, not service history or continuous remaining service distance. A Kaggle Vehicle Maintenance Data set is cited by a later paper as a 50,000-row binary `Need_Maintenance` dataset; that classification target does not support the continuous target required here, and its dataset-page license and original files were not independently verified in this run.

## Synthetic target assumptions

Each row represents a simulated car. A recommended distance interval is sampled from 5,000, 7,500, 10,000, 12,000, or 15,000 km; time interval from 6, 12, 18, or 24 months. Typical city driving, vehicle age above five years, previous service category, and bounded Gaussian variation apply documented simulated multipliers. The synthetic target is the smaller of adjusted remaining distance and the remaining calendar interval converted to kilometres using average daily use, plus the bounded random variation; it is clamped to zero and never exceeds the configured manual distance interval. The multipliers and noise are educational assumptions, not manufacturer facts or empirical effects.

The app asks each user to supply the service interval found in their own car's owner manual. The trained model's suggestion is capped at the earlier manual distance/time limit. The model can advance a reminder but cannot postpone the manual schedule.

Regenerate the same CSV with:

```bash
python -c "from src.dataset import write_dataset; from config import DATA_PATH; write_dataset(DATA_PATH, 8000, 42)"
```

## Schema

| Column | Meaning | Role |
|---|---|---|
| current_odometer_km | Current simulated odometer | Input |
| last_service_odometer_km | Odometer at prior service | Input |
| km_since_last_service | Current minus previous service odometer | Engineered input |
| vehicle_age_years | Age of car | Input |
| average_daily_km | Average daily driving distance | Input |
| months_since_last_service | Elapsed months | Input |
| recommended_interval_km | User/manual-configured interval | Input |
| recommended_interval_months | User/manual-configured time interval | Input |
| driving_condition | City, Mixed, Highway | Categorical input |
| previous_service_type | Oil and filter, Inspection, Major service | Categorical input |
| remaining_km_to_service | Simulated remaining distance | Target only |
