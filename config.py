from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
EDA_DIR = ROOT / "reports" / "eda"
APP_REPORT_DIR = ROOT / "reports" / "generated"
DATA_PATH = DATASET_DIR / "vehicle_service_data.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"
RANDOM_STATE = 42
N_SYNTHETIC_ROWS = 8000

NUMERIC_FEATURES = [
    "current_odometer_km", "last_service_odometer_km", "km_since_last_service",
    "vehicle_age_years", "average_daily_km", "months_since_last_service",
    "recommended_interval_km", "recommended_interval_months",
]
CATEGORICAL_FEATURES = ["driving_condition", "previous_service_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "remaining_km_to_service"
