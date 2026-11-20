"""Input preparation and conservative service recommendation rules."""
from __future__ import annotations
from datetime import date, timedelta
import pandas as pd
from config import FEATURES, MODEL_PATH


def validate_vehicle_input(data: dict, today: date | None = None) -> list[str]:
    today = today or date.today()
    errors = []
    for key in ("current_odometer_km", "last_service_odometer_km", "average_daily_km",
                "recommended_interval_km", "recommended_interval_months"):
        try:
            if float(data[key]) < 0:
                errors.append(f"{key.replace('_', ' ').title()} cannot be negative.")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key.replace('_', ' ').title()} must be a number.")
    try:
        if float(data["current_odometer_km"]) < float(data["last_service_odometer_km"]):
            errors.append("Current odometer must be at least the last-service odometer.")
    except (KeyError, TypeError, ValueError):
        pass
    if float(data.get("average_daily_km", 0) or 0) <= 0:
        errors.append("Average daily distance must be greater than zero to estimate time.")
    if not 1000 <= float(data.get("recommended_interval_km", 0) or 0) <= 50000:
        errors.append("Enter the manual's distance interval between 1,000 and 50,000 km.")
    if not 1 <= float(data.get("recommended_interval_months", 0) or 0) <= 60:
        errors.append("Enter the manual's time interval between 1 and 60 months.")
    try:
        service_date = data["last_service_date"]
        if isinstance(service_date, str):
            service_date = date.fromisoformat(service_date)
        if service_date > today:
            errors.append("Last service date cannot be in the future.")
        if service_date < today - timedelta(days=60 * 31):
            errors.append("Last service date must be within the past 60 months.")
    except (KeyError, ValueError, TypeError):
        errors.append("Enter a valid last service date.")
    year = data.get("manufacturing_year")
    if year is not None and not 1980 <= int(year) <= today.year:
        errors.append(f"Manufacturing year must be between 1980 and {today.year}.")
    if not str(data.get("brand", "")).strip() or not str(data.get("model", "")).strip():
        errors.append("Enter the car brand and model.")
    if data.get("driving_condition") not in ["City", "Mixed", "Highway"]:
        errors.append("Driving condition must be City, Mixed, or Highway.")
    if data.get("previous_service_type") not in ["Oil and filter", "Inspection", "Major service"]:
        errors.append("Choose a valid previous service category.")
    return errors


def build_features(data: dict, today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    service_date = data["last_service_date"]
    if isinstance(service_date, str):
        service_date = date.fromisoformat(service_date)
    current = float(data["current_odometer_km"])
    last = float(data["last_service_odometer_km"])
    features = {
        "current_odometer_km": current,
        "last_service_odometer_km": last,
        "km_since_last_service": max(0.0, current - last),
        "vehicle_age_years": max(0, today.year - int(data["manufacturing_year"])),
        "average_daily_km": float(data["average_daily_km"]),
        "months_since_last_service": max(0.0, (today - service_date).days / 30.4375),
        "recommended_interval_km": float(data["recommended_interval_km"]),
        "recommended_interval_months": float(data["recommended_interval_months"]),
        "driving_condition": data["driving_condition"],
        "previous_service_type": data["previous_service_type"],
    }
    return pd.DataFrame([{name: features[name] for name in FEATURES}])


def service_status(remaining_km: float, remaining_days: int, interval_km: float) -> str:
    if remaining_km <= 0 or remaining_days <= 0:
        return "Service Due"
    if remaining_km <= .05 * interval_km or remaining_days <= 30:
        return "Service Due"
    if remaining_km <= .15 * interval_km or remaining_days <= 90:
        return "Service Approaching"
    return "Service Not Due"


def predict_service(data: dict, model=None, today: date | None = None) -> dict:
    today = today or date.today()
    errors = validate_vehicle_input(data, today)
    if errors:
        raise ValueError(" ".join(errors))
    if model is None:
        import joblib
        model = joblib.load(MODEL_PATH)
    features = build_features(data, today)
    model_km = max(0.0, float(model.predict(features)[0]))
    km_since = float(features.iloc[0]["km_since_last_service"])
    months_since = float(features.iloc[0]["months_since_last_service"])
    manual_km_budget = max(0.0, float(data["recommended_interval_km"]) - km_since)
    manual_day_budget = max(0.0, (float(data["recommended_interval_months"]) - months_since) * 30.4375)
    # A model suggestion never delays the owner's manual's service limit.
    remaining_km = min(model_km, manual_km_budget, manual_day_budget * float(data["average_daily_km"]))
    remaining_days = max(0, min(
        int(remaining_km / float(data["average_daily_km"])),
        int(manual_day_budget),
    ))
    due_date = today + timedelta(days=remaining_days)
    return {
        "model_estimated_remaining_km": round(model_km),
        "remaining_km": round(remaining_km),
        "remaining_days": remaining_days,
        "service_date": due_date,
        "status": service_status(remaining_km, remaining_days, float(data["recommended_interval_km"])),
        "manual_distance_limit_km": round(manual_km_budget),
        "manual_time_limit_days": round(manual_day_budget),
        "explanation": "Estimate is capped at the earlier distance/time limit entered from the car owner's manual. Synthetic model estimates are demonstrations, not a mechanical diagnosis.",
    }
