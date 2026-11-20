from datetime import date, timedelta
import numpy as np
import pytest
from src.dataset import generate_dataset
from src.evaluation import regression_metrics
from src.predict import build_features, predict_service, validate_vehicle_input
from config import FEATURES, TARGET

class FixedEstimator:
    def __init__(self, estimate): self.estimate=estimate
    def predict(self, features): return np.array([self.estimate])

def sample(today=date(2026,11,20)):
    return {"brand":"Honda","model":"Civic","manufacturing_year":2020,
            "current_odometer_km":50000,"last_service_odometer_km":45000,
            "last_service_date":today-timedelta(days=90),"average_daily_km":40,
            "driving_condition":"Mixed","recommended_interval_km":10000,
            "recommended_interval_months":12,"previous_service_type":"Oil and filter"}

def test_generator_is_reproducible_and_car_only():
    a=generate_dataset(300,seed=10); b=generate_dataset(300,seed=10)
    assert a.equals(b)
    assert len(a)==300 and TARGET in a.columns
    assert set(a["driving_condition"]) <= {"City","Mixed","Highway"}
    assert (a[TARGET]>=0).all()
    manual_cap=(a["recommended_interval_km"]-a["km_since_last_service"]).clip(lower=0)
    assert (a[TARGET] <= manual_cap+1e-8).all()

def test_generated_schema_matches_training_features():
    frame=generate_dataset(250)
    assert list(frame[FEATURES].columns)==FEATURES
    assert frame.isna().sum().sum()==0
    assert frame.duplicated().sum()==0

def test_validation_rejects_invalid_odo_and_future_service_date():
    data=sample(); data["current_odometer_km"]=40000; data["last_service_date"]=date(2027,1,1)
    errors=validate_vehicle_input(data,today=date(2026,11,20))
    assert any("odometer" in e.lower() for e in errors)
    assert any("future" in e.lower() for e in errors)

def test_features_are_engineered_and_prediction_is_manual_interval_capped():
    data=sample(); f=build_features(data,today=date(2026,11,20))
    assert f.iloc[0]["km_since_last_service"]==5000
    assert f.iloc[0]["months_since_last_service"]==pytest.approx(90/30.4375)
    result=predict_service(data,FixedEstimator(9000),today=date(2026,11,20))
    assert result["remaining_km"] <= 5000
    assert result["status"] in {"Service Not Due","Service Approaching","Service Due"}
    assert result["service_date"] >= date(2026,11,20)

def test_regression_metrics_are_correct():
    m=regression_metrics([0,10,20],[0,12,18])
    assert m["MAE"]==pytest.approx(4/3)
    assert m["MSE"]==pytest.approx(8/3)
    assert m["RMSE"]==pytest.approx(np.sqrt(8/3))
