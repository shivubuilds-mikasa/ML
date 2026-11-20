"""Generate/load data, compare models, tune and persist the selected pipeline."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from config import DATA_PATH, EDA_DIR, FEATURES, METRICS_PATH, MODEL_PATH, MODELS_DIR, N_SYNTHETIC_ROWS, RANDOM_STATE, TARGET
from src.dataset import write_dataset
from src.eda import create_eda_plots
from src.evaluation import regression_metrics
from src.preprocessing import build_preprocessor


def load_and_inspect_dataset(path=DATA_PATH) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        df = write_dataset(path, N_SYNTHETIC_ROWS, RANDOM_STATE)
    else:
        df = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    report = {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "source_type": "reproducible synthetic simulation" if path == DATA_PATH else "user-supplied CSV (verify provenance)",
        "target": TARGET,
        "feature_columns": FEATURES,
        "numeric_iqr_outliers_by_column": {},
    }
    for column in df.select_dtypes(include="number").columns:
        q1, q3 = df[column].quantile([.25, .75])
        spread = q3 - q1
        report["numeric_iqr_outliers_by_column"][column] = int(((df[column] < q1 - 1.5 * spread) | (df[column] > q3 + 1.5 * spread)).sum())
    if df[TARGET].isna().any() or df.duplicated().any():
        df = df.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("At least 100 complete rows are required for training.")
    return df, report


def make_pipeline(estimator):
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def train_and_evaluate(dataset_path=DATA_PATH) -> dict:
    df, audit = load_and_inspect_dataset(dataset_path)
    x_dev, x_test, y_dev, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=.20, random_state=RANDOM_STATE,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_dev, y_dev, test_size=.20, random_state=RANDOM_STATE + 1,
    )
    models = {
        "Median baseline": make_pipeline(DummyRegressor(strategy="median")),
        "Decision Tree": make_pipeline(DecisionTreeRegressor(max_depth=14, min_samples_leaf=8, random_state=RANDOM_STATE)),
        "Extra Trees": make_pipeline(ExtraTreesRegressor(n_estimators=180, max_depth=18, min_samples_leaf=4, max_features=.9, random_state=RANDOM_STATE, n_jobs=-1)),
        "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=180, max_depth=18, min_samples_leaf=4, max_features=.9, random_state=RANDOM_STATE, n_jobs=-1)),
    }
    validation_metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        validation_metrics[name] = regression_metrics(y_validation, model.predict(x_validation))

    tuned = make_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    search = RandomizedSearchCV(
        tuned,
        param_distributions={
            "model__n_estimators": [120, 180, 240],
            "model__max_depth": [12, 18, 24, None],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features": [.7, .9, 1.0],
        },
        n_iter=8, cv=3, scoring="neg_mean_absolute_error", random_state=RANDOM_STATE,
        n_jobs=-1, refit=True,
    )
    search.fit(x_train, y_train)
    tuned_model = search.best_estimator_
    validation_metrics["Tuned Random Forest"] = regression_metrics(y_validation, tuned_model.predict(x_validation))

    # Choose on validation only. Refit every candidate on train+validation and
    # evaluate once on an untouched test partition for reportable comparison.
    selected_name = min(validation_metrics, key=lambda name: (validation_metrics[name]["MAE"], -validation_metrics[name]["R2"]))
    development_x = pd.concat([x_train, x_validation])
    development_y = pd.concat([y_train, y_validation])
    all_models = {**models, "Tuned Random Forest": tuned_model}
    outputs, predictions = {}, {}
    for name, model in all_models.items():
        model.fit(development_x, development_y)
        pred = model.predict(x_test)
        outputs[name] = regression_metrics(y_test, pred)
        predictions[name] = pred
    selected_pipeline = all_models[selected_name]
    selected_prediction = predictions[selected_name]
    selected_estimator = selected_pipeline.named_steps["model"]
    selected_features = selected_pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = getattr(selected_estimator, "feature_importances_", None)
    top_features = [] if importance is None else [
        {"feature": str(selected_features[i]), "importance": float(importance[i])}
        for i in importance.argsort()[::-1][:10]
    ]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_PATH)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": audit,
        "random_state": RANDOM_STATE,
        "split": {"train": .64, "validation": .16, "test": .20},
        "train_rows": int(len(y_train)), "validation_rows": int(len(y_validation)),
        "test_rows": int(len(y_test)),
        "selection_metric": "Lowest validation MAE; R2 used only as tie-breaker. Test remains untouched until final comparison.",
        "selected_model": selected_name,
        "selected_model_top_feature_importances": top_features,
        "validation_metrics": validation_metrics,
        "tuned_random_forest_best_params": search.best_params_,
        "tuned_random_forest_cv_mae_km": float(-search.best_score_),
        "model_metrics": outputs,
        "prediction_target_scope": "Synthetic simulated remaining kilometres; not real-world or mechanical-diagnosis accuracy.",
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"actual_remaining_km": y_test.to_numpy(), "predicted_remaining_km": selected_prediction,
                  "residual_km": y_test.to_numpy() - selected_prediction}).to_csv(MODELS_DIR / "test_residuals.csv", index=False)
    create_eda_plots(df, EDA_DIR, outputs)
    print(json.dumps(result, indent=2))
    print(f"Selected on validation; held-out test metrics saved. Model: {MODEL_PATH}")
    return result


if __name__ == "__main__":
    train_and_evaluate()
