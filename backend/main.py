from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
META_PATH = ARTIFACT_DIR / "meta.json"

FEATURE_COLUMNS = [
    "sleep_hours",
    "caffeine_level",
    "stress",
    "workout",
    "day_index",
    "weekly_sin",
    "weekly_cos",
    "cycle_day",
    "cycle_day_normalized",
    "late_luteal",
    "menstrual_window",
    "basal_temp",
    "resting_hr",
    "basal_temp_missing",
    "resting_hr_missing",
    "sleep_prev",
    "stress_prev",
    "score_prev",
    "stress_x_late_luteal",
    "sleep_x_late_luteal",
    "stress_x_menstrual",
    "sleep_x_menstrual",
    "caffeine_x_sleep",
]

FEATURE_LABELS = {
    "sleep_hours": "Sleep hours",
    "caffeine_level": "Caffeine level",
    "stress": "Stress",
    "workout": "Workout",
    "day_index": "Day index",
    "weekly_sin": "Weekly pattern (sin)",
    "weekly_cos": "Weekly pattern (cos)",
    "cycle_day": "Cycle day",
    "cycle_day_normalized": "Cycle day (normalized)",
    "late_luteal": "Late luteal phase",
    "menstrual_window": "Menstrual window",
    "basal_temp": "Basal temperature",
    "resting_hr": "Resting heart rate",
    "basal_temp_missing": "Missing basal temperature",
    "resting_hr_missing": "Missing resting heart rate",
    "sleep_prev": "Previous sleep",
    "stress_prev": "Previous stress",
    "score_prev": "Previous bad-day score",
    "stress_x_late_luteal": "Stress x late luteal",
    "sleep_x_late_luteal": "Sleep x late luteal",
    "stress_x_menstrual": "Stress x menstrual window",
    "sleep_x_menstrual": "Sleep x menstrual window",
    "caffeine_x_sleep": "Caffeine x sleep",
}

PRIMARY_DRIVER_FEATURES = [
    "sleep_hours",
    "stress",
    "workout",
    "caffeine_level",
    "cycle_day",
    "late_luteal",
    "menstrual_window",
    "resting_hr",
    "basal_temp",
]


class LogEntry(BaseModel):
    date: date
    mood: float = Field(ge=0, le=5)
    cramps: float = Field(ge=0, le=5)
    fatigue: float = Field(ge=0, le=5)
    headache: float = Field(ge=0, le=5)
    acne: float = Field(ge=0, le=5)
    sleep_hours: float = Field(ge=0, le=24)
    caffeine_level: int = Field(ge=0, le=2)
    stress: float = Field(ge=0, le=5)
    workout: bool
    basal_temp: Optional[float] = None
    resting_hr: Optional[float] = None


class TrainRequest(BaseModel):
    period_starts: List[date] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    use_reference_boost: bool = True


class PredictRequest(BaseModel):
    period_starts: List[date] = Field(default_factory=list)
    history_logs: List[LogEntry] = Field(default_factory=list)
    target_log: LogEntry


class ForecastRequest(BaseModel):
    period_starts: List[date] = Field(default_factory=list)
    last_period_start: Optional[date] = None
    estimated_cycle_length: Optional[float] = None
    typical_features: Dict[str, float] = Field(default_factory=dict)
    start_date: Optional[date] = None


app = FastAPI(title="PMS Mood Compass API", version="1.1.0")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_state: Dict[str, Any] = {
    "model": None,
    "meta": {
        "feature_columns": FEATURE_COLUMNS,
        "target_semantics_version": "mood_inverted_v2",
        "residual_std": 1.0,
        "n_samples": 0,
        "model_name": "untrained",
        "train_feature_mean": [0.0] * len(FEATURE_COLUMNS),
        "train_feature_std": [1.0] * len(FEATURE_COLUMNS),
        "feature_medians": {column: 0.0 for column in FEATURE_COLUMNS},
        "cycle_profile": {},
        "train_target_mean": 2.5,
    },
}


def compute_bad_day_score(entry: LogEntry) -> float:
    mood_burden = 5.0 - float(entry.mood)
    return (
        0.35 * mood_burden
        + 0.25 * entry.fatigue
        + 0.25 * entry.cramps
        + 0.10 * entry.headache
        + 0.05 * entry.acne
    )


def fit_automl_if_available(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, str, str] | None:
    try:
        from flaml import AutoML
    except Exception:
        return None

    estimator_list = ["xgboost", "rf", "extra_tree"]
    try:
        import lightgbm  # noqa: F401

        estimator_list.insert(1, "lgbm")
    except Exception:
        pass

    try:
        automl = AutoML()
        automl.fit(
            X_train=x_train,
            y_train=y_train,
            task="regression",
            metric="mae",
            eval_method="cv",
            split_type="time",
            n_splits=3,
            estimator_list=estimator_list,
            time_budget=20,
            verbose=0,
        )
        best_name = str(getattr(automl, "best_estimator", "automl"))
        return (
            automl,
            f"flaml:{best_name}",
            f"FLAML AutoML selected `{best_name}` after comparing candidate regressors with time-ordered validation.",
        )
    except Exception:
        return None


def choose_model() -> tuple[Any, str]:
    try:
        import lightgbm as lgb

        return (
            lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=2,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            ),
            "lightgbm",
        )
    except Exception:
        pass

    try:
        import xgboost as xgb

        return (
            xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=1,
                random_state=42,
                objective="reg:squarederror",
            ),
            "xgboost",
        )
    except Exception:
        pass

    return (
        HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.06,
            max_iter=350,
            min_samples_leaf=2,
            random_state=42,
        ),
        "sklearn_hist_gradient_boosting",
    )


def build_model_candidates(base_model: Any, model_name: str) -> List[tuple[Any, str]]:
    if model_name == "lightgbm":
        return [
            (base_model, "lightgbm_fast"),
            (
                base_model.__class__(
                    n_estimators=500,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=2,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    random_state=42,
                ),
                "lightgbm_balanced",
            ),
            (
                base_model.__class__(
                    n_estimators=280,
                    learning_rate=0.07,
                    num_leaves=23,
                    min_child_samples=1,
                    subsample=0.95,
                    colsample_bytree=0.95,
                    random_state=42,
                ),
                "lightgbm_compact",
            ),
        ]
    if model_name == "xgboost":
        return [
            (base_model, "xgboost_fast"),
            (
                base_model.__class__(
                    n_estimators=600,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=1,
                    random_state=42,
                    objective="reg:squarederror",
                ),
                "xgboost_balanced",
            ),
            (
                base_model.__class__(
                    n_estimators=350,
                    learning_rate=0.06,
                    max_depth=3,
                    subsample=0.95,
                    colsample_bytree=0.95,
                    min_child_weight=1,
                    random_state=42,
                    objective="reg:squarederror",
                ),
                "xgboost_compact",
            ),
        ]

    return [
        (base_model, "histgb_default"),
        (
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.045,
                max_iter=520,
                min_samples_leaf=2,
                random_state=42,
            ),
            "histgb_deeper",
        ),
        (
            HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.08,
                max_iter=300,
                min_samples_leaf=3,
                random_state=42,
            ),
            "histgb_compact",
        ),
    ]


def time_series_cv_mae(candidate_model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> float:
    n = len(x_train)
    if n < 12:
        return float("inf")

    split_points = [
        int(n * 0.55),
        int(n * 0.7),
        int(n * 0.82),
    ]
    split_points = sorted(set(max(6, min(point, n - 2)) for point in split_points))

    fold_scores = []
    for split in split_points:
        x_tr = x_train.iloc[:split]
        y_tr = y_train.iloc[:split]
        x_val = x_train.iloc[split:]
        y_val = y_train.iloc[split:]
        if len(x_val) < 2:
            continue
        model = clone(candidate_model)
        model.fit(x_tr, y_tr)
        y_hat = model.predict(x_val)
        fold_scores.append(float(mean_absolute_error(y_val, y_hat)))

    if not fold_scores:
        return float("inf")
    return float(np.mean(fold_scores))


def select_best_model(base_model: Any, model_name: str, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, str]:
    candidates = build_model_candidates(base_model, model_name)
    best_cv = float("inf")
    best_candidate = candidates[0]

    for candidate_model, candidate_name in candidates:
        cv_mae = time_series_cv_mae(candidate_model, x_train, y_train)
        if cv_mae < best_cv:
            best_cv = cv_mae
            best_candidate = (candidate_model, candidate_name)

    selected_model = clone(best_candidate[0])
    selected_model.fit(x_train, y_train)
    selected_name = f"{model_name}:{best_candidate[1]}"
    return selected_model, selected_name


def create_reference_augmented_logs(logs: List[LogEntry], use_reference_boost: bool) -> List[LogEntry]:
    if not use_reference_boost or len(logs) >= 21:
        return logs

    reference_logs = load_sample_logs()
    if not reference_logs:
        return logs

    # Shift sample data into a historical window so user-entered rows remain the most recent sequence.
    earliest_user_date = min(entry.date for entry in logs)
    latest_reference = max(entry.date for entry in reference_logs)
    shift_days = (earliest_user_date - latest_reference).days - 45

    shifted_reference = []
    for entry in reference_logs:
        shifted_reference.append(
            LogEntry(
                date=entry.date + timedelta(days=shift_days),
                mood=entry.mood,
                cramps=entry.cramps,
                fatigue=entry.fatigue,
                headache=entry.headache,
                acne=entry.acne,
                sleep_hours=entry.sleep_hours,
                caffeine_level=entry.caffeine_level,
                stress=entry.stress,
                workout=entry.workout,
                basal_temp=entry.basal_temp,
                resting_hr=entry.resting_hr,
            )
        )

    combined = shifted_reference + logs
    combined.sort(key=lambda item: item.date)
    return combined


def compute_cycle_stats(period_starts: List[date]) -> tuple[float, float]:
    starts = sorted(set(period_starts))
    if len(starts) < 2:
        return 28.0, 0.0

    lengths = []
    for idx in range(1, len(starts)):
        diff = (starts[idx] - starts[idx - 1]).days
        if diff > 0:
            lengths.append(diff)

    if not lengths:
        return 28.0, 0.0

    return float(np.median(lengths)), float(np.std(lengths))


def infer_period_starts(log_dates: List[date]) -> List[date]:
    if not log_dates:
        return []

    start = min(log_dates)
    end = max(log_dates)
    inferred: List[date] = []
    cursor = start
    while cursor <= end + timedelta(days=28):
        inferred.append(cursor)
        cursor = cursor + timedelta(days=28)
    return inferred


def cycle_day_for_date(entry_date: date, period_starts: List[date], estimated_cycle_length: float) -> int:
    cycle_len = max(21, int(round(estimated_cycle_length or 28)))
    if not period_starts:
        return ((entry_date.toordinal() % cycle_len) + 1)

    starts = sorted(set(period_starts))
    past_starts = [start for start in starts if start <= entry_date]
    if past_starts:
        last_start = past_starts[-1]
        raw_day = (entry_date - last_start).days + 1
    else:
        first_start = starts[0]
        raw_day = (entry_date - first_start).days + 1

    return ((raw_day - 1) % cycle_len) + 1


def build_feature_frame(logs: List[LogEntry], period_starts: List[date]) -> tuple[pd.DataFrame, Dict[str, float]]:
    if not logs:
        return pd.DataFrame(), {}

    logs_sorted = sorted(logs, key=lambda item: item.date)
    estimated_cycle_length, regularity = compute_cycle_stats(period_starts)
    if not period_starts:
        period_starts = infer_period_starts([entry.date for entry in logs_sorted])
        estimated_cycle_length, regularity = compute_cycle_stats(period_starts)

    records = []
    prev_sleep = np.nan
    prev_stress = np.nan
    prev_score = np.nan
    late_luteal_start = max(1, int(round(estimated_cycle_length)) - 6)

    for idx, entry in enumerate(logs_sorted):
        score = compute_bad_day_score(entry)
        cycle_day = cycle_day_for_date(entry.date, period_starts, estimated_cycle_length)
        late_luteal = 1 if cycle_day >= late_luteal_start else 0
        menstrual_window = 1 if cycle_day <= 5 else 0
        weekday = entry.date.weekday()

        record = {
            "date": entry.date,
            "score": score,
            "sleep_hours": float(entry.sleep_hours),
            "caffeine_level": int(entry.caffeine_level),
            "stress": float(entry.stress),
            "workout": int(entry.workout),
            "day_index": float(idx),
            "weekly_sin": float(np.sin((2.0 * np.pi * weekday) / 7.0)),
            "weekly_cos": float(np.cos((2.0 * np.pi * weekday) / 7.0)),
            "cycle_day": float(cycle_day),
            "cycle_day_normalized": float(cycle_day / max(1.0, estimated_cycle_length)),
            "late_luteal": float(late_luteal),
            "menstrual_window": float(menstrual_window),
            "basal_temp": float(entry.basal_temp) if entry.basal_temp is not None else np.nan,
            "resting_hr": float(entry.resting_hr) if entry.resting_hr is not None else np.nan,
            "basal_temp_missing": 1.0 if entry.basal_temp is None else 0.0,
            "resting_hr_missing": 1.0 if entry.resting_hr is None else 0.0,
            "sleep_prev": float(prev_sleep),
            "stress_prev": float(prev_stress),
            "score_prev": float(prev_score),
            "stress_x_late_luteal": float(entry.stress * late_luteal),
            "sleep_x_late_luteal": float(entry.sleep_hours * late_luteal),
            "stress_x_menstrual": float(entry.stress * menstrual_window),
            "sleep_x_menstrual": float(entry.sleep_hours * menstrual_window),
            "caffeine_x_sleep": float(entry.caffeine_level * entry.sleep_hours),
        }
        records.append(record)

        prev_sleep = entry.sleep_hours
        prev_stress = entry.stress
        prev_score = score

    frame = pd.DataFrame(records)

    for sensor_name, fallback in [("basal_temp", 36.5), ("resting_hr", 70.0)]:
        median = frame[sensor_name].median(skipna=True)
        if np.isnan(median):
            median = fallback
        frame[sensor_name] = frame[sensor_name].fillna(float(median))

    frame["sleep_prev"] = frame["sleep_prev"].fillna(frame["sleep_hours"])
    frame["stress_prev"] = frame["stress_prev"].fillna(frame["stress"])
    frame["score_prev"] = frame["score_prev"].fillna(frame["score"].expanding().mean())

    stats = {
        "estimated_cycle_length": float(estimated_cycle_length),
        "cycle_regularity_std": float(regularity),
    }
    return frame, stats


def low_variance_feature_flags(frame: pd.DataFrame) -> List[str]:
    low_variance = []
    for feature in FEATURE_COLUMNS:
        if feature not in frame.columns:
            continue
        values = frame[feature].astype(float)
        if values.nunique() <= 2 and values.std() < 0.05:
            low_variance.append(feature)
    return low_variance


def build_cycle_profile(frame: pd.DataFrame, cycle_length: float) -> Dict[str, float]:
    if frame.empty:
        return {}

    cycle_len = max(21, int(round(cycle_length)))
    grouped = frame.groupby("cycle_day")["score"].mean().to_dict()
    overall_mean = float(frame["score"].mean())
    profile: Dict[str, float] = {}

    for day in range(1, cycle_len + 1):
        exact = grouped.get(float(day), grouped.get(day))
        if exact is None:
            neighbor_values = []
            for offset in (1, 2, 3):
                for direction in (-1, 1):
                    neighbor_day = ((day - 1 + (direction * offset)) % cycle_len) + 1
                    maybe = grouped.get(float(neighbor_day), grouped.get(neighbor_day))
                    if maybe is not None:
                        neighbor_values.append(float(maybe))
            exact = float(np.mean(neighbor_values)) if neighbor_values else overall_mean

        profile[str(day)] = float(exact - overall_mean)

    return profile


def load_sample_logs() -> List[LogEntry]:
    sample_path = ROOT_DIR.parent / "sample_data.csv"
    if not sample_path.exists():
        return []

    frame = pd.read_csv(sample_path)
    logs: List[LogEntry] = []
    for _, row in frame.iterrows():
        logs.append(
            LogEntry(
                date=pd.to_datetime(row["date"]).date(),
                mood=float(row["mood"]),
                cramps=float(row["cramps"]),
                fatigue=float(row["fatigue"]),
                headache=float(row["headache"]),
                acne=float(row["acne"]),
                sleep_hours=float(row["sleep_hours"]),
                caffeine_level=int(row["caffeine_level"]),
                stress=float(row["stress"]),
                workout=bool(row["workout"]),
                basal_temp=float(row["basal_temp"]) if not pd.isna(row["basal_temp"]) else None,
                resting_hr=float(row["resting_hr"]) if not pd.isna(row["resting_hr"]) else None,
            )
        )
    return logs


def confidence_from_stats(num_rows: int, residual_std: float) -> tuple[float, str]:
    volume_factor = min(1.0, num_rows / 45.0)
    residual_factor = max(0.0, 1.0 - (residual_std / 1.25))
    confidence_value = (0.55 * volume_factor) + (0.45 * residual_factor)

    if confidence_value < 0.45:
        label = "Low"
    elif confidence_value < 0.75:
        label = "Medium"
    else:
        label = "High"

    return float(round(confidence_value, 3)), label


def wellness_from_severity(score: float) -> Dict[str, Any]:
    score = float(np.clip(score, 0.0, 5.0))
    wellness_score = 5.0 - score
    if score < 2.0:
        label = "Good"
        emoji = "🙂"
    elif score < 3.5:
        label = "Moderate"
        emoji = "😐"
    else:
        label = "Poor"
        emoji = "🙁"

    return {
        "severity_score": round(score, 2),
        "wellness_score": round(wellness_score, 2),
        "symptom_burden_pct": round((score / 5.0) * 100.0, 2),
        "status_label": label,
        "status_emoji": emoji,
    }


def human_driver_summary(driver: Dict[str, Any]) -> str:
    label = driver.get("label", driver.get("feature", "feature"))
    current_value = driver.get("current_value")
    baseline_value = driver.get("baseline_value")
    contribution = driver.get("contribution", 0.0)
    direction_text = "made this day look a bit tougher" if contribution >= 0 else "helped keep this day easier"
    return (
        f"{label}: {current_value} compared with your usual {baseline_value}. "
        f"That {direction_text} by about {abs(float(contribution)):.2f} points."
    )


def explain_local(model: Any, x_row: pd.DataFrame, feature_names: List[str], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    shap_values = None
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(x_row)
        if isinstance(raw, list):
            raw = raw[0]
        raw = np.asarray(raw)
        shap_values = raw.reshape(-1)
    except Exception:
        shap_values = None

    if shap_values is None:
        base_pred = float(model.predict(x_row)[0])
        medians = meta.get("feature_medians", {})
        contributions = []
        for feature in feature_names:
            baseline_value = float(medians.get(feature, x_row[feature].iloc[0]))
            changed = x_row.copy()
            changed.loc[:, feature] = baseline_value
            counterfactual_pred = float(model.predict(changed)[0])
            contributions.append(base_pred - counterfactual_pred)
        shap_values = np.asarray(contributions, dtype=float)

    drivers = []
    medians = meta.get("feature_medians", {})
    for idx, feature in enumerate(feature_names):
        contribution = float(shap_values[idx])
        current_value = float(x_row.iloc[0][feature])
        baseline_value = float(medians.get(feature, current_value))
        drivers.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "contribution": round(contribution, 2),
                "current_value": round(current_value, 2),
                "baseline_value": round(baseline_value, 2),
                "direction": "increases risk" if contribution >= 0 else "decreases risk",
            }
        )

    if all(abs(item["contribution"]) < 1e-6 for item in drivers):
        stds = np.asarray(meta.get("train_feature_std", [1.0] * len(feature_names)), dtype=float)
        stds = np.where(stds < 1e-6, 1.0, stds)
        medians_arr = np.asarray(
            [float(meta.get("feature_medians", {}).get(name, x_row[name].iloc[0])) for name in feature_names],
            dtype=float,
        )
        x = x_row.values[0]
        heuristic = ((x - medians_arr) / stds) * 0.04
        for idx, item in enumerate(drivers):
            item["contribution"] = round(float(heuristic[idx]), 2)
            item["direction"] = "increases risk" if item["contribution"] >= 0 else "decreases risk"

    drivers.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    prioritized = [item for item in drivers if item["feature"] in PRIMARY_DRIVER_FEATURES]
    if len(prioritized) < 5:
        missing = [item for item in drivers if item["feature"] not in PRIMARY_DRIVER_FEATURES]
        prioritized.extend(missing)

    return prioritized[:6]


def save_artifacts(model: Any, meta: Dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_artifacts() -> None:
    if MODEL_PATH.exists() and META_PATH.exists():
        loaded_meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        loaded_features = loaded_meta.get("feature_columns", [])
        loaded_target_version = loaded_meta.get("target_semantics_version")
        if loaded_features == FEATURE_COLUMNS and loaded_target_version == "mood_inverted_v2":
            model_state["model"] = joblib.load(MODEL_PATH)
            model_state["meta"] = loaded_meta


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train_model(request: TrainRequest) -> Dict[str, Any]:
    logs = request.logs
    original_log_count = len(logs)
    if not logs:
        logs = load_sample_logs()

    if len(logs) < 10:
        raise HTTPException(status_code=400, detail="At least 10 logs are required to train the model.")

    logs = create_reference_augmented_logs(logs, request.use_reference_boost)

    period_starts = request.period_starts
    if not period_starts:
        period_starts = infer_period_starts([entry.date for entry in logs])

    frame, stats = build_feature_frame(logs, period_starts)
    if frame.empty:
        raise HTTPException(status_code=400, detail="Unable to build training frame from logs.")

    frame = frame.sort_values("date").reset_index(drop=True)
    split_idx = int(len(frame) * 0.8)
    split_idx = max(1, min(split_idx, len(frame) - 1))

    train_frame = frame.iloc[:split_idx]
    test_frame = frame.iloc[split_idx:]

    if len(test_frame) < 2:
        raise HTTPException(status_code=400, detail="Not enough holdout rows for a time-based test split.")

    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["score"]
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["score"]

    automl_selection = fit_automl_if_available(x_train, y_train)
    if automl_selection is not None:
        model, model_name, model_choice_reason = automl_selection
    else:
        base_model, base_model_name = choose_model()
        model, model_name = select_best_model(base_model, base_model_name, x_train, y_train)
        model_choice_reason = "Using a boosted tree model selected with time-series CV."
        if model_name.startswith("sklearn_hist_gradient_boosting"):
            model_choice_reason = (
                "The backend is currently using sklearn HistGradientBoosting because AutoML, XGBoost, or LightGBM are not installed in this environment."
            )
        elif model_name.startswith("xgboost"):
            model_choice_reason = "The backend chose XGBoost after testing several boosted-tree settings with time-series cross-validation."
        elif model_name.startswith("lightgbm"):
            model_choice_reason = "The backend chose LightGBM after testing several boosted-tree settings with time-series cross-validation."

    y_test_pred = model.predict(x_test)
    baseline = np.full(len(y_test), float(y_train.mean()))

    mae = float(mean_absolute_error(y_test, y_test_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    baseline_mae = float(mean_absolute_error(y_test, baseline))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline)))

    y_all_pred = model.predict(frame[FEATURE_COLUMNS])

    source_x = x_test if len(test_frame) >= 3 else x_train
    source_y = y_test if len(test_frame) >= 3 else y_train
    importance_result = permutation_importance(
        model,
        source_x,
        source_y,
        n_repeats=16,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    raw_importance = np.maximum(importance_result.importances_mean, 0.0)
    if np.all(raw_importance < 1e-8):
        # Correlation fallback when permutation importance collapses on tiny/flat data.
        corr_fallback = []
        for feature in FEATURE_COLUMNS:
            feature_values = frame[feature].astype(float)
            if feature_values.nunique() <= 1:
                corr_fallback.append(0.0)
            else:
                corr = np.corrcoef(feature_values, frame["score"])[0, 1]
                corr_fallback.append(abs(float(corr)) if not np.isnan(corr) else 0.0)
        raw_importance = np.asarray(corr_fallback, dtype=float)

    importance_total = float(raw_importance.sum())
    if importance_total <= 1e-8:
        normalized_importance = np.full(len(raw_importance), 1.0 / len(raw_importance))
    else:
        normalized_importance = raw_importance / importance_total

    importance_items = []
    for idx, feature in enumerate(FEATURE_COLUMNS):
        importance_items.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "importance": round(float(raw_importance[idx]), 6),
                "importance_pct": round(float(normalized_importance[idx] * 100.0), 2),
            }
        )
    importance_items.sort(key=lambda item: item["importance"], reverse=True)

    residual_std = float(np.std(y_test - y_test_pred))
    confidence_value, confidence_label = confidence_from_stats(len(frame), residual_std)

    train_feature_mean = x_train.mean().tolist()
    train_feature_std = (x_train.std().fillna(1.0).replace(0.0, 1.0)).tolist()
    feature_medians = x_train.median().to_dict()
    low_variance_features = low_variance_feature_flags(frame)
    cycle_profile = build_cycle_profile(frame, stats.get("estimated_cycle_length", 28.0))

    meta = {
        "model_name": model_name,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(len(frame)),
        "residual_std": residual_std,
        "feature_columns": FEATURE_COLUMNS,
        "target_semantics_version": "mood_inverted_v2",
        "train_feature_mean": [float(item) for item in train_feature_mean],
        "train_feature_std": [float(item) for item in train_feature_std],
        "feature_medians": {key: float(value) for key, value in feature_medians.items()},
        "estimated_cycle_length": float(stats.get("estimated_cycle_length", 28.0)),
        "cycle_regularity_std": float(stats.get("cycle_regularity_std", 0.0)),
        "cycle_profile": cycle_profile,
        "train_target_mean": float(frame["score"].mean()),
        "low_variance_features": low_variance_features,
    }

    model_state["model"] = model
    model_state["meta"] = meta
    save_artifacts(model, meta)

    predictions = []
    for idx, row in frame.iterrows():
        predicted = float(y_all_pred[idx])
        predictions.append(
            {
                "date": str(row["date"]),
                "actual": round(float(row["score"]), 2),
                "predicted": round(predicted, 2),
                "actual_pct": round(float((row["score"] / 5.0) * 100.0), 2),
                "predicted_pct": round(float((predicted / 5.0) * 100.0), 2),
            }
        )

    return {
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "baseline_mae": round(baseline_mae, 2),
            "baseline_rmse": round(baseline_rmse, 2),
            "model_name": model_name,
        },
        "confidence": {
            "value": confidence_value,
            "label": confidence_label,
        },
        "global_importance": importance_items[:10],
        "predictions": predictions,
        "data_quality": {
            "rows_used": int(len(frame)),
            "used_reference_boost": bool(request.use_reference_boost and original_log_count > 0 and original_log_count < len(frame)),
            "low_variance_features": low_variance_features,
        },
        "training_summary": (
            "Training rebuilds the model from the logs currently loaded in this browser session and replaces the previous model. If FLAML AutoML is installed, it first picks the best regressor family."
        ),
        "model_choice_reason": model_choice_reason,
        "next_step_hint": "After training, pick a day to see a plain-language explanation, then run the 7-day outlook.",
        "score_scale": "0 to 5 symptom severity index (not probability)",
        "notes": "These are associations, not causation. Score is symptom severity, not a chance percentage.",
    }


@app.post("/predict")
def predict_day(request: PredictRequest) -> Dict[str, Any]:
    model = model_state.get("model")
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")

    period_starts = request.period_starts
    all_logs = list(request.history_logs)

    # Ensure the target log is included so lag and cycle features align with selected day.
    all_logs = [entry for entry in all_logs if entry.date != request.target_log.date]
    all_logs.append(request.target_log)

    if not period_starts:
        period_starts = infer_period_starts([entry.date for entry in all_logs])

    frame, _ = build_feature_frame(all_logs, period_starts)
    if frame.empty:
        raise HTTPException(status_code=400, detail="Unable to build features for prediction.")

    frame = frame.sort_values("date").reset_index(drop=True)
    match = frame[frame["date"] == request.target_log.date]
    if match.empty:
        raise HTTPException(status_code=400, detail="Target date not found after feature processing.")

    x_row = match.iloc[[-1]][FEATURE_COLUMNS]
    prediction = float(model.predict(x_row)[0])
    actual_logged_score = float(compute_bad_day_score(request.target_log))
    gap = actual_logged_score - prediction
    expected_wellness = wellness_from_severity(prediction)
    actual_wellness = wellness_from_severity(actual_logged_score)

    residual_std = float(model_state["meta"].get("residual_std", 1.0))
    n_samples = int(model_state["meta"].get("n_samples", len(all_logs)))
    confidence_value, confidence_label = confidence_from_stats(n_samples, residual_std)

    drivers = explain_local(model, x_row, FEATURE_COLUMNS, model_state["meta"])
    driver_summaries = [human_driver_summary(item) for item in drivers[:4]]

    if gap > 0.35:
        comparison_text = "This day felt tougher than the model expected."
    elif gap < -0.35:
        comparison_text = "This day felt easier than the model expected."
    else:
        comparison_text = "This day matched the model's expectation fairly closely."

    interpretation = (
        f"Expected score from cycle and lifestyle signals: {expected_wellness['severity_score']:.2f}/5. "
        f"Your logged symptom score for that day was {actual_logged_score:.2f}/5. {comparison_text} "
        "The logged score only uses mood, cramps, fatigue, headache, and acne. The model estimate uses non-symptom inputs such as sleep, stress, workout, caffeine, cycle timing, prior history, and optional sensors."
    )

    return {
        "prediction": expected_wellness["severity_score"],
        "actual_logged_score": round(actual_logged_score, 2),
        "actual_status_label": actual_wellness["status_label"],
        "actual_status_emoji": actual_wellness["status_emoji"],
        "predicted_status_label": expected_wellness["status_label"],
        "predicted_status_emoji": expected_wellness["status_emoji"],
        "expected_gap": round(gap, 2),
        "symptom_burden_pct": expected_wellness["symptom_burden_pct"],
        "wellness_score": expected_wellness["wellness_score"],
        "status_label": expected_wellness["status_label"],
        "status_emoji": expected_wellness["status_emoji"],
        "score_scale": "0 to 5 symptom severity index (not probability)",
        "confidence": {
            "value": confidence_value,
            "label": confidence_label,
        },
        "drivers": drivers,
        "driver_summaries": driver_summaries,
        "interpretation": interpretation,
        "notes": "This estimate uses only non-symptom inputs such as sleep, stress, caffeine, workout, cycle timing, prior history, and optional sensors. The logged symptom score is defined only by mood, cramps, fatigue, headache, and acne.",
    }


@app.post("/forecast")
def forecast_days(request: ForecastRequest) -> Dict[str, Any]:
    model = model_state.get("model")
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")

    start_date = request.start_date or date.today()

    if request.last_period_start:
        last_period_start = request.last_period_start
    elif request.period_starts:
        last_period_start = max(request.period_starts)
    else:
        last_period_start = start_date

    if request.estimated_cycle_length:
        estimated_cycle_length = float(request.estimated_cycle_length)
    elif request.period_starts:
        estimated_cycle_length, _ = compute_cycle_stats(request.period_starts)
    else:
        estimated_cycle_length = float(model_state["meta"].get("estimated_cycle_length", 28.0))

    cycle_length = max(21, int(round(estimated_cycle_length)))
    late_luteal_start = max(1, cycle_length - 6)

    typical = request.typical_features
    mean_map = dict(zip(FEATURE_COLUMNS, model_state["meta"].get("train_feature_mean", [0.0] * len(FEATURE_COLUMNS))))

    base_sleep = float(typical.get("sleep_hours", mean_map.get("sleep_hours", 7.0)))
    base_caffeine = float(typical.get("caffeine_level", mean_map.get("caffeine_level", 1.0)))
    base_stress = float(typical.get("stress", mean_map.get("stress", 2.5)))
    base_workout = float(typical.get("workout", 0.0))
    base_temp = float(typical.get("basal_temp", mean_map.get("basal_temp", 36.5)))
    base_hr = float(typical.get("resting_hr", mean_map.get("resting_hr", 70.0)))

    score_prev = float(typical.get("score_prev", 2.5))
    sleep_prev = float(typical.get("sleep_prev", base_sleep))
    stress_prev = float(typical.get("stress_prev", base_stress))
    base_day_index = float(typical.get("day_index", model_state["meta"].get("n_samples", 0)))

    forecast = []
    residual_std = float(model_state["meta"].get("residual_std", 1.0))
    n_samples = int(model_state["meta"].get("n_samples", 0))
    confidence_value, confidence_label = confidence_from_stats(n_samples, residual_std)
    cycle_profile = model_state["meta"].get("cycle_profile", {})
    train_target_mean = float(model_state["meta"].get("train_target_mean", 2.5))

    for offset in range(7):
        day = start_date + timedelta(days=offset)
        cycle_day = ((day - last_period_start).days % cycle_length) + 1
        late_luteal = 1.0 if cycle_day >= late_luteal_start else 0.0
        menstrual_window = 1.0 if cycle_day <= 5 else 0.0
        weekday = day.weekday()
        day_index = base_day_index + offset

        row = {
            "sleep_hours": base_sleep,
            "caffeine_level": base_caffeine,
            "stress": base_stress,
            "workout": base_workout,
            "day_index": day_index,
            "weekly_sin": float(np.sin((2.0 * np.pi * weekday) / 7.0)),
            "weekly_cos": float(np.cos((2.0 * np.pi * weekday) / 7.0)),
            "cycle_day": float(cycle_day),
            "cycle_day_normalized": float(cycle_day / cycle_length),
            "late_luteal": late_luteal,
            "menstrual_window": menstrual_window,
            "basal_temp": base_temp,
            "resting_hr": base_hr,
            "basal_temp_missing": 0.0,
            "resting_hr_missing": 0.0,
            "sleep_prev": sleep_prev,
            "stress_prev": stress_prev,
            "score_prev": score_prev,
            "stress_x_late_luteal": base_stress * late_luteal,
            "sleep_x_late_luteal": base_sleep * late_luteal,
            "stress_x_menstrual": base_stress * menstrual_window,
            "sleep_x_menstrual": base_sleep * menstrual_window,
            "caffeine_x_sleep": base_caffeine * base_sleep,
        }

        x_row = pd.DataFrame([row])[FEATURE_COLUMNS]
        model_pred = float(model.predict(x_row)[0])
        cycle_effect = float(cycle_profile.get(str(cycle_day), 0.0))
        predicted_score = (0.72 * model_pred) + (0.28 * (train_target_mean + cycle_effect))
        predicted_score = float(np.clip(predicted_score, 0.0, 5.0))
        wellness = wellness_from_severity(predicted_score)
        score_prev = predicted_score

        if predicted_score < 2.0:
            risk = "Low"
        elif predicted_score < 3.5:
            risk = "Medium"
        else:
            risk = "High"

        forecast.append(
            {
                "date": str(day),
                "predicted_bad_day_score": wellness["severity_score"],
                "symptom_burden_pct": wellness["symptom_burden_pct"],
                "wellness_score": wellness["wellness_score"],
                "status_label": wellness["status_label"],
                "status_emoji": wellness["status_emoji"],
                "risk_label": risk,
            }
        )

    return {
        "score_scale": "0 to 5 symptom severity index (not probability)",
        "confidence": {
            "value": confidence_value,
            "label": confidence_label,
        },
        "forecast": forecast,
        "notes": "These are associations, not causation. Forecast labels describe symptom load, not mood quality.",
    }
