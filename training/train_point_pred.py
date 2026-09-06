"""
train.py

Strategy:
  1. TUNE on a ~150k sample. Best hyperparameters on 150k == best on 1M; you only
     need the full data to train the FINAL model, not to search.
  2. Let OPTUNA choose configs (smart search) instead of an exhaustive
     grid. 50 smart trials beat hundreds of dumb ones. 
  3. EARLY STOPPING on the boosting models: n_estimators is a high ceiling and each
     fit stops when validation stops improving.
  4. REFIT each winner ONCE on the full dataset

Logging: Optuna stores EVERY trial (params + score + state) in its own SQLite
study DB automatically, and can resume after a crash. We add one small `results`
table for the final refit-on-full-data metrics per model.

Run:  python train.py
Inspect trials later:  optuna-dashboard sqlite:///Outputs/optuna_study.db
"""

import json
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
import uuid, subprocess

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score, make_scorer,
)
import argparse
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--features", choices=["base", "full"], default="full",
                    help="base = property attributes only; full = everything")
args = parser.parse_args()

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "input": "./Outputs/clean_property_data.parquet",
    "full_rows": None,      # rows used to refit the final winners. None = all
    "tune_rows": 300_000,        # rows used for the Optuna search
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,                # fold 
    "n_trials": 150,              # Optuna trials per model
    "scoring": "MAPE",            # the objective Optuna minimises - for logger
    "study_db": "./Outputs/optuna_study.db",
    "results_db": "./Outputs/experiments.db",
}
RS = CONFIG["random_state"]

# =============================================================================
# LOAD + FEATURES
# =============================================================================
print("Loading data...")
df_full = pd.read_parquet(CONFIG["input"])
if CONFIG["full_rows"]:
    df_full = df_full.sample(n=min(CONFIG["full_rows"], len(df_full)), random_state=RS)
print(f"  Full: {len(df_full):,} rows")

target = "price"

# --- Base: No spatial features, just very basic features ---
base_numeric = [
    "total_floor_area", "current_energy_efficiency", "number_habitable_rooms",
    "sale_time", "construction_year", "construction_year_exact", "property_age",
]
base_categorical = [
    "property_type_x", "new_build", "duration", "current_energy_rating",
    "tenure", "built_form", "construction_age_band", "main_fuel",
]

# --- Full: All features ---

numeric_features = [
    "total_floor_area", "current_energy_efficiency", "number_habitable_rooms",
    "exact_lat", "exact_lon", "dist_primary_km", "dist_secondary_km",
    "dist_rail_km", "rail_within_1km", "rail_within_5km", "dist_metro_km",
    "metro_within_1km", "dist_airport_km", "dist_coast_km", "dist_town_km",
    "sale_time", "construction_year", "construction_year_exact", "property_age", 
    'area_past_price',
    # comparables
    "comp1_price", "comp1_floor_area", "comp1_distance_km", "comp1_days_ago",
    "comp2_price", "comp2_floor_area", "comp2_distance_km", "comp2_days_ago",
    "comp3_price", "comp3_floor_area", "comp3_distance_km", "comp3_days_ago",
]
categorical_features = [
    "property_type_x", "new_build", "duration", "current_energy_rating",
    "tenure", "built_form", "construction_age_band", "main_fuel",
]

if args.features == "base":
    numeric_features = base_numeric
    categorical_features = base_categorical

numeric_features = [f for f in numeric_features if f in df_full.columns]
categorical_features = [f for f in categorical_features if f in df_full.columns]
all_features = numeric_features + categorical_features

X_full = df_full[all_features]
y_full = np.log1p(df_full[target])

# Tuning subsample (drawn from full)
tune_idx = X_full.sample(n=min(CONFIG["tune_rows"], len(X_full)), random_state=RS).index
X_tune, y_tune = X_full.loc[tune_idx], y_full.loc[tune_idx]

# Hold-out test from the FULL data, for final evaluation
X_tr_full, X_te, y_tr_full, y_te = train_test_split(
    X_full, y_full, test_size=CONFIG["test_size"], random_state=RS)

print(f"  Tune on {len(X_tune):,} | final refit on {len(X_tr_full):,} | test {len(X_te):,}")


# TODO add SEARCH_SPACE and add it to logging.

# =============================================================================
# PREPROCESSOR  (built once; wrapped into each model's pipeline)
# =============================================================================
def make_preprocessor():
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, numeric_features),
                              ("cat", cat, categorical_features)])

def _mape_pounds(y_true_log, y_pred_log):
    return mean_absolute_percentage_error(np.expm1(y_true_log), np.expm1(y_pred_log))

mape_scorer = make_scorer(_mape_pounds, greater_is_better=False)

# MAPE as an sklearn scorer (higher = better -> negative MAPE)
cv = KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=RS)


def cv_mape(estimator, X, y):
    """Mean CV MAPE (%). Positive, lower is better."""
    scores = cross_val_score(estimator, X, y, scoring=mape_scorer, cv=cv, n_jobs=1)
    return -scores.mean() * 100


# =============================================================================
# OPTUNA OBJECTIVES  (one per model; each returns CV MAPE % to minimize)
# =============================================================================
def obj_ridge(trial):
    alpha = trial.suggest_float("alpha", 1.0, 1000, log=True)
    pipe = Pipeline([("pre", make_preprocessor()), ("reg", Ridge(alpha=alpha))])
    return cv_mape(pipe, X_tune, y_tune)


def obj_lasso(trial):
    alpha = trial.suggest_float("alpha", 0.01, 10, log=True)
    pipe = Pipeline([("pre", make_preprocessor()), ("reg", Lasso(alpha=alpha))])
    return cv_mape(pipe, X_tune, y_tune)


def obj_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }
    pipe = Pipeline([("pre", make_preprocessor()),
                     ("reg", RandomForestRegressor(random_state=RS, n_jobs=-1, **params))])
    return cv_mape(pipe, X_tune, y_tune)


def obj_lgbm(trial):
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 18),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100, log=True),
    }
    # Early stopping needs a validation split -> preprocess once, fit with callback.
    pre = make_preprocessor()
    Xt = pre.fit_transform(X_tune)
    Xtr, Xval, ytr, yval = train_test_split(Xt, y_tune.to_numpy(), test_size=0.2, random_state=RS)
    model = LGBMRegressor(random_state=RS, n_jobs=-1, verbose=-1, **params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="mape", callbacks=[early_stopping(50, verbose=False)])
    pred = np.expm1(model.predict(Xval))
    return mean_absolute_percentage_error(np.expm1(yval), pred) * 100


def obj_xgb(trial):
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100, log=True),
        "gamma": trial.suggest_float("gamma", 0, 2),
    }
    pre = make_preprocessor()
    Xt = pre.fit_transform(X_tune)
    Xtr, Xval, ytr, yval = train_test_split(Xt, y_tune.to_numpy(), test_size=0.2, random_state=RS)
    model = XGBRegressor(random_state=RS, n_jobs=-1, tree_method="hist", early_stopping_rounds=50, eval_metric="mape", **params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    pred = np.expm1(model.predict(Xval))
    return mean_absolute_percentage_error(np.expm1(yval), pred) * 100

OBJECTIVES = {
    #"Ridge": obj_ridge,
    #"Lasso": obj_lasso,
    #"XGBoost": obj_xgb,
    #"LightGBM": obj_lgbm,
    "RandomForest": obj_rf,
}

# =============================================================================
# FINAL-MODEL BUILDERS  (winner refit on FULL data)
# =============================================================================
def build_final(name, best_params):
    """Return an unfitted pipeline for the winning params, ready to fit on full data."""
    if name == "Ridge":
        reg = Ridge(**best_params)
    elif name == "Lasso":
        reg = Lasso(**best_params)
    elif name == "RandomForest":
        reg = RandomForestRegressor(random_state=RS, n_jobs=-1, **best_params)
    elif name == "LightGBM":
        reg = LGBMRegressor(random_state=RS, n_jobs=-1, verbose=-1, n_estimators=2000, **best_params)
    elif name == "XGBoost":
        reg = XGBRegressor(random_state=RS, n_jobs=-1, tree_method="hist", n_estimators=2000, device="cuda", **best_params)
    return Pipeline([("pre", make_preprocessor()), ("reg", reg)])


def full_metrics(pipe, Xtr, ytr, Xte, yte):
    pr_tr, pr_te = np.expm1(pipe.predict(Xtr)), np.expm1(pipe.predict(Xte))
    ytr, yte = np.expm1(ytr), np.expm1(yte)
    return {
        "train_rmse": float(np.sqrt(mean_squared_error(ytr, pr_tr))),
        "test_rmse": float(np.sqrt(mean_squared_error(yte, pr_te))),
        "train_mae": float(mean_absolute_error(ytr, pr_tr)),
        "test_mae": float(mean_absolute_error(yte, pr_te)),
        "train_r2": float(r2_score(ytr, pr_tr)),
        "test_r2": float(r2_score(yte, pr_te)),
        "train_mape": float(mean_absolute_percentage_error(ytr, pr_tr) * 100),
        "test_mape": float(mean_absolute_percentage_error(yte, pr_te) * 100),
    }


#TODO : get rid of git commit column, its not needed
def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def init_db(db):
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, timestamp TEXT, git_commit TEXT,
            scoring TEXT, cv_folds INTEGER, n_trials INTEGER,
            tune_rows INTEGER, full_rows INTEGER, n_train INTEGER, n_test INTEGER,
            test_size REAL, random_state INTEGER,
            n_features INTEGER, numeric_features TEXT, categorical_features TEXT,
            target TEXT, input_path TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS results (
            run_id TEXT, model TEXT, best_params TEXT,
            tune_score REAL, n_trials INTEGER, tune_seconds REAL, refit_seconds REAL,
            train_rmse REAL, test_rmse REAL, train_mae REAL, test_mae REAL,
            train_r2 REAL, test_r2 REAL, train_mape REAL, test_mape REAL,
            timestamp TEXT,
            PRIMARY KEY (run_id, model),
            FOREIGN KEY (run_id) REFERENCES runs(run_id))""")


def log_run(db, run_id):
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            run_id, datetime.now().isoformat(timespec="seconds"), _git_commit(),
            CONFIG["scoring"], CONFIG["cv_folds"], CONFIG["n_trials"],
            len(X_tune), len(X_full), len(X_tr_full), len(X_te),
            CONFIG["test_size"], RS,
            len(all_features), json.dumps(numeric_features),
            json.dumps(categorical_features), target, CONFIG["input"]))


def log_result(db, row):
    with sqlite3.connect(db) as c:
        cols = ",".join(row.keys())
        c.execute(f"INSERT INTO results ({cols}) VALUES ({','.join('?'*len(row))})",
                  list(row.values()))

# =============================================================================
# RUN
# =============================================================================
run_id = uuid.uuid4().hex[:12]
init_db(CONFIG["results_db"])
log_run(CONFIG["results_db"], run_id)
print(f"\nRun ID: {run_id}  (scoring={CONFIG['scoring']}, cv={CONFIG['cv_folds']})")
summary = []

for name, objective in OBJECTIVES.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{name}_{run_id}",
        storage=f"sqlite:///{CONFIG['study_db']}",
        load_if_exists=True,
    )
    n = 30 if name in ("Ridge", "Lasso") else CONFIG["n_trials"]  # Ridge has 1 knob - don't waste trials
    t0 = time.time()
    study.optimize(objective, n_trials=n, show_progress_bar=True)
    tune_seconds = time.time() - t0
    print(f"  Best CV MAPE: {study.best_value:.2f}%  params={study.best_params}")

    # Refit winner on FULL training data, evaluate on the held-out test set.
    final = build_final(name, study.best_params)
    t0 = time.time()
    final.fit(X_tr_full, y_tr_full)
    refit_seconds = time.time() - t0
    m = full_metrics(final, X_tr_full, y_tr_full, X_te, y_te)
    print(f"  FULL refit  Test MAPE {m['test_mape']:.1f}% | "
          f"RMSE £{m['test_rmse']:,.0f} | R² {m['test_r2']:.4f} "
          f"({refit_seconds:.0f}s)")

    row = {
        "run_id": run_id,
        "model": name, "best_params": json.dumps(study.best_params),
        "tune_score": float(study.best_value), "n_trials": n,
        "tune_seconds": round(tune_seconds, 1), "refit_seconds": round(refit_seconds, 1),
        **m,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    log_result(CONFIG["results_db"], row)
    summary.append(row)

# =============================================================================
# SUMMARY  (read straight back from the DB — no separate output file)
# =============================================================================
con = sqlite3.connect(CONFIG["results_db"])
print(f"\n{'='*60}\nRESULTS  (run {run_id})\n{'='*60}")
print(pd.read_sql_query(
    "SELECT model, ROUND(test_mape,2) AS test_mape, "
    "ROUND(test_rmse) AS test_rmse, ROUND(test_r2,4) AS test_r2, "
    "ROUND(tune_seconds,1) AS tune_s "
    "FROM results WHERE run_id=? ORDER BY test_mape",
    con, params=(run_id,)).to_string(index=False))

best = pd.read_sql_query(
    "SELECT model, test_mape FROM results WHERE run_id=? ORDER BY test_mape LIMIT 1",
    con, params=(run_id,))
print(f"\nBest: {best.iloc[0]['model']} (Test MAPE {best.iloc[0]['test_mape']:.2f}%)")
print(f"\nEverything is in {CONFIG['results_db']}:")
print("  runs     - this run's full config (scoring, folds, features, sample sizes)")
print("  results  - final metrics per model, joined to runs by run_id")
print(f"Every Optuna trial: {CONFIG['study_db']}  "
      f"(study names '<Model>_{run_id}')")
