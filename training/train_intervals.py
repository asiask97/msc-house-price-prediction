"""
train_intervals.py

Prediction intervals via Conformalized Quantile Regression.

Per model we:
  1. Split the data 4 ways: train / val / calibration / test. - no CV split cause it takes too long to train.
  2. Produce a raw lower + upper quantile band.
    HOW:
       - Boosters (XGBoost, LightGBM): two separate quantile models, each tuned with Optuna on pinball loss and warm-started from that model's best point prediction params (from the DB).
       - Linear (QuantileRegressor): two separate quantile models. The LP solver is very slow, so we train on a capped subsample with a tiny search.
       - RandomForest (quantile-forest): One forest that answers both quantiles at once (a forest already holds the full spread of leaf values, so you don't train two models).
         TODO: Try two RF models, one for each quantile, and see if that improves the band.
  3. conformalize on the calibration set: raw bands are usually off (an 80%" band might really catch ~76%), so we measure that gap on calibration set houses and get one number, Q which widens the band just enough to make 80% actually hold.
  4. evaluate on test with coverage + width. log and save the fitted model to Outputs/interval-models/ so it can be reloaded later without retraining.


Run:  python train_intervals.py --features full
      python train_intervals.py --features base ( those will use point predictions with base params as warm up point)
"""

import argparse
import json
import os
import sqlite3
import time
import uuid
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from quantile_forest import RandomForestQuantileRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--features", choices=["base", "full"], default="full",
                    help="base = property attributes only; full = everything "
                         "(spatial + comparables), matching train.py")
args = parser.parse_args()


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "input": "./Outputs/clean_property_data.parquet",
    "point_db": "./Outputs/experiments.db",
    "interval_db": "./Outputs/interval_experiments.db",

    # Every model here gets its own interval band + saved model.
    "models": ["XGBoost", "LightGBM", "RandomForest", "Linear"],

    "confidence": 0.80,
    "full_rows": 1_000_000,
    "test_size": 0.20,
    "calib_size": 0.20,
    "val_size": 0.15,
    "random_state": 42,

    "n_trials": 30,          # boosters
    "n_trials_linear": 6,    # QuantileRegressor
    "n_trials_rf": 5,        # quantile-forest: each fit is minutes

    "n_estimators": 2000,
    "early_stopping": 50,

    # Hard row caps for the slow families
    "linear_max_rows": 10_000,
    "rf_max_rows": 50_000,
}
RS = CONFIG["random_state"]
ALPHA  = 1.0 - CONFIG["confidence"]
Q_LOW  = ALPHA / 2.0
Q_HIGH = 1.0 - ALPHA / 2.0
FAMILY = {
    "XGBoost": "booster",
    "LightGBM": "booster",
    "RandomForest": "qrf",
    "Linear": "linear",
}
SEARCH_KEYS = {
    "XGBoost":  ["learning_rate", "max_depth", "min_child_weight", "subsample",
                 "colsample_bytree", "reg_lambda", "reg_alpha", "gamma"],
    "LightGBM": ["learning_rate", "max_depth", "num_leaves", "min_child_samples",
                 "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"],
    "RandomForest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
}


# =============================================================================
# FEATURES
# =============================================================================
FULL_NUMERIC = [
    "total_floor_area", "current_energy_efficiency", "number_habitable_rooms",
    "exact_lat", "exact_lon", "dist_primary_km", "dist_secondary_km",
    "dist_rail_km", "rail_within_1km", "rail_within_5km", "dist_metro_km",
    "metro_within_1km", "dist_airport_km", "dist_coast_km", "dist_town_km",
    "sale_time", "construction_year", "construction_year_exact", "property_age",
    "area_past_price",
    # comparables
    "comp1_price", "comp1_floor_area", "comp1_distance_km", "comp1_days_ago",
    "comp2_price", "comp2_floor_area", "comp2_distance_km", "comp2_days_ago",
    "comp3_price", "comp3_floor_area", "comp3_distance_km", "comp3_days_ago",
]
FULL_CATEGORICAL = [
    "property_type_x", "new_build", "duration", "current_energy_rating",
    "tenure", "built_form", "construction_age_band", "main_fuel",
]

BASE_NUMERIC = [
    "total_floor_area", "current_energy_efficiency", "number_habitable_rooms",
    "sale_time", "construction_year", "construction_year_exact", "property_age",
]
BASE_CATEGORICAL = [
    "property_type_x", "new_build", "duration", "current_energy_rating",
    "tenure", "built_form", "construction_age_band", "main_fuel",
]

if args.features == "base":
    NUMERIC_FEATURES, CATEGORICAL_FEATURES = BASE_NUMERIC, BASE_CATEGORICAL
else:
    NUMERIC_FEATURES, CATEGORICAL_FEATURES = FULL_NUMERIC, FULL_CATEGORICAL

TARGET = "price"


# =============================================================================
# DATA
# =============================================================================
def load_and_split():
    '''
    using fixed validation instead of cv for time efficiency. 
    It already takes long time to train. training cv=3 would take 3 time longer. 
    
    so we are splitting the data into 4 piles:
     - train - the model learns the patterns here.
     - val - try different settings and pick the best. Need unseen houses to judge settings honestly.
     - calibration -  measure the honesty number Q here (how much to widen the band). 
     - test - the final exam. Coverage and width get reported from here, and this pile is used once at the very end.
    '''
    df = pd.read_parquet(CONFIG["input"])
    df = df.sample(n=min(CONFIG["full_rows"], len(df)), random_state=RS)
    num = [f for f in NUMERIC_FEATURES if f in df.columns]
    cat = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    X, y = df[num + cat], np.log1p(df[TARGET])
    # does some fraction maths to get each section size of dataset set correctly
    X_rest, X_test, y_rest, y_test = train_test_split( X, y, test_size=CONFIG["test_size"], random_state=RS)
    calib_frac = CONFIG["calib_size"] / (1 - CONFIG["test_size"])
    X_tr2, X_cal, y_tr2, y_cal = train_test_split(X_rest, y_rest, test_size=calib_frac, random_state=RS)
    val_frac = CONFIG["val_size"] / (1 - CONFIG["test_size"] - CONFIG["calib_size"])
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr2, y_tr2, test_size=val_frac, random_state=RS)
    return (X_tr, y_tr, X_val, y_val, X_cal, y_cal, X_test, y_test), num, cat


def make_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),cat),
    ])



# =============================================================================
# WARM-START
# =============================================================================

def point_params(db, model, n_features):
    """
    Using best point prediction params to help and find the most accurate params for each quantile
    base both full params in args will use only those models with those base features as a warm up
    """
    if not os.path.exists(db):
        return {}, None
    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT r.best_params, r.run_id FROM results r "
        "JOIN runs u ON r.run_id = u.run_id "
        "WHERE r.model=? AND u.n_features=? AND r.test_mape IS NOT NULL "
        "ORDER BY r.test_mape ASC LIMIT 1",
        (model, n_features)).fetchone()
    con.close()
    if row is None:
        return {}, None
    return json.loads(row[0]), row[1]


# =============================================================================
# QUANTILE LOSS
# =============================================================================
def pinball(y_true, y_pred, q):
    '''
    Garde the low and high predictions using pinball loss/ quantile loss
    '''
    d = y_true - y_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# -----------------------------------------------------------------------------
# BOOSTER models (XGBoost / LightGBM)
# -----------------------------------------------------------------------------
def suggest_booster_params(trial, model):
    if model == "XGBoost":
        return {
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 100, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        }
    return {  # LightGBM
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 14),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 100, log=True),
    }


def fit_booster_quantile(model, params, q, Xtr, ytr, Xval, yval):
    """
    Trains one model. it takes in which quantile, params, and train and val data to train one model. 
    """
    if model == "XGBoost":
        m = XGBRegressor(objective="reg:quantileerror", #train it as a quantile model, not a normal one
                         quantile_alpha=q,
                         n_estimators=CONFIG["n_estimators"],
                         early_stopping_rounds=CONFIG["early_stopping"],
                         tree_method="hist", 
                         random_state=RS, 
                         n_jobs=-1, 
                         **params #adding all the suggest_booster_params
                        )
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    else:  # LightGBM
        m = LGBMRegressor(objective="quantile", 
                          alpha=q,
                          n_estimators=CONFIG["n_estimators"],
                          random_state=RS, 
                          n_jobs=-1, 
                          verbose=-1, 
                          **params #adding all the suggest_booster_params
                        )
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="quantile", callbacks=[early_stopping(CONFIG["early_stopping"], verbose=False)])
    return m


def tune_booster(model, q, Xtr, ytr, Xval, yval, warm):
    '''
    Runs the actual optuna search. calls fit_booster_quantile n_trials amount for each quantile.
    '''
    def objective(trial):
        params = suggest_booster_params(trial, model) # 1 - get dials to try
        m = fit_booster_quantile(model, params, q, Xtr, ytr, Xval, yval) # 2 - train a one model with them
        return pinball(yval, m.predict(Xval), q) # 3 - grade that model on val. pinball loss score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RS))
    if warm:  # first trial = use the best point config, if we have one
        study.enqueue_trial({k: warm[k] for k in SEARCH_KEYS[model] if k in warm})
    study.optimize(objective, n_trials=CONFIG["n_trials"], show_progress_bar=True)

    best = fit_booster_quantile(model, study.best_params, q, Xtr, ytr, Xval, yval)
    return best, study.best_params, study.best_value


# -----------------------------------------------------------------------------
# LINEAR models (QuantileRegressor)
# -----------------------------------------------------------------------------
def fit_linear_quantile(alpha, q, Xtr, ytr):
    # solver="highs" is the fastest but still slow
    m = QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
    m.fit(Xtr, ytr)
    return m


def tune_linear(q, Xtr, ytr, Xval, yval):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        m = fit_linear_quantile(alpha, q, Xtr, ytr)
        return pinball(yval, m.predict(Xval), q)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RS))
    # not using the best linear point model params like with boosting cause QuantileRegressor is different and those do not transfer.
    study.optimize(objective, n_trials=CONFIG["n_trials_linear"], show_progress_bar=True)

    best = fit_linear_quantile(study.best_params["alpha"], q, Xtr, ytr)
    return best, study.best_params, study.best_value


# -----------------------------------------------------------------------------
#  FOREST (RandomForest) 
# -----------------------------------------------------------------------------


def fit_qrf(params, Xtr, ytr):
    qrf = RandomForestQuantileRegressor(random_state=RS, n_jobs=-1, **params)
    qrf.fit(Xtr, ytr)
    return qrf


def tune_qrf(Xtr, ytr, Xval, yval, warm):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 200, step=50),
            "max_depth":        trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
        }
        qrf = fit_qrf(params, Xtr, ytr)
        pred = np.asarray(qrf.predict(Xval, quantiles=[Q_LOW, Q_HIGH]))
        # One forest serves both bounds, so we optimise the average pinball of the low and high quantile
        return 0.5 * (pinball(yval, pred[:, 0], Q_LOW) + pinball(yval, pred[:, 1], Q_HIGH))


# -----------------------------------------------------------------------------
# Unified train + predict over families
# -----------------------------------------------------------------------------
def train_interval(model, family, Xtr, ytr, Xval, yval, warm):
    """
    Train one interval model (two quantile models, or one forest) and 
    return the fitted model + the best params + the best pinball score for each quantile.
    """
    if family == "booster":
        lo_m, lo_p, lo_pin = tune_booster(model, Q_LOW, Xtr, ytr, Xval, yval, warm)
        hi_m, hi_p, hi_pin = tune_booster(model, Q_HIGH, Xtr, ytr, Xval, yval, warm)
        return {"kind": "pair", "lo_m": lo_m, "hi_m": hi_m}, lo_p, hi_p, lo_pin, hi_pin

    if family == "linear":
        # LP solver is slow, so we cap the training rows.
        cap = CONFIG["linear_max_rows"]
        Xs, ys = Xtr[:cap], ytr[:cap]
        lo_m, lo_p, lo_pin = tune_linear(Q_LOW, Xs, ys, Xval, yval)
        hi_m, hi_p, hi_pin = tune_linear(Q_HIGH, Xs, ys, Xval, yval)
        return {"kind": "pair", "lo_m": lo_m, "hi_m": hi_m}, lo_p, hi_p, lo_pin, hi_pin

    if family == "qrf":
        cap = CONFIG["rf_max_rows"]
        Xs, ys = Xtr[:cap], ytr[:cap]
        qrf, p, val = tune_qrf(Xs, ys, Xval, yval, warm)
        # One forest model - same params/score reported for both bounds.
        return {"kind": "qrf", "qrf": qrf, "q_low": Q_LOW, "q_high": Q_HIGH}, p, p, val, val

    raise ValueError(f"unknown family: {family}")


def pred_lo_hi(interval_model, Xproc):
    """
    returns on best optuna model results for lower and upper predictions. 
    
    Two for boosters/linear, one for qrf. 
    """
    if interval_model["kind"] == "pair":
        return interval_model["lo_m"].predict(Xproc), interval_model["hi_m"].predict(Xproc)
    pred = np.asarray(interval_model["qrf"].predict(Xproc, quantiles=[interval_model["q_low"], interval_model["q_high"]]))
    return pred[:, 0], pred[:, 1]


# =============================================================================
# CQR + EVAL
# =============================================================================
def conformal_offset(lo_cal, hi_cal, y_cal, alpha):
    """
    we check how far off band misses on calibration set, and widen the band by that amount to make it honest.
    (honest means that the band actually covers the target confidence level)
    """
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal) # how far each price fell outside the band
    # (1 - alpha)  -> target coverage (e.g. 0.80)
    # (len(y_cal) + 1) * 0.80 -> the 80% position,
    # np.ceil(...) -> round up to a whole position (can't grab the 800.4th item) cause position is sorted, so position = coverage 
    # / len(y_cal) -> convert that position back into a fraction between 0 and 1, because the next line (np.quantile) wants a fraction, not an index.
    # min(..., 1.0) -> safety cap , never let it exceed 1.0 or 100%
    level = min(np.ceil((len(y_cal) + 1) * (1 - alpha)) / len(y_cal), 1.0) 
    return float(np.quantile(scores, level, method="higher")) # Q is the miss-distance at that cutoff. how much to widen the band


def evaluate(lo, hi, y_true_log):
    """
    Final exam gives a score to a finished band on the test houses.
    Returns coverage (if we catch ~80%) and width (how tight) the two numbers each model is judged on. 
    """
    inside = (y_true_log >= lo) & (y_true_log <= hi) # check if the real price land in the band
    lo_gbp, hi_gbp = np.expm1(lo), np.expm1(hi)      # undo the log on band edges
    price = np.expm1(y_true_log)                     # undo the log o real price in GBP
    width = hi_gbp - lo_gbp                          # band size in GBP
    rel = width / price * 100.0                      # width as % of price 
    return (float(np.mean(inside)), float(np.mean(width)), float(np.median(width)), float(np.mean(rel)), float(np.median(rel)))
    #            coverage          |  mean band width, GBP | median band width, GBP | mean width as % of price | median width as % of price


# =============================================================================
# SAVE - each winning interval model -> Outputs/interval-models/
# =============================================================================
def models_dir():
    d = os.path.join(os.path.dirname(CONFIG["interval_db"]), "interval-models")
    os.makedirs(d, exist_ok=True)
    return d


def save_model(interval_model, pre, Q, model, family, num, cat, run_id):
    """
    save the whole model to disk so we can reload and predict later without retraining.
    we save the preprocessor, the fitted models, Q and some metadata
    """
    payload = {
        **interval_model,
        "pre": pre,
        "Q": float(Q),
        "model": model,
        "family": family,
        "features": args.features,
        "numeric": num,
        "categorical": cat,
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "confidence": CONFIG["confidence"],
        "run_id": run_id,
    }
    path = os.path.join(models_dir(), f"interval_{run_id}_{model}_{args.features}.joblib")
    joblib.dump(payload, path)
    return path


def predict_intervals(payload, X_raw):
    """
    use a saved model on new houses. raw houses in GBP.
    not used in training. 
    this is for loading a saved model later and predicting.
    """
    Xp = payload["pre"].transform(X_raw)
    lo, hi = pred_lo_hi(payload, Xp)
    lo, hi = lo - payload["Q"], hi + payload["Q"]
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return np.expm1(lo), np.expm1(hi)


# =============================================================================
# DB
# =============================================================================
def _ensure_columns(conn, table, coldefs):
    """
    Added new columns and it made DB angry.
    TODO: This can be removed later 
    """
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in coldefs:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_db(db):
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, timestamp TEXT, features TEXT,
            confidence REAL, alpha REAL, q_low REAL, q_high REAL,
            n_trials INTEGER, full_rows INTEGER,
            n_train INTEGER, n_val INTEGER, n_calib INTEGER, n_test INTEGER,
            random_state INTEGER, n_features INTEGER, input_path TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS results (
            run_id TEXT, model TEXT, family TEXT, features TEXT, confidence REAL,
            coverage REAL, mean_width_gbp REAL, median_width_gbp REAL,
            mean_rel_width_pct REAL, median_rel_width_pct REAL,
            conformal_offset_log REAL,
            lower_params TEXT, upper_params TEXT,
            lower_pinball REAL, upper_pinball REAL,
            warm_start_run TEXT, model_path TEXT,
            tune_seconds REAL, timestamp TEXT,
            PRIMARY KEY (run_id, model),
            FOREIGN KEY (run_id) REFERENCES runs(run_id))""")
        # migrations for DBs created before these columns existed
        _ensure_columns(c, "runs", [("features", "TEXT")])
        _ensure_columns(c, "results", [("family", "TEXT"), ("features", "TEXT"),
                                       ("model_path", "TEXT")])


def log_run(db, run_id, sizes, num, cat):
    n_tr, n_val, n_cal, n_te = sizes
    row = {
        "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "features": args.features, "confidence": CONFIG["confidence"], "alpha": ALPHA,
        "q_low": Q_LOW, "q_high": Q_HIGH, "n_trials": CONFIG["n_trials"],
        "full_rows": CONFIG["full_rows"], "n_train": n_tr, "n_val": n_val,
        "n_calib": n_cal, "n_test": n_te, "random_state": RS,
        "n_features": len(num) + len(cat), "input_path": CONFIG["input"],
    }
    _insert(db, "runs", row)


def log_result(db, row):
    _insert(db, "results", row)


def _insert(db, table, row):
    with sqlite3.connect(db) as c:
        cols = ",".join(row.keys())
        c.execute(f"INSERT INTO {table} ({cols}) VALUES ({','.join('?'*len(row))})",
                  list(row.values()))


# =============================================================================
# RUN
# =============================================================================
def main():

    # --------------------------------------------------------------
    # SETUP - load, split, preprocess, open the DB
    # --------------------------------------------------------------
    (X_tr, y_tr, X_val, y_val, X_cal, y_cal, X_test, y_test), num, cat = load_and_split()
    print(f"Features: {args.features}  ({len(num)} numeric + {len(cat)} categorical)")
    print(f"Split -> train {len(X_tr):,} | val {len(X_val):,} | "f"calib {len(X_cal):,} | test {len(X_test):,}")
    print(f"Target coverage: {CONFIG['confidence']:.0%}  (quantiles {Q_LOW:.3f} / {Q_HIGH:.3f})")

    # clean the data once here - not inside every optuna trial
    # fit on train only, then apply to the other piles so we don't leak
    pre = make_preprocessor(num, cat)
    pre.fit(X_tr)
    Xtr, ytr = pre.transform(X_tr),  y_tr.to_numpy()
    Xval, yval = pre.transform(X_val), y_val.to_numpy()
    Xc, yc = pre.transform(X_cal),  y_cal.to_numpy()
    Xt, yt = pre.transform(X_test), y_test.to_numpy()

    run_id = uuid.uuid4().hex[:12]
    init_db(CONFIG["interval_db"])
    log_run(CONFIG["interval_db"], run_id, (len(X_tr), len(X_val), len(X_cal), len(X_test)), num, cat)

    for model in CONFIG["models"]:
        # --------------------------------------------------------------
        # PRE-TRAIN per model : pick family, grab any point params params
        # --------------------------------------------------------------
        family = FAMILY[model]

        # Warm-start is only when a matching point model exists. But not for linear cause those are different
        # and only from a point run with the same matched feature set ( since in args im using base vs full features flag)
        warm, warm_run = ({}, None)
        if family in ("booster", "qrf"):
            warm, warm_run = point_params(CONFIG["point_db"], model, len(num) + len(cat))

        print(f"\n{'='*55}\n{model}  [{family}]  (warm-start: {warm_run or 'none'})\n{'='*55}")

        # --------------------------------------------------------------
        # TRAIN  per model: tune + fit the interval model
        # --------------------------------------------------------------
        t0 = time.time()
        interval_model, lo_p, hi_p, lo_pin, hi_pin = train_interval(model, family, Xtr, ytr, Xval, yval, warm)
        tune_seconds = time.time() - t0

        # --------------------------------------------------------------
        # POST-TRAIN per model: conformalize, evaluate, save, log
        # --------------------------------------------------------------
        
        # Conformalize on calibration, then evaluate on test.
        lo_cal, hi_cal = pred_lo_hi(interval_model, Xc)
        Q = conformal_offset(lo_cal, hi_cal, yc, ALPHA)

        lo_t, hi_t = pred_lo_hi(interval_model, Xt)
        lo, hi = lo_t - Q, hi_t + Q
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)   # guard any crossing
        coverage, mean_w, med_w, mean_rel, med_rel = evaluate(lo, hi, yt)

        # Save the fitted interval model (self-contained).
        model_path = save_model(interval_model, pre, Q, model, family, num, cat, run_id)

        print(f"  Coverage {coverage:.3f} (target {CONFIG['confidence']:.2f}) | "
              f"mean width GBP {mean_w:,.0f} | median GBP {med_w:,.0f} | "
              f"rel {med_rel:.1f}% median / {mean_rel:.1f}% mean")
        print(f"  saved -> {model_path}  ({tune_seconds:.0f}s)")

        log_result(CONFIG["interval_db"], {
            "run_id": run_id, "model": model, "family": family,
            "features": args.features, "confidence": CONFIG["confidence"],
            "coverage": coverage, "mean_width_gbp": mean_w, "median_width_gbp": med_w,
            "mean_rel_width_pct": mean_rel, "median_rel_width_pct": med_rel,
            "conformal_offset_log": float(Q),
            "lower_params": json.dumps(lo_p), "upper_params": json.dumps(hi_p),
            "lower_pinball": float(lo_pin), "upper_pinball": float(hi_pin),
            "warm_start_run": warm_run, "model_path": model_path,
            "tune_seconds": round(tune_seconds, 1),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    print(f"\nLogged run {run_id} -> {CONFIG['interval_db']}  "f"({len(CONFIG['models'])} models)")
    print(f"Saved models -> {models_dir()}")


if __name__ == "__main__":
    main()