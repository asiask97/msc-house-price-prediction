"""
shap_analysis.py

Pull each model's BEST params (lowest test_mape) from experiments.db, refit on the full data, and run fast/reliable SHAP feature importance.

- TreeExplainer (exact for trees) + a small row SAMPLE  -> fast.
- Explains the TRANSFORMED matrix, groups one-hot cols back to features -> right.
- SHAP values are in LOG-price space (target is log1p) -> use for RANKING only.

Run:  python shap_analysis.py
"""

import json
import sqlite3
import warnings

import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
INPUT        = "./Outputs/clean_property_data.parquet"
RESULTS_DB   = "./Outputs/experiments.db"
FULL_ROWS    = 4_290_000
TEST_SIZE    = 0.2
RS           = 42
SHAP_SAMPLE  = 3_000
MODELS       = ["XGBoost", "LightGBM", "RandomForest"]
OUT_CSV      = "./Outputs/shap_importance.csv"

NUMERIC_FEATURES = [
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
CATEGORICAL_FEATURES = [
    "property_type_x", "new_build", "duration", "current_energy_rating",
    "tenure", "built_form", "construction_age_band", "main_fuel",
]
TARGET = "price"


# =============================================================================
# HELPERS
# =============================================================================
def load_xy():
    df = pd.read_parquet(INPUT)
    df = df.sample(n=min(FULL_ROWS, len(df)), random_state=RS)
    num = [f for f in NUMERIC_FEATURES if f in df.columns]
    cat = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    X = df[num + cat]
    y = np.log1p(df[TARGET])
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RS), num, cat


def best_params(db, model):
    """ Gets the best hyper params from DB - Lowest-test_mape row for one model. Returns (params_dict, run_id, test_mape)."""
    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT best_params, run_id, test_mape FROM results "
        "WHERE model=? AND test_mape IS NOT NULL ORDER BY test_mape ASC LIMIT 1",
        (model,)).fetchone()
    con.close()
    if row is None:
        return None
    return json.loads(row[0]), row[1], row[2]


def make_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         cat),
    ])


def build_model(name, params, num, cat):
    """Mirror train.py's refit: boosters get n_estimators=2000 added; RF keeps its own."""
    p = dict(params)
    if name == "RandomForest":
        reg = RandomForestRegressor(random_state=RS, n_jobs=-1, **p)
    elif name == "LightGBM":
        p.setdefault("n_estimators", 2000)
        reg = LGBMRegressor(random_state=RS, n_jobs=-1, verbose=-1, **p)
    elif name == "XGBoost":
        p.setdefault("n_estimators", 2000)
        reg = XGBRegressor(random_state=RS, n_jobs=-1, tree_method="hist",  device="cuda", **p) 
    else:
        raise ValueError(name)
    return Pipeline([("pre", make_preprocessor(num, cat)), ("reg", reg)])


def shap_importance(pipe, X, num, cat, title):
    """SHAP on a row sample; returns per-original-feature mean|SHAP| (log-price space)."""
    pre, reg = pipe.named_steps["pre"], pipe.named_steps["reg"]
    Xs = X.sample(n=min(SHAP_SAMPLE, len(X)), random_state=RS)
    Xt = pre.transform(Xs)
    out_names = pre.get_feature_names_out()

    sv = shap.TreeExplainer(reg).shap_values(Xt)
    per_col = np.abs(sv).mean(axis=0)

    # group one-hot encoded columns back together
    cats_by_len = sorted(cat, key=len, reverse=True)
    grouped = {}
    for col, val in zip(out_names, per_col):
        if col.startswith("num__"):
            src = col[5:]
        elif col.startswith("cat__"):
            body = col[5:]
            src = next((c for c in cats_by_len if body == c or body.startswith(c + "_")), body)
        else:
            src = col
        grouped[src] = grouped.get(src, 0.0) + val

    return (pd.DataFrame({"model": title, "feature": list(grouped),
                          "mean_abs_shap": list(grouped.values())})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True))


# =============================================================================
# RUN
# =============================================================================
def main():
    (X_tr, X_te, y_tr, y_te), num, cat = load_xy()
    print(f"Loaded {len(X_tr) + len(X_te):,} rows | {len(num)} num + {len(cat)} cat features")

    all_imp = []
    for name in MODELS:
        bp = best_params(RESULTS_DB, name)
        if bp is None:
            print(f"\n[{name}] no rows in {RESULTS_DB} - skipping")
            continue
        params, run_id, tmape = bp
        print(f"\n=== {name} === best from run {run_id} (test_mape {tmape:.2f})")

        pipe = build_model(name, params, num, cat)
        print("  training..."); pipe.fit(X_tr, y_tr)
        print("  explaining...")
        imp = shap_importance(pipe, X_te, num, cat, name)
        print(imp.to_string(index=False))
        all_imp.append(imp)

    if all_imp:
        combined = pd.concat(all_imp, ignore_index=True)
        combined.to_csv(OUT_CSV, index=False)
        print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()