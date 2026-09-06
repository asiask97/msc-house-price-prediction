"""
test_best_model.py

Trains one xgboost model based on best params i have found during traning but on whole dataset.
Just a test script, can be removed later.

Run:  python test_best_model.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
)
import time, warnings
warnings.filterwarnings("ignore")

INPUT       = "./Outputs/clean_property_data.parquet"
FULL_ROWS   = 4_291_959
TEST_SIZE   = 0.2
RS          = 42
TARGET      = "price"

BEST_PARAMS = {
    "learning_rate":    0.022447348973403,
    "max_depth":        10,
    "min_child_weight": 3,
    "subsample":        0.7123849746229174,
    "colsample_bytree": 0.6854650581984649,
    "reg_lambda":       0.3591999721183231,
    "reg_alpha":        0.015132818879491708,
    "gamma":             0.0016357097409610431,
}
N_ESTIMATORS = 2000

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

print("Loading...")
df = pd.read_parquet(INPUT)
df = df.sample(n=min(FULL_ROWS, len(df)), random_state=RS)
print(f"  {len(df):,} rows")

numeric_features     = [f for f in numeric_features if f in df.columns]
categorical_features = [f for f in categorical_features if f in df.columns]
X = df[numeric_features + categorical_features]
y = np.log1p(df[TARGET])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RS)
print(f"  train {len(X_tr):,} | test {len(X_te):,}")

pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
     categorical_features),
])

model = XGBRegressor(
    random_state=RS, n_jobs=-1, tree_method="hist",
    # device="cuda",          # <- uncomment if you have a GPU
    n_estimators=N_ESTIMATORS, **BEST_PARAMS,
)
pipe = Pipeline([("pre", pre), ("reg", model)])

print("Fitting...")
t0 = time.time()
pipe.fit(X_tr, y_tr)
print(f"  done in {time.time()-t0:.0f}s")

pr_tr = np.expm1(pipe.predict(X_tr))
pr_te = np.expm1(pipe.predict(X_te))
yt_tr, yt_te = np.expm1(y_tr), np.expm1(y_te)

def show(split, yt, pr):
    print(f"  {split:5} | MAPE {mean_absolute_percentage_error(yt, pr)*100:6.2f}% "
          f"| RMSE £{np.sqrt(mean_squared_error(yt, pr)):>10,.0f} "
          f"| MAE £{mean_absolute_error(yt, pr):>9,.0f} "
          f"| R² {r2_score(yt, pr):.4f}")

print("\nResults:")
show("train", yt_tr, pr_tr)
show("test",  yt_te, pr_te)