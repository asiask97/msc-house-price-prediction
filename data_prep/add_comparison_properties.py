"""
add_comparison_properties.py
----------------------------
Adds up to N_COMPARABLES nearest prior comparable sales to each property from
FIRST_YEAR onwards, using exact OpenUPRN coordinates only (postcode centroids
are not used).

For each property, comparables must:
    - Have exact coordinates.
    - Be from the previous RECENCY_MONTHS months.
    - Be from an earlier month than the target sale (prevents leakage).
    - Be among the N_COMPARABLES geographically nearest eligible sales.

Sales before FIRST_YEAR are used as comparables but excluded from the output.
Properties without exact coordinates stay in the output but get no comparables.

Input:  Outputs/lr_epc_coords.parquet
Output: Outputs/lr_epc_comparables.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
INPUT = Path("./Outputs/lr_epc_coords.parquet")
OUTPUT = Path("./Outputs/lr_epc_comparables.parquet")
FIRST_YEAR = 2020        # Rows from this year onwards appear in the final output.
N_COMPARABLES = 3        # Number of nearest comparable properties to add.
RECENCY_MONTHS = 12      # Maximum look-back period for comparable sales.
FULL_ROWS = None         # None = all rows; set an int to test on a sample.
R_KM = 6371.0088         # Earth radius, converts haversine distance to km.

# data is later cleaned to remove those extreme outliers. Im removing them here too 
POOL_PRICE_MIN, POOL_PRICE_MAX = 10_000, 5_000_000
POOL_AREA_MIN, POOL_AREA_MAX = 10, 1000


# =============================================================================
# VALIDATE INPUT / OUTPUT AND LOAD DATA
# =============================================================================
print("Validating files...")
if not INPUT.exists():
    raise FileNotFoundError(f"\nInput parquet not found:\n  {INPUT}\n\nExpected output from add_exact_coordinates.py.")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
 
print(f"\nLoading parquet: {INPUT}")
df = pd.read_parquet(INPUT)
print(f"Loaded {len(df):,} rows | {len(df.columns):,} columns")
 
# =============================================================================
# CLEAN DATA TYPES
# =============================================================================
print("\nCleaning data types...")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["total_floor_area"] = pd.to_numeric(df["total_floor_area"], errors="coerce")
df["exact_lat"] = pd.to_numeric(df["exact_lat"], errors="coerce")
df["exact_lon"] = pd.to_numeric(df["exact_lon"], errors="coerce")
 
# Rows without a valid sale date cannot be used as targets or comparables.
before = len(df)
df = df[df["date"].notna()].copy()
print(f"Dropped {before - len(df):,} rows with invalid or missing dates")
 
# =============================================================================
# CHECK EXACT COORDINATE COVERAGE
# =============================================================================
print("\nChecking exact coordinate coverage...")
exact_ok = np.isfinite(df["exact_lat"]) & np.isfinite(df["exact_lon"])
print(f"Exact coordinates available: {exact_ok.sum():,} / {len(df):,} ({exact_ok.mean() * 100:.1f}%)")
print(f"No exact coordinates: {(~exact_ok).sum():,} / {len(df):,} ({(~exact_ok).mean() * 100:.1f}%)")
 
# =============================================================================
# REPORT DATE RANGE
# =============================================================================
yrs = df["date"].dt.year
print(f"\nDate range: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Years present: {int(yrs.min())} -> {int(yrs.max())}")
if yrs.min() > FIRST_YEAR - 1:
    print(f"\nWARNING: No rows before {FIRST_YEAR} were found; early {FIRST_YEAR} rows will have fewer comparables.")
 
# =============================================================================
# BUILD COMPARABLE SEARCH POOL
# =============================================================================
print("\nBuilding comparable search pool...")
# The pool includes ALL years (the RECENCY_MONTHS window handles recency).
# Only rows with exact coordinates can be selected as comparables.
pool = df[["date", "price", "total_floor_area", "exact_lat", "exact_lon"]].copy()
pool = pool.dropna(subset=["date", "price", "exact_lat", "exact_lon"])
 
before_pool = len(pool)
pool = pool[(pool["price"] >= POOL_PRICE_MIN) & (pool["price"] <= POOL_PRICE_MAX)].copy()
junk_price_rows = before_pool - len(pool)
 
# Invalid floor areas don't drop the sale, only null its floor-area feature.
bad_area = (pool["total_floor_area"] < POOL_AREA_MIN) | (pool["total_floor_area"] > POOL_AREA_MAX)
pool.loc[bad_area, "total_floor_area"] = np.nan
pool = pool.reset_index(drop=True)
 
print(f"Comparable pool rows: {len(pool):,}")
print(f"Dropped junk-priced sales: {junk_price_rows:,}")
print(f"Nulled invalid floor areas: {int(bad_area.sum()):,}")
 
# =============================================================================
# BUILD QUERY SET
# =============================================================================
print(f"\nBuilding final output query set ({FIRST_YEAR}+ only)...")
# Making sure 2019 is only used as comprison and not used in final data
query = df[df["date"].dt.year >= FIRST_YEAR].copy()
history_rows = int((df["date"].dt.year < FIRST_YEAR).sum())
 
if FULL_ROWS is not None and FULL_ROWS < len(query):
    query = query.sample(n=FULL_ROWS, random_state=42)
    print(f"Using sample of {FULL_ROWS:,} query rows")
 
query = query.reset_index(drop=True)
print(f"Final output rows: {len(query):,}")
print(f"History-only rows (before {FIRST_YEAR}): {history_rows:,}")
if query.empty:
    raise SystemExit(f"ERROR: No rows from {FIRST_YEAR} onwards found.")
 
# =============================================================================
# PREPARE NUMPY ARRAYS
# =============================================================================
pool_date = pool["date"].values.astype("datetime64[ns]")
pool_price = pool["price"].values.astype(float)
pool_area = pool["total_floor_area"].values.astype(float)
pool_coords_rad = np.deg2rad(pool[["exact_lat", "exact_lon"]].values.astype(float))
 
q_month = query["date"].values.astype("datetime64[M]")
q_date = query["date"].values.astype("datetime64[ns]")
q_lat = query["exact_lat"].values.astype(float)
q_lon = query["exact_lon"].values.astype(float)
q_ok = np.isfinite(q_lat) & np.isfinite(q_lon)   # only exact-coord targets get comparables
 
# =============================================================================
# INITIALISE OUTPUT ARRAYS
# =============================================================================
# make empty boxes to fill. comp_price starts as a wall of NaN then its overwriten.
K = N_COMPARABLES
n_q = len(query)
comp_price = np.full((n_q, K), np.nan, dtype=float)
comp_area = np.full((n_q, K), np.nan, dtype=float)
comp_dist = np.full((n_q, K), np.nan, dtype=float)
comp_days = np.full((n_q, K), np.nan, dtype=float)
 
# =============================================================================
# CHRONOLOGICAL COMPARABLE SEARCH
# =============================================================================
print(f"\nSearching for {N_COMPARABLES} nearest prior comparable properties...")
unique_months = np.unique(q_month)
 
for qm in tqdm(unique_months, desc="Processing months"):
    # Targets in this month that have exact coordinates.
    q_mask = (q_month == qm) & q_ok
    if not q_mask.any():
        continue
 
    # Candidate window: Get sales from the past year to caompre to.
    # Keep whole month, and do day cut later
    window_start = np.datetime64(pd.Timestamp(qm) - pd.DateOffset(months=RECENCY_MONTHS))
    month_end = np.datetime64(pd.Timestamp(qm) + pd.DateOffset(months=1))
    cand_mask = (pool_date >= window_start) & (pool_date < month_end)
    if not cand_mask.any():
        continue
 
    cand_idx = np.where(cand_mask)[0]
    tree = BallTree(pool_coords_rad[cand_idx], metric="haversine")
 
    q_idx = np.where(q_mask)[0]
    query_coords_rad = np.deg2rad(np.column_stack([q_lat[q_idx], q_lon[q_idx]]))
 
    # Over-fetch: get nearest few properties and only use 3 best ones
    k_search = min(len(cand_idx), K + 25)
    distances_rad, neighbour_idx = tree.query(query_coords_rad, k=k_search)
 
    real_idx = cand_idx[neighbour_idx]
    dist_km = distances_rad * R_KM
    days = (q_date[q_idx][:, None] - pool_date[real_idx]) / np.timedelta64(1, "D")
 
    # Keep only comparables strictly before the target's day; push the rest to
    # the back (inf distance) and re-sort so the nearest valid ones come first.
    valid = days > 0
    dist_km = np.where(valid, dist_km, np.inf)
    order = np.argsort(dist_km, axis=1, kind="stable")
    dist_km = np.take_along_axis(dist_km, order, axis=1)
    real_idx = np.take_along_axis(real_idx, order, axis=1)
    days = np.take_along_axis(np.where(valid, days, np.nan), order, axis=1)
 
    # Take the K nearest valid; an inf distance means no valid comparable there.
    n_slots = min(K, dist_km.shape[1])
    dist_km = dist_km[:, :n_slots]
    real_idx = real_idx[:, :n_slots]
    days = days[:, :n_slots]
    keep = np.isfinite(dist_km)
 
    comp_price[q_idx, :n_slots] = np.where(keep, pool_price[real_idx], np.nan)
    comp_area[q_idx, :n_slots] = np.where(keep, pool_area[real_idx], np.nan)
    comp_dist[q_idx, :n_slots] = np.where(keep, dist_km, np.nan)
    comp_days[q_idx, :n_slots] = np.where(keep, days, np.nan)
 
# =============================================================================
# ADD COMPARABLE FEATURES TO FINAL DATASET
# =============================================================================
print("\nAdding comparable features to output...")
result = query.copy()
for j in range(K):
    n = j + 1
    result[f"comp{n}_price"] = comp_price[:, j]
    result[f"comp{n}_floor_area"] = comp_area[:, j]
    result[f"comp{n}_distance_km"] = comp_dist[:, j]
    result[f"comp{n}_days_ago"] = comp_days[:, j]
 
result["n_comparables"] = np.isfinite(comp_price).sum(axis=1)   # finite price = real comparable
result["has_comparables"] = result["n_comparables"] > 0
 
# =============================================================================
# REPORT RESULTS
# =============================================================================
cold = result["n_comparables"] == 0
print("\n--- RESULTS ---")
print(f"Output rows: {len(result):,}")
print(f"Rows with no comparables: {cold.sum():,} / {len(result):,} ({cold.mean() * 100:.2f}%)")
print("\nComparable count distribution:")
print(result["n_comparables"].value_counts().sort_index())
 
# =============================================================================
# LEAKAGE CHECK
# =============================================================================
days_cols = [f"comp{i}_days_ago" for i in range(1, K + 1)]
bad = int((result[days_cols] <= 0).sum().sum())   # every comparable must predate its target
print(f"\nLeakage check - comparables dated on/after target sale: {bad:,} {'PASS' if bad == 0 else 'INVESTIGATE'}")
if bad > 0:
    raise RuntimeError("Leakage check failed: a comparable is dated on or after its target sale.")
 
# =============================================================================
# SAVE FINAL PARQUET
# =============================================================================
print(f"\nSaving final parquet to: {OUTPUT}")
result.to_parquet(OUTPUT, index=False, compression="snappy")
print(f"\nDone. Saved {len(result):,} rows to: {OUTPUT}")