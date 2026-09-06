"""
add_exact_coordinates.py
------------------------
Enriches the matched LR-EPC dataset with precise per-property coordinates
from OpenUPRN, where a UPRN is available in the EPC data.

For properties without a UPRN match, coordinates fall back to the
postcode centroid from the ONS Postcode Directory (already in the dataset).

Usage:
    python add_exact_coordinates.py

Requirements:
    - Outputs/lr_epc_matched.parquet  (output of joining_epc_and_lr.py)
    - Data/OpenUPRN.csv            (downloaded from osdatahub.os.uk)
    - Data/Online_ONS_Postcode_Directory_Live.csv

Output:
    Outputs/lr_epc_coords.parquet
"""

import pandas as pd
from tqdm import tqdm

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

print("Loading matched LR-EPC dataset...")
df = pd.read_parquet("./Outputs/lr_epc_matched.parquet")
print(f"Loaded {len(df):,} records")

print("\nLoading OpenUPRN...")
df_uprn = pd.read_csv(
    "./Data/osopenuprn_202605.csv",
    usecols=["UPRN", "LATITUDE", "LONGITUDE"]
)
df_uprn["UPRN"] = df_uprn["UPRN"].astype(str).str.strip()
print(f"Loaded {len(df_uprn):,} UPRN records")

print("\nLoading ONS Postcode Directory...")
postcodes = pd.read_csv(
    "./Data/Online_ONS_Postcode_Directory_Live.csv",
    usecols=["PCDS", "LAT", "LONG", "IMD20IND"]
)
postcodes["PCDS"] = postcodes["PCDS"].str.strip().str.upper()
print(f"Loaded {len(postcodes):,} postcode records")


# =============================================================================
# SECTION 2: ADD POSTCODE CENTROID COORDINATES
# =============================================================================
print("\nAdding postcode centroid coordinates...")
df["postcode_merge"] = df["postcode_x"].str.strip().str.upper()
df = df.merge(postcodes, left_on="postcode_merge", right_on="PCDS", how="left")

# Remove ONSPD placeholder values (no grid reference)
df.loc[df["LAT"] == 99.999999, "LAT"] = None
df.loc[df["LONG"] == 0.000000, "LONG"] = None

centroid_coverage = df["LAT"].notna().sum()
print(f"Postcode centroid coverage: {centroid_coverage:,} ({centroid_coverage/len(df)*100:.1f}%)")


# =============================================================================
# SECTION 3: ADD EXACT COORDINATES FROM OPENUPRN
# =============================================================================

print("\nAdding exact coordinates from OpenUPRN...")

# Convert both to int64 to avoid float/int mismatch
df["uprn_int"] = pd.to_numeric(df["uprn"], errors="coerce").astype("Int64")
df_uprn["UPRN"] = df_uprn["UPRN"].astype("Int64")

# Merge on UPRN
df = df.merge(
    df_uprn.rename(columns={"LATITUDE": "exact_lat", "LONGITUDE": "exact_lon"}),
    left_on="uprn_int",
    right_on="UPRN",
    how="left"
)

# Drop redundant columns
df = df.drop(columns=["UPRN", "uprn_int"], errors="ignore")

exact_coverage = df["exact_lat"].notna().sum()
print(f"Exact coordinate coverage: {exact_coverage:,} ({exact_coverage/len(df)*100:.1f}%)")


# =============================================================================
# SECTION 4: ADD FLAG AND SUMMARY
# =============================================================================

# Flag whether exact or centroid coordinates are available
df["has_exact_coords"] = df["exact_lat"].notna()

print(f"\n--- Coordinate Coverage ---")
print(f"Total records:              {len(df):,}")
print(f"Exact coords (OpenUPRN):    {df['has_exact_coords'].sum():,} ({df['has_exact_coords'].sum()/len(df)*100:.1f}%)")
print(f"Centroid only (ONSPD):      {(~df['has_exact_coords'] & df['LAT'].notna()).sum():,}")
print(f"No coordinates at all:      {(df['LAT'].isna() & df['exact_lat'].isna()).sum():,}")


# =============================================================================
# SECTION 5: SAVE
# =============================================================================

print("\nSaving to parquet...")
df.to_parquet("./Outputs/lr_epc_coords.parquet", compression="snappy")
print("Done — saved to ./Outputs/lr_epc_coords.parquet")