"""
add_school_distances.py
-----------------------
Adds distance to nearest primary and secondary school for each property.

Two separate school datasets are needed because England and Wales have
different education systems and different data sources:
  - England: GIAS (Get Information About Schools) from DfE
  - Wales: DataMapWales maintained schools dataset from Welsh Government

This script loads both, filters to open primary and secondary schools,
standardizes the columns, merges them, then computes the distance from
each property to its nearest primary and nearest secondary school.

Both use British National Grid coordinates (Easting/Northing) which are
converted to lat/long before distance calculation. Wales labels are in
Welsh (Cynradd = Primary, Uwchradd = Secondary).

Usage:
    python add_school_distances.py

Requirements:
    - Data/schools/edubasealldata20260707.csv    (GIAS all establishments download)
    - Data/schools/maintained_schools_wg.csv     (DataMapWales maintained schools)
    - Outputs/lr_epc_coords.parquet                 (output of add_exact_coordinates.py)

Output:
    Outputs/lr_epc_schools.parquet
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.neighbors import BallTree

# Shared coordinate converter: British National Grid → WGS84 lat/long
transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


# =============================================================================
# SECTION 1: ENGLAND SCHOOLS (GIAS)
# =============================================================================

print("Loading England schools (GIAS)...")
eng = pd.read_csv("Data/schools/edubasealldata20260707.csv", encoding="latin-1", low_memory=False)

eng = eng[eng["EstablishmentStatus (name)"] == "Open"]
eng = eng[eng["PhaseOfEducation (name)"].isin(["Primary", "Secondary"])]
no_coords_eng = eng[eng["Easting"].isna() | eng["Northing"].isna()].shape[0]
eng = eng.dropna(subset=["Easting", "Northing"])

print(f"  Open Primary + Secondary: {len(eng) + no_coords_eng:,}")
print(f"  With coordinates:         {len(eng):,}")
print(f"  Missing coordinates:      {no_coords_eng:,}")

lon, lat = transformer.transform(eng["Easting"].values, eng["Northing"].values)

eng_clean = pd.DataFrame({
    "name": eng["EstablishmentName"].values,
    "phase": eng["PhaseOfEducation (name)"].values,
    "latitude": lat,
    "longitude": lon,
})


# =============================================================================
# SECTION 2: WALES SCHOOLS (DataMapWales)
# =============================================================================

print("\nLoading Wales schools (DataMapWales)...")
wal = pd.read_csv("Data/schools/maintained_schools_wg.csv")

wal = wal[wal["sector"].isin(["Cynradd", "Uwchradd"])]
no_coords_wal = wal[wal["geom"].isna()].shape[0]
wal = wal.dropna(subset=["geom"])

print(f"  Primary + Secondary:      {len(wal) + no_coords_wal:,}")
print(f"  With coordinates:         {len(wal):,}")
print(f"  Missing coordinates:      {no_coords_wal:,}")

# geom is WKT with Easting/Northing: "POINT (244071.99 393130.99)"
coords = wal["geom"].str.extract(r"POINT\s*\(\s*([0-9.\-]+)\s+([0-9.\-]+)\s*\)")
easting = coords[0].astype(float).values
northing = coords[1].astype(float).values
lon, lat = transformer.transform(easting, northing)

wal_clean = pd.DataFrame({
    "name": wal["school_name"].values,
    "phase": wal["sector"].map({"Cynradd": "Primary", "Uwchradd": "Secondary"}).values,
    "latitude": lat,
    "longitude": lon,
})


# =============================================================================
# SECTION 3: COMBINE AND SPLIT BY PHASE
# =============================================================================

print("\nCombining...")
schools = pd.concat([eng_clean, wal_clean], ignore_index=True)

primary = schools[schools["phase"] == "Primary"].reset_index(drop=True)
secondary = schools[schools["phase"] == "Secondary"].reset_index(drop=True)
print(f"  Primary schools:   {len(primary):,}")
print(f"  Secondary schools: {len(secondary):,}")


# =============================================================================
# SECTION 4: LOAD PROPERTIES
# =============================================================================

print("\nLoading property dataset...")
df = pd.read_parquet("Outputs/lr_epc_comparables.parquet")
print(f"  Records: {len(df):,}")

# Use exact coords where available, fall back to postcode centroid
df["lat"] = df["exact_lat"].fillna(df["LAT"])
df["lon"] = df["exact_lon"].fillna(df["LONG"])
mask = df["lat"].notna() & df["lon"].notna()
print(f"  With coordinates: {mask.sum():,}")


# =============================================================================
# SECTION 5: COMPUTE DISTANCES
# =============================================================================

def nearest_km(prop_lat, prop_lon, target_df):
    """Find distance in km from each property to nearest target using BallTree."""
    target_coords = np.deg2rad(target_df[["latitude", "longitude"]].values)
    prop_coords = np.deg2rad(np.column_stack([prop_lat.values, prop_lon.values]))

    tree = BallTree(target_coords, metric="haversine")
    dist, _ = tree.query(prop_coords, k=1)
    return dist.flatten() * 6371.0  # radians → km


print("\nComputing distance to nearest primary school...")
df.loc[mask, "dist_primary_km"] = nearest_km(df.loc[mask, "lat"], df.loc[mask, "lon"], primary)

print("Computing distance to nearest secondary school...")
df.loc[mask, "dist_secondary_km"] = nearest_km(df.loc[mask, "lat"], df.loc[mask, "lon"], secondary)


# =============================================================================
# SECTION 6: SUMMARY AND SAVE
# =============================================================================

print(f"\n--- Distance Summary ---")
for col, label in [("dist_primary_km", "Primary"), ("dist_secondary_km", "Secondary")]:
    print(f"{label}: mean={df[col].mean():.2f}km, median={df[col].median():.2f}km, max={df[col].max():.2f}km")

df = df.drop(columns=["lat", "lon"], errors="ignore")

print("\nSaving...")
df.to_parquet("Outputs/lr_epc_schools.parquet", compression="snappy")
print("Done — saved to Outputs/lr_epc_schools.parquet")