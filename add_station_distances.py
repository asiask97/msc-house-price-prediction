"""
add_station_distances.py
------------------------
Adds distance to nearest transport stops and density counts for each
property, split by transport type.

Usage:
    python add_station_distances.py

Input:
    - Data/transport/stops.csv
    - Outputs/lr_epc_schools.parquet

Output:
    - Outputs/lr_epc_stations.parquet
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


# =============================================================================
# SECTION 1: LOAD AND FILTER STATIONS
# =============================================================================

print("Loading transport stops data...")
raw = pd.read_csv("Data/transport/stops.csv", low_memory=False)
print(f"  Total stops: {len(raw):,}")

# Active only, must have coordinates
raw = raw[raw["Status"] == "active"]
raw = raw.dropna(subset=["Latitude", "Longitude"])

# Split by type
rail = raw[raw["StopType"] == "RLY"].drop_duplicates(subset=["CommonName", "NptgLocalityCode"])
metro = raw[raw["StopType"] == "MET"].drop_duplicates(subset=["CommonName", "NptgLocalityCode"])
bus = raw[raw["StopType"] == "BCT"].drop_duplicates(subset=["CommonName", "NptgLocalityCode"])
airport = raw[raw["StopType"] == "AIR"].drop_duplicates(subset=["CommonName", "NptgLocalityCode"])

print(f"  Rail stations:  {len(rail):,}")
print(f"  Metro stations: {len(metro):,}")
print(f"  Bus stops:      {len(bus):,}")
print(f"  Airports:       {len(airport):,}")


# =============================================================================
# SECTION 2: LOAD PROPERTIES
# =============================================================================

print("\nLoading property dataset...")
df = pd.read_parquet("Outputs/lr_epc_schools.parquet")
print(f"  Records: {len(df):,}")

df["lat"] = df["exact_lat"].fillna(df["LAT"])
df["lon"] = df["exact_lon"].fillna(df["LONG"])
mask = df["lat"].notna() & df["lon"].notna() #only rows with coordinates for distance calculations
print(f"  With coordinates: {mask.sum():,}")

prop_coords = np.deg2rad(df.loc[mask, ["lat", "lon"]].values)


# =============================================================================
# SECTION 3: COMPUTE DISTANCES AND COUNTS
# =============================================================================

def build_tree(target_df):
    coords = np.deg2rad(target_df[["Latitude", "Longitude"]].values)
    return BallTree(coords, metric="haversine")

def nearest_dist(tree, prop_coords):
    dist, _ = tree.query(prop_coords, k=1)
    return dist.flatten() * 6371.0

def count_within(tree, prop_coords, km):
    radius = km / 6371.0
    return tree.query_radius(prop_coords, r=radius, count_only=True)


# --- Rail ---
print("\nProcessing rail stations...")
tree_rail = build_tree(rail)
df.loc[mask, "dist_rail_km"] = nearest_dist(tree_rail, prop_coords)
df.loc[mask, "rail_within_1km"] = count_within(tree_rail, prop_coords, 1)
df.loc[mask, "rail_within_5km"] = count_within(tree_rail, prop_coords, 5)

# --- Metro ---
if len(metro) > 0:
    print("Processing metro stations...")
    tree_metro = build_tree(metro)
    df.loc[mask, "dist_metro_km"] = nearest_dist(tree_metro, prop_coords)
    df.loc[mask, "metro_within_1km"] = count_within(tree_metro, prop_coords, 1)
else:
    print("No metro stations found — skipping")
    df["dist_metro_km"] = np.nan
    df["metro_within_1km"] = 0

# --- Bus ---
print("Processing bus stops (this may take a few minutes)...")
tree_bus = build_tree(bus)
df.loc[mask, "dist_bus_km"] = nearest_dist(tree_bus, prop_coords)
df.loc[mask, "bus_within_1km"] = count_within(tree_bus, prop_coords, 1)

# --- Airport ---
if len(airport) > 0:
    print("Processing airports...")
    tree_airport = build_tree(airport)
    df.loc[mask, "dist_airport_km"] = nearest_dist(tree_airport, prop_coords)
else:
    print("No airports found — skipping")
    df["dist_airport_km"] = np.nan


# =============================================================================
# SECTION 4: SUMMARY AND SAVE
# =============================================================================

print(f"\n--- Distance Summary (km) ---")
for col, label in [
    ("dist_rail_km", "Rail"),
    ("dist_metro_km", "Metro"),
    ("dist_bus_km", "Bus"),
    ("dist_airport_km", "Airport"),
]:
    if df[col].notna().any():
        print(f"  {label:8s}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, max={df[col].max():.2f}")

print(f"\n--- Count Summary ---")
for col, label in [
    ("rail_within_1km", "Rail <1km"),
    ("rail_within_5km", "Rail <5km"),
    ("metro_within_1km", "Metro <1km"),
    ("bus_within_1km", "Bus <1km"),
]:
    print(f"  {label:12s}: mean={df[col].mean():.2f}, max={df[col].max():.0f}")

df = df.drop(columns=["lat", "lon"], errors="ignore")

print("\nSaving...")
df.to_parquet("Outputs/lr_epc_stations.parquet", compression="snappy")
print("Done — saved to Outputs/lr_epc_stations.parquet")