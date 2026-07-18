"""
add_town_distance.py
--------------------
Adds distance to nearest major town or city for each property.
 
Major Towns and Cities (December 2015) dataset contains 112 towns/cities in England and Wales 
with a population above 75,000. Coordinates are the centroids of each town boundary.
 
Only 112 towns are included so smaller towns and villages are not included.
 
Usage:
    python add_town_distance.py
 
Input:
    - Data/cities/Major_Towns_and_Cities_Dec_2015_Boundaries_V2_2022.csv
    - Outputs/lr_epc_coast.parquet
 
Output:
    - Outputs/lr_epc_towns.parquet
"""
 
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
 
 
# =============================================================================
# SECTION 1: LOAD TOWNS
# =============================================================================
 
print("Loading Major Towns and Cities...")
towns = pd.read_csv("Data/cities/Major_Towns_and_Cities_Dec_2015_Boundaries_V2_2022.csv")
towns = towns.dropna(subset=["LAT", "LONG"])
print(f"  Towns: {len(towns):,}")
print(f"  Examples: {towns['TCITY15NM'].head(5).tolist()}")
 
 
# =============================================================================
# SECTION 2: LOAD PROPERTIES
# =============================================================================
 
print("\nLoading property dataset...")
df = pd.read_parquet("Outputs/lr_epc_coast.parquet")
print(f"  Records: {len(df):,}")
 
df["lat"] = df["exact_lat"].fillna(df["LAT"])
df["lon"] = df["exact_lon"].fillna(df["LONG"])
mask = df["lat"].notna() & df["lon"].notna()
print(f"  With coordinates: {mask.sum():,}")
 
 
# =============================================================================
# SECTION 3: COMPUTE DISTANCES
# =============================================================================
 
print("\nComputing distance to nearest town...")
town_coords = np.deg2rad(towns[["LAT", "LONG"]].values)
prop_coords = np.deg2rad(df.loc[mask, ["lat", "lon"]].values)
 
tree = BallTree(town_coords, metric="haversine")
dist, idx = tree.query(prop_coords, k=1)
 
df.loc[mask, "dist_town_km"] = dist.flatten() * 6371.0
df.loc[mask, "nearest_town"] = towns["TCITY15NM"].values[idx.flatten()]
 
 
# =============================================================================
# SECTION 4: SUMMARY AND SAVE
# =============================================================================
 
col = "dist_town_km"
print(f"\n--- Town Distance Summary ---")
print(f"  Mean:   {df[col].mean():.2f} km")
print(f"  Median: {df[col].median():.2f} km")
print(f"  Max:    {df[col].max():.2f} km")
print(f"  < 5km:  {(df[col] < 5).sum():,} ({(df[col] < 5).mean()*100:.1f}%)")
print(f"  < 20km: {(df[col] < 20).sum():,} ({(df[col] < 20).mean()*100:.1f}%)")
 
print(f"\n--- Most Common Nearest Towns ---")
print(df["nearest_town"].value_counts().head(10).to_string())
 
df = df.drop(columns=["lat", "lon"], errors="ignore")
 
print("\nSaving...")
df.to_parquet("Outputs/lr_epc_towns.parquet", compression="snappy")
print("Done — saved to Outputs/lr_epc_towns.parquet")
 