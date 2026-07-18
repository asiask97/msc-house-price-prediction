"""
add_coast_distance.py
---------------------
Adds distance to nearest coastline for each property.

Coastline data from the Boundary Line (high_water_polyline) gives the
Mean High Water mark around UK.

The coastline is a line, not a set of points, so we sample points
along it every ~500m to create a dense points, then use BallTree
to find the nearest coastal point for each property.

Each point on the coastline is the position where sea neets land at average high tide.

Source: https://osdatahub.os.uk/downloads/open/BoundaryLine

Usage:
    python add_coast_distance.py

Input:
    - Data/coastline/Data/GB/high_water_polyline.shp
    - Outputs/lr_epc_stations.parquet

Output:
    - Outputs/lr_epc_coast.parquet
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: LOAD AND SAMPLE COASTLINE
# =============================================================================

print("Loading coastline shapefile...")
coast = gpd.read_file("Data/coastline/Data/GB/high_water_polyline.shp")
print(f"  Coastline segments: {len(coast):,}")

# Sample points along the coastline every 500m
# The shapefile uses British National Grid (meters), so 500 = 500m
print("Sampling points along coastline every 500m...")
points = []
for geom in coast.geometry:
    if geom is None:
        continue
    length = geom.length
    n_points = max(int(length / 500), 1)
    for i in range(n_points + 1):
        point = geom.interpolate(i * 500)
        points.append((point.x, point.y))

print(f"  Sampled {len(points):,} coastal points")

# Convert from British National Grid to lat/long
print("Converting to lat/long...")
transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
eastings = [p[0] for p in points]
northings = [p[1] for p in points]
lon, lat = transformer.transform(eastings, northings)

coast_coords = np.column_stack([lat, lon])
print(f"  Coastal points with coordinates: {len(coast_coords):,}")

plt.figure(figsize=(8, 12))
plt.scatter(lon, lat, s=0.1, alpha=0.5, color="#2E86C1")
plt.title(f"Sampled Coastal Points ({len(lat):,})")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.savefig("coastal_points_check.png", dpi=150)
plt.show()

# =============================================================================
# SECTION 2: LOAD PROPERTIES
# =============================================================================

print("\nLoading property dataset...")
df = pd.read_parquet("Outputs/lr_epc_stations.parquet")
print(f"  Records: {len(df):,}")

df["lat"] = df["exact_lat"].fillna(df["LAT"])
df["lon"] = df["exact_lon"].fillna(df["LONG"])
mask = df["lat"].notna() & df["lon"].notna()
print(f"  With coordinates: {mask.sum():,}")


# =============================================================================
# SECTION 3: COMPUTE DISTANCES
# =============================================================================

print("\nBuilding BallTree from coastal points...")
tree = BallTree(np.deg2rad(coast_coords), metric="haversine") # Convert coastal points from degrees to radians (haversine needs radians not degrees)

prop_coords = np.deg2rad(df.loc[mask, ["lat", "lon"]].values)

print("Computing distance to nearest coastline...")
dist, _ = tree.query(prop_coords, k=1)
df.loc[mask, "dist_coast_km"] = dist.flatten() * 6371.0


# =============================================================================
# SECTION 4: SUMMARY AND SAVE
# =============================================================================

col = "dist_coast_km"
print(f"\n--- Coastline Distance Summary ---")
print(f"  Mean:   {df[col].mean():.2f} km")
print(f"  Median: {df[col].median():.2f} km")
print(f"  Max:    {df[col].max():.2f} km")
print(f"  < 5km:  {(df[col] < 5).sum():,} ({(df[col] < 5).mean()*100:.1f}%)")
print(f"  < 20km: {(df[col] < 20).sum():,} ({(df[col] < 20).mean()*100:.1f}%)")

df = df.drop(columns=["lat", "lon"], errors="ignore")

print("\nSaving...")
df.to_parquet("Outputs/lr_epc_coast.parquet", compression="snappy")
print("Done — saved to Outputs/lr_epc_coast.parquet")
