"""
loading_data.py
---------------
One-time data preparation script.
1) Loads all Land Registry CSVs
2) Loads all EPC CSVs file by file, filters to LR postcodes immediately,
   and saves as parquet. Keeps all EPCs per property so joining script
   can pick the closest one relative to each sale date.

Usage:
    python loading_data.py

Output:
    Data/EPC/epc.parquet
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

lr_folder = Path("./Data/LR")
epc_folder = Path("./Data/EPC")

lr_columns = [
    "transaction_id", "price", "date", "postcode",
    "property_type", "new_build", "duration",
    "paon", "saon", "street", "locality",
    "town", "district", "county",
    "ppd_category", "record_status"
]

epc_cols = [
    "postcode", "address1", "lodgement_date", "total_floor_area",
    "number_habitable_rooms", "current_energy_rating",
    "current_energy_efficiency", "construction_age_band",
    "built_form", "property_type", "main_fuel",
    "mains_gas_flag", "transaction_type", "tenure", "uprn"
]

# Load Land Registry
lr_files = list(lr_folder.glob("*.csv"))
df_lr = pd.concat(
    [pd.read_csv(f, header=None, names=lr_columns) for f in tqdm(lr_files, desc="Loading LR")],
    ignore_index=True
)
print(f"Loaded {len(df_lr):,} LR records")
lr_postcodes = set(df_lr["postcode"].unique())

# Load EPC file by file, filter immediately to save memory
epc_files = list(epc_folder.rglob("certificates-*.csv"))
chunks = []

# For each file: load only needed columns, filter to LR postcodes,
# parse date, and build address key by removing commas/whitespace and uppercasing to match LR format e.g. "23 WESTMORLAND CLOSE"
for f in tqdm(epc_files, desc="Loading EPC"):
    chunk = pd.read_csv(f, low_memory=False, usecols=epc_cols)
    chunk = chunk[chunk["postcode"].isin(lr_postcodes)]
    chunk["lodgement_date"] = pd.to_datetime(chunk["lodgement_date"])
    chunk["address_key"] = (chunk["address1"]
                            .str.replace(",", "", regex=False)
                            .str.strip()
                            .str.upper())
    chunks.append(chunk)

df_epc = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df_epc):,} EPC records after filtering")

# Save all records — joining script will pick closest EPC per sale
print(f"Saving {len(df_epc):,} records to parquet...")
df_epc.to_parquet("./Data/EPC/epc.parquet", compression="snappy")
print("Done")