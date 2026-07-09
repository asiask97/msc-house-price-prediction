"""
joining_epc_and_lr.py
------------
Matches each Land Registry sale to an EPC record using
exact match on normalised postcode + address key.
Only keeps EPCs lodged before or on sale date to prevent ML leakage.

Usage:
    python joining_epc_and_lr.py

Output:
    Data/lr_epc_matched.parquet
"""

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm

lr_folder = Path("Data/LR")

lr_columns = [
    "transaction_id", "price", "date", "postcode",
    "property_type", "new_build", "duration",
    "paon", "saon", "street", "locality",
    "town", "district", "county",
    "ppd_category", "record_status"
]

def normalise_postcode(postcode):
    """Uppercase and remove spaces from postcode for consistent matching."""
    return str(postcode).upper().replace(" ", "")

def normalise_address(address):
    """
    Normalise address string for matching.
    Handles common formatting differences between Land Registry and EPC data:
    - ST. -> ST  (saint abbreviation)
    - Hyphens -> spaces
    - Apostrophes removed
    - Double spaces -> single space
    - Commas removed
    - Strip and uppercase
    """
    address = str(address).upper().strip()
    address = address.replace("ST.", "ST")
    address = address.replace("-", " ")
    address = re.sub(r"'", "", address)
    address = address.replace(",", "")
    address = re.sub(r"\bRD\b", "ROAD", address)
    address = re.sub(r"\bAVE\b", "AVENUE", address)
    address = re.sub(r"\s+", " ", address)
    return address.strip()


# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

lr_files = list(lr_folder.glob("*.csv"))
print("Loading Land Registry...")
df_lr = pd.concat(
    [pd.read_csv(f, header=None, names=lr_columns) for f in tqdm(lr_files, desc="Loading LR")],
    ignore_index=True
)
print(f"Loaded {len(df_lr):,} LR records")
print(f"Unique postcodes in LR: {df_lr['postcode'].nunique():,}")

print("\nLoading EPC...")
df_epc = pd.read_parquet("Data/EPC/epc.parquet")
print(f"Loaded {len(df_epc):,} EPC records")
print(f"Unique postcodes in EPC: {df_epc['postcode'].nunique():,}")


# =============================================================================
# SECTION 2: PREPARE KEYS FOR MATCHING
# =============================================================================

print("\nParsing dates...")
df_lr["date"] = pd.to_datetime(df_lr["date"])
df_epc["lodgement_date"] = pd.to_datetime(df_epc["lodgement_date"])

# Normalise postcodes
print("Normalising postcodes...")
df_lr["postcode_clean"] = df_lr["postcode"].apply(normalise_postcode)
df_epc["postcode_clean"] = df_epc["postcode"].apply(normalise_postcode)

# Build normalised address key for LR
# Include saon (flat number) + paon (house number) + street
print("Building normalised address keys...")
df_lr["address_key"] = (
    df_lr["saon"].fillna("").str.strip() + " " +
    df_lr["paon"].fillna("").str.strip() + " " +
    df_lr["street"].fillna("").str.strip()
).apply(normalise_address)

df_epc["address_key"] = df_epc["address_key"].apply(normalise_address)

print(f"Sample LR address keys:  {df_lr['address_key'].head(3).tolist()}")
print(f"Sample EPC address keys: {df_epc['address_key'].head(3).tolist()}")


# =============================================================================
# SECTION 3: EXACT MATCH ON POSTCODE + ADDRESS KEY
# =============================================================================

print(f"\nMerging {len(df_lr):,} LR records with {len(df_epc):,} EPC records...")
merged = df_lr.merge(df_epc, on=["postcode_clean", "address_key"], how="left")
print("Merge complete")


# =============================================================================
# SECTION 4: KEEP ONLY EPCs BEFORE SALE DATE (PREVENT ML LEAKAGE)
# =============================================================================

print("\nFiltering to EPCs lodged before or on sale date...")

# Split matched and unmatched
has_epc = merged[merged["lodgement_date"].notna() & (merged["lodgement_date"] <= merged["date"])]
no_epc = merged[~merged["transaction_id"].isin(has_epc["transaction_id"].unique())]

# For each transaction keep the most recent EPC before the sale
print("Picking most recent EPC per transaction...")
best_before = (has_epc.sort_values("lodgement_date").groupby("transaction_id").last().reset_index())

# For unmatched keep one row per transaction, only LR columns
lr_cols = [c for c in no_epc.columns if c in df_lr.columns or c.endswith("_x") or c in ["postcode_clean", "address_key"]]
no_epc = no_epc.drop_duplicates(subset="transaction_id")[lr_cols]

final = pd.concat([best_before, no_epc], ignore_index=True)

print(f"\n--- Results ---")
print(f"Total:              {len(final):,}")
print(f"Matched to EPC:     {final['lodgement_date'].notna().sum():,} ({final['lodgement_date'].notna().sum()/len(final)*100:.1f}%)")
print(f"Unmatched:          {final['lodgement_date'].isna().sum():,} ({final['lodgement_date'].isna().sum()/len(final)*100:.1f}%)")


# =============================================================================
# SECTION 5: SAVE RESULTS
# =============================================================================

print("\nSaving to parquet...")
final.to_parquet("Outputs/lr_epc_matched.parquet", compression="snappy")
print("Done")

# =============================================================================
# SECTION 6: SAVE SAMPLE LOG
# =============================================================================

print("\nSaving sample log...")
sample = final[final["lodgement_date"].notna()].sample(n=5000, random_state=42)
sample.to_csv("Outputs/sample_log.csv", index=False)
print(f"Saved {len(sample):,} rows to Outputs/sample_log.csv")