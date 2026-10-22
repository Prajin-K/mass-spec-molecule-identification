import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import time

print("Testing chunk loading from train.parquet...")
t0 = time.time()
pf = pq.ParquetFile("data_set/train.parquet")
# Read row group 0
cols = ["normalized_smiles", "inchikey14", "adduct", "ionization_mode", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]
df_chunk = pf.read_row_group(0, columns=cols).to_pandas()
print(f"Loaded {len(df_chunk)} rows from group 0 in {time.time()-t0:.2f}s")

# Filter for spectra with valid peaks and reasonable precursor mz
df_chunk = df_chunk[df_chunk["precursor_mz"].between(150, 700)]
df_chunk = df_chunk[df_chunk["ms2_mzs"].apply(lambda x: len(x) > 3 if x is not None else False)]
print(f"Filtered to {len(df_chunk)} high quality spectra")
print(f"Unique molecules: {df_chunk['inchikey14'].nunique()}")
