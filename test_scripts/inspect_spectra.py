import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# Inspect test parquet spectra
test = pd.read_parquet("data_set/test.parquet")
print("Test spectra count:", len(test))
sample_row = test.iloc[0]
print("Test sample row:")
print("  molecule_id:", sample_row["molecule_id"])
print("  precursor_mz:", sample_row["precursor_mz"])
print("  adduct:", sample_row["adduct"])
print("  collision_energy_ev:", sample_row["collision_energy_ev"])
print("  num_peaks:", len(sample_row["ms2_mzs"]))
print("  mzs sample (first 5):", sample_row["ms2_mzs"][:5])
print("  intensities sample (first 5):", sample_row["ms2_normalized_intensities"][:5])
print("  intensity min, max, sum:", min(sample_row["ms2_normalized_intensities"]), max(sample_row["ms2_normalized_intensities"]), sum(sample_row["ms2_normalized_intensities"]))

# Inspect train parquet spectra
pf = pq.ParquetFile("data_set/train.parquet")
train_sample = pf.read_row_group(0, columns=["normalized_smiles", "precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities", "num_peaks"]).to_pandas().iloc[0]
print("\nTrain sample row:")
print("  smiles:", train_sample["normalized_smiles"])
print("  precursor_mz:", train_sample["precursor_mz"])
print("  num_peaks:", len(train_sample["ms2_mzs"]))
print("  mzs sample (first 5):", train_sample["ms2_mzs"][:5])
print("  intensities sample (first 5):", train_sample["ms2_normalized_intensities"][:5])
