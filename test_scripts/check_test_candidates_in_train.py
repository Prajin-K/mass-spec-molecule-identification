import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
import time

print("Loading train unique molecules...")
t0 = time.time()
pf = pq.ParquetFile("data_set/train.parquet")

seen = set()
records = []
for i in range(pf.num_row_groups):
    df = pf.read_row_group(i, columns=["normalized_smiles", "molecular_formula"]).to_pandas()
    # deduplicate on smiles
    df_unique = df.drop_duplicates(subset=["normalized_smiles"])
    for _, row in df_unique.iterrows():
        smi = row["normalized_smiles"]
        if smi not in seen:
            seen.add(smi)
            records.append((smi, row["molecular_formula"]))
    print(f"Group {i+1}/21: {len(seen)} unique molecules")

print(f"Loaded {len(records)} molecules in {time.time()-t0:.1f}s")
df_unique_mols = pd.DataFrame(records, columns=["smiles", "formula"])

# Calculate exact weights using RDKit
print("Calculating exact molecular weights...")
t0 = time.time()
weights = []
valid_indices = []
for idx, smi in enumerate(df_unique_mols["smiles"]):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        weights.append(CalcExactMolWt(mol))
        valid_indices.append(idx)

df_db = df_unique_mols.iloc[valid_indices].copy()
df_db["exact_mass"] = weights
df_db = df_db.sort_values("exact_mass").reset_index(drop=True)
print(f"Calculated {len(df_db)} exact weights in {time.time()-t0:.1f}s")

# Save cached database for super fast retrieval
df_db.to_parquet("checkpoints/train_molecules_db.parquet", index=False)
print("Saved checkpoints/train_molecules_db.parquet")

# Now check test set molecules
print("\nChecking test molecules against database...")
test = pd.read_parquet("data_set/test.parquet")
adduct_shifts = {
    '[M+H]+': -1.007276,
    '[M-H]-': +1.007276,
    '[M+Na]+': -22.989218,
    '[M+K]+': -38.963158,
    '[M+NH4]+': -18.033823,
    '[M+CH2O2-H]-': -44.998201,
    '[M+Cl]-': -34.969402,
}

test["neutral_mass"] = test["precursor_mz"] + test["adduct"].map(adduct_shifts).fillna(0)
mol_consensus = test.groupby("molecule_id")["neutral_mass"].median().reset_index()

db_masses = df_db["exact_mass"].values

candidates_5ppm = []
candidates_10ppm = []
candidates_20ppm = []

for _, row in mol_consensus.iterrows():
    m = row["neutral_mass"]
    # 5 ppm
    tol_5 = m * 5e-6
    i_left = np.searchsorted(db_masses, m - tol_5)
    i_right = np.searchsorted(db_masses, m + tol_5)
    candidates_5ppm.append(i_right - i_left)

    # 10 ppm
    tol_10 = m * 10e-6
    i_left = np.searchsorted(db_masses, m - tol_10)
    i_right = np.searchsorted(db_masses, m + tol_10)
    candidates_10ppm.append(i_right - i_left)

    # 20 ppm
    tol_20 = m * 20e-6
    i_left = np.searchsorted(db_masses, m - tol_20)
    i_right = np.searchsorted(db_masses, m + tol_20)
    candidates_20ppm.append(i_right - i_left)

c5 = pd.Series(candidates_5ppm)
c10 = pd.Series(candidates_10ppm)
c20 = pd.Series(candidates_20ppm)

print(f"Test molecules with >=1 candidate at 5 ppm:  {(c5 >= 1).sum()} / 400 (avg candidates: {c5.mean():.1f}, median: {c5.median():.0f})")
print(f"Test molecules with >=1 candidate at 10 ppm: {(c10 >= 1).sum()} / 400 (avg candidates: {c10.mean():.1f}, median: {c10.median():.0f})")
print(f"Test molecules with >=1 candidate at 20 ppm: {(c20 >= 1).sum()} / 400 (avg candidates: {c20.mean():.1f}, median: {c20.median():.0f})")
print(f"Test molecules with >=25 candidates at 10 ppm: {(c10 >= 25).sum()} / 400")
