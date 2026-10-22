from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
import pyarrow.parquet as pq
import pandas as pd

te = rdMolStandardize.TautomerEnumerator()

def get_inchikey14(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_mol = te.Canonicalize(mol)
    full_inchikey = Chem.MolToInchiKey(canonical_mol)
    return full_inchikey.split('-')[0]

# Check sample smiles
smi1 = "OC=CC"
smi2 = "O=CCC"
k1 = get_inchikey14(smi1)
k2 = get_inchikey14(smi2)
print(f"Tautomer test: {smi1} -> {k1}, {smi2} -> {k2}, match: {k1 == k2}")

# Check train parquet
pf = pq.ParquetFile("data_set/train.parquet")
chunk = pf.read_row_group(0, columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz", "ionization_mode"]).to_pandas().head(10)
print("\nTrain sample validation:")
for idx, row in chunk.iterrows():
    smi = row["normalized_smiles"]
    k_expected = row["inchikey14"]
    k_calc = get_inchikey14(smi)
    mol = Chem.MolFromSmiles(smi)
    exact_wt = CalcExactMolWt(mol) if mol else 0
    diff = row["precursor_mz"] - exact_wt
    print(f"Row {idx:2d} | Adduct: {row['adduct']:<10} | Mode: {row['ionization_mode']:<8} | Diff: {diff:+.4f} | Match: {k_expected == k_calc} (Exp: {k_expected}, Calc: {k_calc})")
