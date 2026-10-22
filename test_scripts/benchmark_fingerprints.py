import time
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

smiles_list = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccccc1", "CCO"] * 2500 # 10,000 molecules

t0 = time.time()
fps = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

fps = np.array(fps)
dt = time.time() - t0
print(f"Generated {len(fps)} fingerprints in {dt:.2f}s ({len(fps)/dt:.0f} mol/s)")
print("Fingerprint shape:", fps.shape)
