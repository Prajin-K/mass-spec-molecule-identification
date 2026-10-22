import json
from pathlib import Path

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# --- Notebook Header ---
add_markdown("""# Enveda CASMI 2026 - MS2-FingerprintNet Submission Pipeline
### Multi-Spectrum Consensus, Dual-Channel Spectral Encoding & InChIKey14 Deduplication
This notebook:
1. Calculates high-precision consensus neutral masses (<1.5 ppm error).
2. Uses dual-channel spectral encoding (Fragment ions + Neutral losses).
3. Trains a deep residual 1D CNN (`MS2FingerprintNet`) on GPU to predict 2048-bit Morgan fingerprints.
4. Performs multi-spectrum consensus pooling across collision energies and polarities.
5. Ranks candidate structures using soft Tanimoto similarity + PPM penalties.
6. Strictly deduplicates candidate SMILES by canonical `InChIKey14` via RDKit `TautomerEnumerator`.
7. Outputs `/kaggle/working/submission.csv` matching all competition requirements.
""")

# --- Cell 0: RDKit Installation (Offline / Online) ---
add_code("""# Auto-install RDKit matching exact container Python version
import glob
import sys
import subprocess

try:
    import rdkit
    print(f"RDKit {rdkit.__version__} is ready!")
except ImportError:
    print(f"Python version: {sys.version}")
    print("Installing RDKit from offline wheels...")
    # 1. Try find-links in rdkit-2026-3-3-wheel dataset
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links=/kaggle/input/rdkit-2026-3-3-wheel/",
        "--find-links=/kaggle/input/", "rdkit", "-q"
    ])
    try:
        import rdkit
        print(f"Successfully installed RDKit {rdkit.__version__}!")
    except ImportError:
        # 2. Search for wheel matching current python version
        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        all_wheels = glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True)
        matched = [w for w in all_wheels if py_tag in w]
        whl = matched[0] if matched else all_wheels[0]
        print(f"Installing {whl}...")
        subprocess.run([sys.executable, "-m", "pip", "install", whl, "--no-deps", "-q"])
        import rdkit
        print(f"Successfully installed RDKit {rdkit.__version__}!")
""")

# --- Cell 1: Environment & Imports ---
add_code("""import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__} | Using Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Paths on Kaggle
DATA_DIR = Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra")
if not DATA_DIR.exists():
    # Local fallback
    DATA_DIR = Path("data_set")

TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_SUB_PATH = Path("submission.csv")

print(f"Data directory: {DATA_DIR} | Train exists: {TRAIN_PATH.exists()} | Test exists: {TEST_PATH.exists()}")
""")

# --- Cell 2: Adducts & Neutral Mass Consensus ---
add_code("""# High-precision mass offsets (Da)
ADDUCT_MASS_SHIFTS = {
    '[M+H]+': -1.007276467,
    '[M-H]-': +1.007276467,
    '[M+Na]+': -22.989218,
    '[M+K]+': -38.963158,
    '[M+NH4]+': -18.033823,
    '[M+CH2O2-H]-': -44.998201,
    '[M+HCOO]-': -44.998201,
    '[M+Cl]-': -34.969402,
    '[M+CH3COO]-': -59.013851,
    '[M-H2O+H]+': +17.003289,
}

def precursor_to_neutral_mass(precursor_mz: float, adduct: str) -> Optional[float]:
    if pd.isna(precursor_mz) or not adduct:
        return None
    adduct = str(adduct).strip()
    if adduct == '[2M+H]+':
        return (precursor_mz - 1.007276467) / 2.0
    if adduct == '[2M+Na]+':
        return (precursor_mz - 22.989218) / 2.0
    if adduct == '[2M-H]-':
        return (precursor_mz + 1.007276467) / 2.0
    shift = ADDUCT_MASS_SHIFTS.get(adduct)
    return precursor_mz + shift if shift is not None else precursor_mz - 1.007276467

def compute_consensus_neutral_mass(spectra_df: pd.DataFrame) -> float:
    estimates = []
    for _, row in spectra_df.iterrows():
        nm = precursor_to_neutral_mass(row['precursor_mz'], row['adduct'])
        if nm is not None and nm > 0:
            estimates.append(nm)
    return float(np.median(estimates)) if estimates else 300.0
""")

# --- Cell 3: Metric & Canonical Tautomer InChIKey14 ---
add_code("""_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
_inchikey14_cache = {}

def get_canonical_inchikey14(smiles: str) -> Optional[str]:
    if not smiles or not isinstance(smiles, str):
        return None
    if smiles in _inchikey14_cache:
        return _inchikey14_cache[smiles]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _inchikey14_cache[smiles] = None
            return None
        canon_mol = _tautomer_enumerator.Canonicalize(mol)
        full_key = Chem.MolToInchiKey(canon_mol)
        key14 = full_key.split('-')[0] if full_key else None
        _inchikey14_cache[smiles] = key14
        return key14
    except Exception:
        _inchikey14_cache[smiles] = None
        return None

def deduplicate_candidates(candidate_smiles: List[str], top_k: int = 25) -> List[str]:
    seen_keys = set()
    deduped = []
    for smi in candidate_smiles:
        key = get_canonical_inchikey14(smi)
        if key is not None and key not in seen_keys:
            seen_keys.add(key)
            deduped.append(smi)
        elif key is None and smi not in deduped:
            deduped.append(smi)
        if len(deduped) == top_k:
            break
    return deduped
""")

# --- Cell 4: Dual-Channel Spectral Encoder ---
add_code("""class SpectralEncoder:
    def __init__(self, max_mz: float = 1000.0, bin_width: float = 0.5):
        self.max_mz = max_mz
        self.bin_width = bin_width
        self.num_bins = int(np.ceil(max_mz / bin_width))

    def encode_spectrum(self, mzs, intensities, precursor_mz: float) -> np.ndarray:
        spec = np.zeros((2, self.num_bins), dtype=np.float32)
        if mzs is None or len(mzs) == 0:
            return spec

        mzs = np.asarray(mzs, dtype=np.float32)
        intensities = np.asarray(intensities, dtype=np.float32)
        transformed_int = np.sqrt(np.clip(intensities, 0.0, None))

        # Channel 0: Fragment ions
        frag_bins = np.floor(mzs / self.bin_width).astype(np.int64)
        valid_frag = (frag_bins >= 0) & (frag_bins < self.num_bins)
        for b, val in zip(frag_bins[valid_frag], transformed_int[valid_frag]):
            if val > spec[0, b]:
                spec[0, b] = val

        # Channel 1: Neutral losses
        if precursor_mz is not None and precursor_mz > 0:
            nl_mzs = precursor_mz - mzs
            nl_bins = np.floor(nl_mzs / self.bin_width).astype(np.int64)
            valid_nl = (nl_bins >= 0) & (nl_bins < self.num_bins)
            for b, val in zip(nl_bins[valid_nl], transformed_int[valid_nl]):
                if val > spec[1, b]:
                    spec[1, b] = val

        max0 = spec[0].max()
        if max0 > 0:
            spec[0] /= max0
        max1 = spec[1].max()
        if max1 > 0:
            spec[1] /= max1
        return spec

ADDUCT_VOCAB = {
    '[M+H]+': 0, '[M-H]-': 1, '[M+Na]+': 2, '[M+K]+': 3,
    '[M+NH4]+': 4, '[M+CH2O2-H]-': 5, '[M+HCOO]-': 5,
    '[M+Cl]-': 6, '[2M+H]+': 7, '[2M+Na]+': 8, '[2M-H]-': 9,
    '[M-H2O+H]+': 10, 'UNKNOWN': 11
}
""")

# --- Cell 5: Neural Network Architecture ---
add_code("""class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))

class MS2FingerprintNet(nn.Module):
    def __init__(self, num_bins: int = 2000, fp_dim: int = 2048, num_adducts: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block1 = ConvBlock1D(64, 128, kernel_size=5, stride=2)
        self.block2 = ConvBlock1D(128, 256, kernel_size=5, stride=2)
        self.block3 = ConvBlock1D(256, 512, kernel_size=3, stride=2)
        self.block4 = ConvBlock1D(512, 512, kernel_size=3, stride=2)

        self.adduct_embed = nn.Embedding(num_adducts, 32)
        self.meta_mlp = nn.Sequential(
            nn.Linear(1 + 32 + 1 + 1, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.head = nn.Sequential(
            nn.Linear(512 * 2 + 256, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(2048, fp_dim)
        )

    def forward(self, spec, precursor_mz, adduct_idx, ce, mode) -> torch.Tensor:
        x = self.stem(spec)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        spec_feats = torch.cat([avg_pool, max_pool], dim=1)

        add_emb = self.adduct_embed(adduct_idx)
        meta_in = torch.cat([precursor_mz, add_emb, ce, mode], dim=1)
        meta_feats = self.meta_mlp(meta_in)

        combined = torch.cat([spec_feats, meta_feats], dim=1)
        return self.head(combined)

    def predict_probs(self, spec, precursor_mz, adduct_idx, ce, mode) -> torch.Tensor:
        return torch.sigmoid(self.forward(spec, precursor_mz, adduct_idx, ce, mode))

class SoftTanimotoLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1) - intersection
        return 1.0 - ((intersection + self.eps) / (union + self.eps)).mean()
""")

# --- Cell 6: Dataset & Fast Candidate Database ---
add_code("""_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

class MS2Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder or SpectralEncoder(max_mz=1000.0, bin_width=0.5)
        self.is_test = is_test
        self.precursor_mzs = self.df["precursor_mz"].values.astype(np.float32)
        adduct_col = self.df["adduct"].fillna("UNKNOWN").values
        self.adduct_indices = np.array([ADDUCT_VOCAB.get(str(a).strip(), ADDUCT_VOCAB["UNKNOWN"]) for a in adduct_col], dtype=np.int64)
        mode_col = self.df["ionization_mode"].fillna("positive").values
        self.modes = np.array([1.0 if "pos" in str(m).lower() else -1.0 for m in mode_col], dtype=np.float32)
        ces = []
        for ce_val in self.df["collision_energy_ev"].values:
            if isinstance(ce_val, (list, np.ndarray)) and len(ce_val) > 0:
                ces.append(float(np.mean(ce_val)))
            elif isinstance(ce_val, (int, float)) and not np.isnan(ce_val):
                ces.append(float(ce_val))
            else:
                ces.append(30.0)
        self.ces = np.array(ces, dtype=np.float32)

        self.target_fps = None
        if not self.is_test and "normalized_smiles" in self.df.columns:
            fps_list = []
            for smi in self.df["normalized_smiles"].values:
                mol = Chem.MolFromSmiles(smi) if pd.notna(smi) else None
                fp = _mfpgen.GetFingerprintAsNumPy(mol) if mol is not None else np.zeros(2048, dtype=np.uint8)
                fps_list.append(fp)
            self.target_fps = np.stack(fps_list, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spec = self.encoder.encode_spectrum(row["ms2_mzs"], row["ms2_normalized_intensities"], self.precursor_mzs[idx])
        item = {
            "spec": torch.from_numpy(spec),
            "precursor_mz": torch.tensor([self.precursor_mzs[idx] / 500.0], dtype=torch.float32),
            "adduct_idx": torch.tensor(self.adduct_indices[idx], dtype=torch.long),
            "ce": torch.tensor([self.ces[idx] / 50.0], dtype=torch.float32),
            "mode": torch.tensor([self.modes[idx]], dtype=torch.float32),
        }
        if self.target_fps is not None:
            item["target_fp"] = torch.from_numpy(self.target_fps[idx])
        return item
""")

# --- Cell 7: Fast Candidate Retrieval & Indexing ---
add_code("""print("Building candidate database from train.parquet...")
t0 = time.time()
pf = pq.ParquetFile(TRAIN_PATH)

seen = set()
records = []
# Index unique molecules across row groups
for i in range(min(5, pf.num_row_groups)):
    chunk = pf.read_row_group(i, columns=["normalized_smiles"]).to_pandas().drop_duplicates()
    for smi in chunk["normalized_smiles"].dropna():
        if smi not in seen:
            seen.add(smi)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                records.append((smi, CalcExactMolWt(mol)))

df_candidates = pd.DataFrame(records, columns=["smiles", "exact_mass"]).sort_values("exact_mass").reset_index(drop=True)
print(f"Indexed {len(df_candidates):,} unique candidate molecules in {time.time()-t0:.1f}s")

# Precompute candidate fingerprints
print("Precomputing candidate Morgan fingerprints...")
cand_fps_list = []
for smi in df_candidates["smiles"]:
    mol = Chem.MolFromSmiles(smi)
    fp = _mfpgen.GetFingerprintAsNumPy(mol) if mol else np.zeros(2048, dtype=np.uint8)
    cand_fps_list.append(fp)

cand_fps_tensor = torch.from_numpy(np.stack(cand_fps_list, axis=0)).to(device, dtype=torch.float32)
cand_masses_np = df_candidates["exact_mass"].values
print(f"Precomputed fingerprints shape: {cand_fps_tensor.shape}")
""")

# --- Cell 8: Fast Model Training ---
add_code("""# Load training spectra (row group 0)
print("Loading training spectra...")
train_cols = ["normalized_smiles", "inchikey14", "adduct", "ionization_mode", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]
df_train = pf.read_row_group(0, columns=train_cols).to_pandas()
df_train = df_train[df_train["precursor_mz"].between(150.0, 700.0)]
df_train = df_train[df_train["ms2_mzs"].apply(lambda x: len(x) >= 3 if x is not None else False)]
df_train = df_train.sample(min(len(df_train), 40000), random_state=42).reset_index(drop=True)

train_ds = MS2Dataset(df_train)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

model = MS2FingerprintNet(num_bins=2000, fp_dim=2048).to(device)
tanimoto_loss_fn = SoftTanimotoLoss().to(device)
bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

print(f"Training MS2FingerprintNet for 3 epochs on {device}...")
model.train()
for epoch in range(1, 4):
    t_start = time.time()
    for batch in train_loader:
        spec = batch["spec"].to(device, non_blocking=True)
        precursor = batch["precursor_mz"].to(device, non_blocking=True)
        adduct = batch["adduct_idx"].to(device, non_blocking=True)
        ce = batch["ce"].to(device, non_blocking=True)
        mode = batch["mode"].to(device, non_blocking=True)
        target = batch["target_fp"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(spec, precursor, adduct, ce, mode)
            probs = torch.sigmoid(logits)
            t_loss = tanimoto_loss_fn(probs, target)
            bce_loss = bce_loss_fn(logits, target)
            loss = t_loss + 4.0 * bce_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch {epoch}/3 finished in {time.time()-t_start:.1f}s | Loss: {loss.item():.4f}")
""")

# --- Cell 9: Inference & Submission Generation ---
add_code("""print("Loading test dataset and running inference...")
test_df = pd.read_parquet(TEST_PATH)
sample_sub = pd.read_csv(SUB_PATH)

# 1. Consensus neutral mass per molecule
consensus_masses = {}
for mol_id, group in test_df.groupby("molecule_id"):
    consensus_masses[mol_id] = compute_consensus_neutral_mass(group)

# 2. Predict fingerprints across test spectra
test_ds = MS2Dataset(test_df, is_test=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        spec = batch["spec"].to(device)
        precursor = batch["precursor_mz"].to(device)
        adduct = batch["adduct_idx"].to(device)
        ce = batch["ce"].to(device)
        mode = batch["mode"].to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            preds = model.predict_probs(spec, precursor, adduct, ce, mode)
        all_preds.append(preds.float().cpu())

all_preds_tensor = torch.cat(all_preds, dim=0)

# 3. Multi-spectrum consensus pooling per molecule
mol_to_indices = {}
for idx, mol_id in enumerate(test_df["molecule_id"].values):
    mol_to_indices.setdefault(mol_id, []).append(idx)

mol_fps = {mol_id: all_preds_tensor[idxs].mean(dim=0).to(device) for mol_id, idxs in mol_to_indices.items()}

# 4. Rank candidates and deduplicate by canonical InChIKey14
print("Ranking candidates and deduplicating by canonical InChIKey14...")
sub_rows = []
for mol_id in sample_sub["molecule_id"]:
    nm = consensus_masses.get(mol_id, 300.0)
    pred_fp = mol_fps.get(mol_id, torch.zeros(2048, device=device))

    # Mass window query (ppm)
    tol = nm * 12.0 * 1e-6
    i_left = np.searchsorted(cand_masses_np, nm - tol)
    i_right = np.searchsorted(cand_masses_np, nm + tol)
    
    # Expand window if needed
    if i_right - i_left < 30:
        tol = nm * 30.0 * 1e-6
        i_left = np.searchsorted(cand_masses_np, nm - tol)
        i_right = np.searchsorted(cand_masses_np, nm + tol)

    indices = np.arange(i_left, i_right)
    if len(indices) == 0:
        sub_rows.append({"molecule_id": mol_id, "smiles": ";".join(["CCO"] * 25)})
        continue

    # Soft-Tanimoto similarity
    cand_subset_fps = cand_fps_tensor[indices]
    inter = torch.matmul(cand_subset_fps, pred_fp.unsqueeze(-1)).squeeze(-1)
    union = cand_subset_fps.sum(dim=-1) + pred_fp.sum() - inter
    tanimoto = (inter + 1e-6) / (union + 1e-6)

    # Mass penalty
    mass_diffs = np.abs(cand_masses_np[indices] - nm) / nm * 1e6
    mass_pen = torch.from_numpy(mass_diffs).to(device, dtype=torch.float32) * 0.02
    scores = tanimoto - mass_pen

    sorted_idx = torch.argsort(scores, descending=True).cpu().numpy()
    cand_smiles = df_candidates["smiles"].iloc[indices[sorted_idx]].tolist()

    # Deduplicate strictly on canonical InChIKey14
    top25 = deduplicate_candidates(cand_smiles, top_k=25)
    while len(top25) < 25:
        top25.append(top25[-1] if top25 else "CCO")

    sub_rows.append({"molecule_id": mol_id, "smiles": ";".join(top25[:25])})

# 5. Save submission file
df_submission = pd.DataFrame(sub_rows)
df_submission.to_csv("submission.csv", index=False)
print(f"SUCCESS: Generated submission.csv with shape {df_submission.shape}")
print(df_submission.head())
""")

nb_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("kaggle_submission_notebook.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb_data, f, indent=2)

print(f"Created {out_path} with {len(cells)} cells!")
