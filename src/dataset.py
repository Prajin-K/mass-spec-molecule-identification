"""
PyTorch Dataset and DataLoader for Tandem Mass Spectrometry (LC-MS/MS).
Efficiently processes spectra into dual-channel binned vectors (fragment + neutral loss),
with leakage-free molecule-level cross-validation splits.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src.spectral_encoder import SpectralEncoder, ADDUCT_VOCAB

_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class MS2Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        encoder: Optional[SpectralEncoder] = None,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder or SpectralEncoder(max_mz=1000.0, bin_width=0.5)
        self.is_test = is_test

        # Pre-extract numerical columns for fast indexing
        self.precursor_mzs = self.df["precursor_mz"].values.astype(np.float32)
        
        # Parse adducts to vocabulary indices
        adduct_col = self.df["adduct"].fillna("UNKNOWN").values
        self.adduct_indices = np.array(
            [ADDUCT_VOCAB.get(str(a).strip(), ADDUCT_VOCAB["UNKNOWN"]) for a in adduct_col],
            dtype=np.int64
        )

        # Parse ionization mode
        mode_col = self.df["ionization_mode"].fillna("positive").values
        self.modes = np.array(
            [1.0 if "pos" in str(m).lower() else -1.0 for m in mode_col],
            dtype=np.float32
        )

        # Parse collision energies (take mean if list)
        ces = []
        for ce_val in self.df["collision_energy_ev"].values:
            if isinstance(ce_val, (list, np.ndarray)) and len(ce_val) > 0:
                ces.append(float(np.mean(ce_val)))
            elif isinstance(ce_val, (int, float)) and not np.isnan(ce_val):
                ces.append(float(ce_val))
            else:
                ces.append(30.0)  # Default median collision energy
        self.ces = np.array(ces, dtype=np.float32)

        # Precompute target fingerprints if training
        self.target_fps: Optional[np.ndarray] = None
        if not self.is_test and "normalized_smiles" in self.df.columns:
            self._precompute_targets()

    def _precompute_targets(self):
        fps_list = []
        for smi in self.df["normalized_smiles"].values:
            mol = Chem.MolFromSmiles(smi) if pd.notna(smi) else None
            if mol is not None:
                fp = _mfpgen.GetFingerprintAsNumPy(mol)
            else:
                fp = np.zeros(2048, dtype=np.uint8)
            fps_list.append(fp)
        self.target_fps = np.stack(fps_list, axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        mzs = row["ms2_mzs"]
        intensities = row["ms2_normalized_intensities"]
        precursor = self.precursor_mzs[idx]

        spec = self.encoder.encode_spectrum(mzs, intensities, precursor)

        item = {
            "spec": torch.from_numpy(spec),                                  # (2, num_bins)
            "precursor_mz": torch.tensor([precursor / 500.0], dtype=torch.float32),  # (1,)
            "adduct_idx": torch.tensor(self.adduct_indices[idx], dtype=torch.long),  # ()
            "ce": torch.tensor([self.ces[idx] / 50.0], dtype=torch.float32),        # (1,)
            "mode": torch.tensor([self.modes[idx]], dtype=torch.float32),            # (1,)
        }

        if self.target_fps is not None:
            item["target_fp"] = torch.from_numpy(self.target_fps[idx])       # (2048,)

        if "molecule_id" in self.df.columns:
            item["molecule_id"] = str(row["molecule_id"])

        if "spectrum_id" in self.df.columns:
            item["spectrum_id"] = str(row["spectrum_id"])

        return item
