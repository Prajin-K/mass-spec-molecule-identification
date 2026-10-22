"""
Candidate Molecule Database and Fast Retrieval Engine.
Indexes all reference molecules from train data by exact monoisotopic mass,
stores pre-computed Morgan fingerprints (2048-bit), and provides sub-millisecond retrieval.
"""

from typing import List, Tuple, Optional
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch

_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

DB_PATH = Path("checkpoints/train_molecules_db.parquet")
FP_PATH = Path("checkpoints/train_molecules_fps.npy")


class CandidateDatabase:
    def __init__(self, db_path: Path = DB_PATH, fp_path: Path = FP_PATH):
        self.db_path = db_path
        self.fp_path = fp_path
        self.df: Optional[pd.DataFrame] = None
        self.masses: Optional[np.ndarray] = None
        self.fps: Optional[np.ndarray] = None
        self.torch_fps: Optional[torch.Tensor] = None

    def load(self, device: str = "cpu"):
        """Loads database and precomputed fingerprints into memory."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        print(f"Loading candidate database from {self.db_path}...")
        self.df = pd.read_parquet(self.db_path)
        self.masses = self.df["exact_mass"].values

        if self.fp_path.exists():
            print(f"Loading precomputed fingerprints from {self.fp_path}...")
            self.fps = np.load(self.fp_path)
        else:
            print("Precomputing Morgan fingerprints for all database molecules...")
            self._build_fingerprints()

        if device == "cuda" and torch.cuda.is_available():
            self.torch_fps = torch.from_numpy(self.fps).to("cuda", dtype=torch.float32)
        else:
            self.torch_fps = torch.from_numpy(self.fps).to(dtype=torch.float32)

        print(f"CandidateDatabase loaded: {len(self.df):,} molecules indexed.")

    def _build_fingerprints(self):
        """Generates and persists 2048-bit Morgan fingerprints for all molecules."""
        fps_list = []
        for smi in self.df["smiles"]:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = _mfpgen.GetFingerprintAsNumPy(mol)
            else:
                fp = np.zeros(2048, dtype=np.uint8)
            fps_list.append(fp)

        self.fps = np.stack(fps_list, axis=0).astype(np.uint8)
        self.fp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.fp_path, self.fps)
        print(f"Saved {len(self.fps)} fingerprints to {self.fp_path}")

    def query_by_mass(self, neutral_mass: float, ppm_tol: float = 10.0, min_candidates: int = 25) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Retrieves candidate molecules matching the neutral mass within ppm_tol.
        If fewer than min_candidates are found, automatically expands tolerance window.
        Returns: (candidate_indices, candidate_df)
        """
        if self.masses is None:
            raise RuntimeError("Database not loaded. Call db.load() first.")

        tol = neutral_mass * ppm_tol * 1e-6
        i_left = np.searchsorted(self.masses, neutral_mass - tol)
        i_right = np.searchsorted(self.masses, neutral_mass + tol)

        indices = np.arange(i_left, i_right)

        # Fallback / dynamic expansion if fewer than min_candidates
        current_ppm = ppm_tol
        while len(indices) < min_candidates and current_ppm < 50.0:
            current_ppm += 10.0
            tol = neutral_mass * current_ppm * 1e-6
            i_left = np.searchsorted(self.masses, neutral_mass - tol)
            i_right = np.searchsorted(self.masses, neutral_mass + tol)
            indices = np.arange(i_left, i_right)

        return indices, self.df.iloc[indices]
