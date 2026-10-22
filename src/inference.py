"""
Inference Pipeline:
Generates final competition submission file matching sample_submission.csv format.
Applies:
1. Consensus neutral mass estimation across multi-adduct/multi-energy spectra
2. Multi-spectrum deep fingerprint prediction & consensus pooling
3. Candidate retrieval from 277k+ reference molecules
4. Tanimoto similarity + PPM fine ranking
5. Strict InChIKey14 tautomer deduplication to maximize MRR@25
"""

import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import MS2Dataset
from src.models.fingerprint_net import MS2FingerprintNet
from src.candidate_db import CandidateDatabase
from src.ranker import CandidateRanker
from src.adduct_utils import compute_consensus_neutral_mass


def generate_submission(
    model_checkpoint: str = "checkpoints/best_fingerprint_net.pt",
    test_parquet_path: str = "data_set/test.parquet",
    sample_sub_path: str = "data_set/sample_submission.csv",
    output_path: str = "submissions/submission.csv",
    batch_size: int = 128,
    ppm_tol: float = 12.0,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    device = torch.device(device_str)
    print(f"--- Running Inference on {device} ---")
    t0 = time.time()

    # 1. Load Candidate Database
    db = CandidateDatabase()
    db.load(device=device_str)
    ranker = CandidateRanker(db, device=device_str)

    # 2. Load Test Data
    print(f"Reading test spectra from {test_parquet_path}...")
    test_df = pd.read_parquet(test_parquet_path)
    print(f"Test spectra: {len(test_df)}, Unique molecules: {test_df['molecule_id'].nunique()}")

    # 3. Compute Consensus Neutral Mass per Molecule
    print("Computing consensus neutral masses per molecule...")
    consensus_masses: Dict[str, float] = {}
    for mol_id, group in test_df.groupby("molecule_id"):
        cm, ppm_spread, n = compute_consensus_neutral_mass(group)
        consensus_masses[mol_id] = cm

    # 4. Load Trained Model
    model = MS2FingerprintNet(num_bins=2000, fp_dim=2048).to(device)
    if Path(model_checkpoint).exists():
        print(f"Loading checkpoint {model_checkpoint}...")
        chk = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(chk.get("model_state_dict", chk))
    else:
        print(f"WARNING: Checkpoint {model_checkpoint} not found! Using initialized model.")
    model.eval()

    # 5. Predict Fingerprints for all Test Spectra
    print("Predicting fingerprints across all test spectra...")
    test_ds = MS2Dataset(test_df, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            spec = batch["spec"].to(device)
            precursor = batch["precursor_mz"].to(device)
            adduct = batch["adduct_idx"].to(device)
            ce = batch["ce"].to(device)
            mode = batch["mode"].to(device)

            with torch.amp.autocast('cuda', enabled=(device_str == "cuda")):
                preds = model.predict_probs(spec, precursor, adduct, ce, mode)
            all_preds.append(preds.float().cpu())

    all_preds_tensor = torch.cat(all_preds, dim=0)  # (1213, 2048)
    print(f"Generated {len(all_preds_tensor)} spectrum fingerprint predictions.")

    # 6. Multi-Spectrum Consensus Pooling per Molecule
    print("Performing multi-spectrum consensus pooling per molecule...")
    mol_to_indices: Dict[str, List[int]] = {}
    for idx, mol_id in enumerate(test_df["molecule_id"].values):
        mol_to_indices.setdefault(mol_id, []).append(idx)

    mol_consensus_fps: Dict[str, torch.Tensor] = {}
    for mol_id, indices in mol_to_indices.items():
        sub_preds = all_preds_tensor[indices]  # (K, 2048)
        # Consensus: average predicted bit probabilities across all spectra of the molecule
        consensus_fp = sub_preds.mean(dim=0)
        mol_consensus_fps[mol_id] = consensus_fp

    # 7. Candidate Retrieval, Ranking, and Tautomer Deduplication
    print("Ranking candidates and deduplicating by InChIKey14 for all 400 test molecules...")
    sub_sample = pd.read_csv(sample_sub_path)
    test_mol_ids = sub_sample["molecule_id"].tolist()

    submission_rows = []
    t_rank_start = time.time()
    for i, mol_id in enumerate(test_mol_ids):
        pred_fp = mol_consensus_fps.get(mol_id, torch.zeros(2048))
        neutral_mass = consensus_masses.get(mol_id, 300.0)

        top25_smiles = ranker.rank_candidates_for_molecule(
            predicted_fp=pred_fp,
            consensus_neutral_mass=neutral_mass,
            ppm_tol=ppm_tol,
            top_k=25,
        )

        smiles_str = ";".join(top25_smiles)
        submission_rows.append({"molecule_id": mol_id, "smiles": smiles_str})

        if (i + 1) % 50 == 0 or (i + 1) == len(test_mol_ids):
            print(f"Processed {i+1}/{len(test_mol_ids)} molecules in {time.time()-t_rank_start:.1f}s")

    # 8. Save and Validate Submission
    sub_df = pd.DataFrame(submission_rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path} ({len(sub_df)} rows)")

    # Validation checks
    assert len(sub_df) == len(sub_sample), f"Row count mismatch: {len(sub_df)} vs {len(sub_sample)}"
    assert list(sub_df.columns) == ["molecule_id", "smiles"], f"Columns mismatch: {sub_df.columns}"
    for idx, row in sub_df.head(5).iterrows():
        preds = row["smiles"].split(";")
        assert len(preds) == 25, f"Row {idx} does not have 25 predictions: {len(preds)}"

    print(f"All validation checks PASSED in {time.time()-t0:.1f}s!")
    return output_path


if __name__ == "__main__":
    generate_submission()
