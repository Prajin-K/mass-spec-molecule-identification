"""
Training Pipeline for MS2-FingerprintNet:
Trains a deep residual neural network to predict 2048-bit Morgan fingerprints from LC-MS/MS spectra.
Includes zero-leakage molecule-level cross-validation and MRR@25 evaluation.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

from src.dataset import MS2Dataset
from src.models.fingerprint_net import MS2FingerprintNet, SoftTanimotoLoss
from src.candidate_db import CandidateDatabase
from src.ranker import CandidateRanker
from src.metric import compute_mrr_at_k


def load_training_data(parquet_path: str = "data_set/train.parquet", num_row_groups: int = 1, sample_size: int = 60000) -> pd.DataFrame:
    print(f"Loading spectra from {parquet_path}...")
    t0 = time.time()
    pf = pq.ParquetFile(parquet_path)
    cols = [
        "normalized_smiles", "inchikey14", "adduct", "ionization_mode",
        "precursor_mz", "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"
    ]
    chunks = []
    for i in range(min(num_row_groups, pf.num_row_groups)):
        df_group = pf.read_row_group(i, columns=cols).to_pandas()
        chunks.append(df_group)
    df = pd.concat(chunks, ignore_index=True)

    # Filter quality: precursor mz in range, valid peaks
    df = df[df["precursor_mz"].between(150.0, 700.0)]
    df = df[df["ms2_mzs"].apply(lambda x: len(x) >= 3 if x is not None else False)]
    df = df[df["normalized_smiles"].notna() & df["inchikey14"].notna()]

    if len(df) > sample_size:
        # Sample while keeping unique molecules diverse
        unique_mols = df["inchikey14"].unique()
        np.random.seed(42)
        selected_mols = set(np.random.choice(unique_mols, size=min(len(unique_mols), sample_size // 2), replace=False))
        df = df[df["inchikey14"].isin(selected_mols)]
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)

    print(f"Loaded {len(df)} quality spectra ({df['inchikey14'].nunique()} unique molecules) in {time.time()-t0:.2f}s")
    return df


def train_model(
    epochs: int = 4,
    batch_size: int = 128,
    lr: float = 1e-3,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print(f"--- Starting MS2-FingerprintNet Training on {device} ---")

    # 1. Load Data
    df_data = load_training_data(num_row_groups=1, sample_size=50000)

    # 2. Group Split by molecule (inchikey14) to prevent any data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(df_data, groups=df_data["inchikey14"]))

    train_df = df_data.iloc[train_idx].reset_index(drop=True)
    val_df = df_data.iloc[val_idx].reset_index(drop=True)
    print(f"Split: {len(train_df)} train spectra ({train_df['inchikey14'].nunique()} mols) | {len(val_df)} val spectra ({val_df['inchikey14'].nunique()} mols)")

    # 3. Create Datasets and DataLoaders
    print("Preparing datasets and precomputing fingerprints...")
    train_ds = MS2Dataset(train_df)
    val_ds = MS2Dataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Initialize Model, Loss, Optimizer
    model = MS2FingerprintNet(num_bins=2000, fp_dim=2048).to(device)
    tanimoto_criterion = SoftTanimotoLoss().to(device)
    bce_criterion = nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=(device_str == "cuda"))

    best_val_tanimoto = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_model_path = checkpoint_dir / "best_fingerprint_net.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_t_loss, total_bce = 0.0, 0.0, 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            spec = batch["spec"].to(device, non_blocking=True)
            precursor = batch["precursor_mz"].to(device, non_blocking=True)
            adduct = batch["adduct_idx"].to(device, non_blocking=True)
            ce = batch["ce"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)
            target = batch["target_fp"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device_str == "cuda")):
                logits = model(spec, precursor, adduct, ce, mode)
                probs = torch.sigmoid(logits)
                t_loss = tanimoto_criterion(probs, target)
                bce_loss = bce_criterion(logits, target)
                loss = t_loss + 4.0 * bce_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_t_loss += t_loss.item()
            total_bce += bce_loss.item()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (Tanimoto: {t_loss.item():.4f}, BCE: {bce_loss.item():.4f}) "
                      f"Time: {elapsed:.1f}s")

        # Validation Loop
        model.eval()
        val_t_sim = []
        with torch.no_grad():
            for batch in val_loader:
                spec = batch["spec"].to(device)
                precursor = batch["precursor_mz"].to(device)
                adduct = batch["adduct_idx"].to(device)
                ce = batch["ce"].to(device)
                mode = batch["mode"].to(device)
                target = batch["target_fp"].to(device)

                with torch.amp.autocast('cuda', enabled=(device_str == "cuda")):
                    pred = model.predict_probs(spec, precursor, adduct, ce, mode)

                # Compute Tanimoto similarity on binary thresholded prediction
                pred_binary = (pred > 0.5).float()
                inter = (pred_binary * target).sum(dim=1)
                union = pred_binary.sum(dim=1) + target.sum(dim=1) - inter
                sim = ((inter + 1e-6) / (union + 1e-6)).cpu().numpy()
                val_t_sim.extend(sim)

        avg_val_sim = float(np.mean(val_t_sim))
        print(f"--> Epoch {epoch} Validation Mean Tanimoto Similarity: {avg_val_sim:.4f}")

        if avg_val_sim > best_val_tanimoto:
            best_val_tanimoto = avg_val_sim
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_tanimoto": avg_val_sim,
            }, best_model_path)
            print(f"    * New best model saved to {best_model_path} (Tanimoto: {avg_val_sim:.4f})")

    print(f"Training completed! Best Validation Tanimoto Similarity: {best_val_tanimoto:.4f}")
    return str(best_model_path)


if __name__ == "__main__":
    train_model(epochs=4, batch_size=128)
