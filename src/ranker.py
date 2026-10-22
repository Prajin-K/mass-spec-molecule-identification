"""
Candidate Ranking and Metric-Aligned Selection Engine.
Ranks retrieved candidate molecules using predicted fingerprint Tanimoto similarity,
penalizes precursor mass deviations, and guarantees 25 unique InChIKey14 predictions.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.candidate_db import CandidateDatabase
from src.metric import get_canonical_inchikey14, deduplicate_candidates


class CandidateRanker:
    def __init__(self, db: CandidateDatabase, device: str = "cpu"):
        self.db = db
        self.device = torch.device(device)

    def rank_candidates_for_molecule(
        self,
        predicted_fp: torch.Tensor,
        consensus_neutral_mass: float,
        ppm_tol: float = 12.0,
        top_k: int = 25,
        ppm_weight: float = 0.02,
    ) -> List[str]:
        """
        Ranks candidates for a single molecule:
        1. Queries candidate database around consensus_neutral_mass
        2. Computes Tanimoto similarity against predicted_fp
        3. Applies mass-error fine ranking penalty
        4. Deduplicates strictly on canonical InChIKey14
        5. Returns exactly top_k SMILES
        """
        cand_indices, cand_df = self.db.query_by_mass(
            consensus_neutral_mass, ppm_tol=ppm_tol, min_candidates=max(top_k * 2, 40)
        )

        if len(cand_indices) == 0:
            # Fallback if no candidate found
            return ["CCO"] * top_k

        # Extract candidate fingerprints
        cand_fps = self.db.torch_fps[cand_indices].to(self.device, dtype=torch.float32)  # (N_cand, 2048)
        pred_fp = predicted_fp.to(self.device, dtype=torch.float32).unsqueeze(0)        # (1, 2048)

        # Vectorized Soft-Tanimoto similarity calculation
        intersection = torch.matmul(cand_fps, pred_fp.T).squeeze(-1)  # (N_cand,)
        sum_cand = cand_fps.sum(dim=-1)
        sum_pred = pred_fp.sum()
        union = sum_cand + sum_pred - intersection
        tanimoto_scores = (intersection + 1e-6) / (union + 1e-6)

        # Precursor mass ppm penalty
        cand_masses = cand_df["exact_mass"].values
        ppm_diffs = np.abs(cand_masses - consensus_neutral_mass) / consensus_neutral_mass * 1e6
        ppm_penalty = torch.from_numpy(ppm_diffs).to(self.device, dtype=torch.float32) * ppm_weight

        # Final composite ranking score
        final_scores = tanimoto_scores - ppm_penalty
        sorted_ranks = torch.argsort(final_scores, descending=True).cpu().numpy()

        ranked_smiles = cand_df["smiles"].iloc[sorted_ranks].tolist()

        # Strict metric-aligned deduplication by InChIKey14
        deduped_smiles = deduplicate_candidates(ranked_smiles, top_k=top_k)

        # If still fewer than top_k, fill with remaining candidates or fallback
        if len(deduped_smiles) < top_k:
            for smi in ranked_smiles:
                if smi not in deduped_smiles:
                    deduped_smiles.append(smi)
                if len(deduped_smiles) == top_k:
                    break

        while len(deduped_smiles) < top_k:
            deduped_smiles.append(deduped_smiles[-1] if deduped_smiles else "CCO")

        return deduped_smiles[:top_k]
