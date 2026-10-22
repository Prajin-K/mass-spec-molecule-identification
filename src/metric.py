"""
Official Kaggle Evaluation Metric implementation for Enveda CASMI 2026:
MRR@25 (Mean Reciprocal Rank @ 25) with RDKit Tautomer Canonicalization.
"""

from typing import List, Optional, Dict
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Global TautomerEnumerator instance
_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

# Fast memory cache for InChIKey14 lookups
_inchikey14_cache: Dict[str, Optional[str]] = {}


def get_canonical_inchikey14(smiles: str) -> Optional[str]:
    """
    Computes the 14-character InChIKey prefix after RDKit tautomer canonicalization.
    Matches the official competition evaluation logic.
    """
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
    """
    Deduplicates candidate SMILES list using their canonical InChIKey14.
    If two candidates share the same InChIKey14, only the higher-ranked one is kept.
    This guarantees every slot in top_k represents a unique molecular skeleton.
    """
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


def compute_mrr_at_k(candidate_smiles: List[str], true_smiles: str, k: int = 25) -> float:
    """
    Computes Reciprocal Rank @ k for a single sample.
    Returns 1 / rank if true molecule's canonical InChIKey14 appears in top_k (1-indexed), else 0.0.
    """
    true_key = get_canonical_inchikey14(true_smiles)
    if true_key is None:
        return 0.0

    # Ensure candidate list is deduplicated by InChIKey14
    deduped = deduplicate_candidates(candidate_smiles, top_k=k)

    for rank, smi in enumerate(deduped[:k], start=1):
        cand_key = get_canonical_inchikey14(smi)
        if cand_key == true_key:
            return 1.0 / rank

    return 0.0


def evaluate_predictions(predictions: Dict[str, List[str]], ground_truth: Dict[str, str], k: int = 25) -> float:
    """
    Evaluates MRR@k across a dictionary of {molecule_id: [candidate_smiles, ...]}
    and ground truth {molecule_id: true_smiles}.
    """
    scores = []
    for mol_id, true_smi in ground_truth.items():
        preds = predictions.get(mol_id, [])
        score = compute_mrr_at_k(preds, true_smi, k=k)
        scores.append(score)

    return float(pd.Series(scores).mean()) if scores else 0.0
