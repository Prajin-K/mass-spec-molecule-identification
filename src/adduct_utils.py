"""
Adduct and Exact Mass utilities for LC-MS/MS spectral matching.
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

# Precise physical atomic/ion masses (Daltons)
# H: 1.007825032, electron: 0.00054858
# H+ = 1.007276467
# Na+ = 22.989218
# K+ = 38.963158
# NH4+ = 18.033823
# HCOO- = 44.998201
# Cl- = 34.969402

ADDUCT_MASS_SHIFTS: Dict[str, float] = {
    '[M+H]+': -1.007276467,
    '[M-H]-': +1.007276467,
    '[M+Na]+': -22.989218,
    '[M+K]+': -38.963158,
    '[M+NH4]+': -18.033823,
    '[M+CH2O2-H]-': -44.998201,  # Formate adduct [M+HCOO]-
    '[M+HCOO]-': -44.998201,
    '[M+Cl]-': -34.969402,
    '[M+CH3COO]-': -59.013851,   # Acetate adduct
    '[M-H2O+H]+': +17.003289,
}


def precursor_to_neutral_mass(precursor_mz: float, adduct: str) -> Optional[float]:
    """
    Computes exact neutral monoisotopic mass from precursor m/z and adduct type.
    Handles monomers and dimeric adducts (e.g. [2M+H]+, [2M+Na]+).
    """
    if pd.isna(precursor_mz) or not adduct:
        return None

    adduct = str(adduct).strip()

    # Dimeric adducts
    if adduct == '[2M+H]+':
        return (precursor_mz - 1.007276467) / 2.0
    if adduct == '[2M+Na]+':
        return (precursor_mz - 22.989218) / 2.0
    if adduct == '[2M-H]-':
        return (precursor_mz + 1.007276467) / 2.0

    shift = ADDUCT_MASS_SHIFTS.get(adduct)
    if shift is not None:
        return precursor_mz + shift

    # Fallback to standard protonation if adduct unknown
    return precursor_mz - 1.007276467


def compute_consensus_neutral_mass(spectra_df: pd.DataFrame) -> Tuple[float, float, int]:
    """
    Given a DataFrame containing spectra for a single molecule,
    computes the high-precision consensus neutral mass across all spectra/adducts.
    Returns: (consensus_neutral_mass, ppm_spread, num_valid_estimates)
    """
    neutral_estimates: List[float] = []
    for _, row in spectra_df.iterrows():
        nm = precursor_to_neutral_mass(row['precursor_mz'], row['adduct'])
        if nm is not None and nm > 0:
            neutral_estimates.append(nm)

    if not neutral_estimates:
        return 0.0, 0.0, 0

    consensus = float(np.median(neutral_estimates))
    if len(neutral_estimates) > 1:
        spread = float(max(neutral_estimates) - min(neutral_estimates))
        ppm_spread = (spread / consensus) * 1e6
    else:
        ppm_spread = 0.0

    return consensus, ppm_spread, len(neutral_estimates)
