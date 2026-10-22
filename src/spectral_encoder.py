"""
High-Resolution Spectral Feature Extractor:
Constructs a dual-channel spectral representation:
Channel 0: Fragment ion m/z binned intensities
Channel 1: Neutral loss (precursor_mz - fragment_mz) binned intensities
"""

from typing import List, Tuple, Union
import numpy as np
import torch


class SpectralEncoder:
    def __init__(self, max_mz: float = 1000.0, bin_width: float = 0.5):
        self.max_mz = max_mz
        self.bin_width = bin_width
        self.num_bins = int(np.ceil(max_mz / bin_width))

    def encode_spectrum(
        self,
        mzs: Union[List[float], np.ndarray],
        intensities: Union[List[float], np.ndarray],
        precursor_mz: float,
    ) -> np.ndarray:
        """
        Encodes a single spectrum into a (2, num_bins) float32 array.
        Channel 0: Fragment peaks
        Channel 1: Neutral loss peaks
        """
        spec = np.zeros((2, self.num_bins), dtype=np.float32)

        if mzs is None or len(mzs) == 0:
            return spec

        mzs = np.asarray(mzs, dtype=np.float32)
        intensities = np.asarray(intensities, dtype=np.float32)

        # Intensity power transform (sqrt) to enhance diagnostic fragment peaks
        transformed_int = np.sqrt(np.clip(intensities, 0.0, None))

        # Channel 0: Fragment ions
        frag_bins = np.floor(mzs / self.bin_width).astype(np.int64)
        valid_frag = (frag_bins >= 0) & (frag_bins < self.num_bins)

        for b, val in zip(frag_bins[valid_frag], transformed_int[valid_frag]):
            if val > spec[0, b]:
                spec[0, b] = val

        # Channel 1: Neutral losses (precursor_mz - peak_mz)
        if precursor_mz is not None and precursor_mz > 0:
            nl_mzs = precursor_mz - mzs
            nl_bins = np.floor(nl_mzs / self.bin_width).astype(np.int64)
            valid_nl = (nl_bins >= 0) & (nl_bins < self.num_bins)

            for b, val in zip(nl_bins[valid_nl], transformed_int[valid_nl]):
                if val > spec[1, b]:
                    spec[1, b] = val

        # Normalize each channel by its maximum peak
        max0 = spec[0].max()
        if max0 > 0:
            spec[0] /= max0

        max1 = spec[1].max()
        if max1 > 0:
            spec[1] /= max1

        return spec


# Common adducts mapping for neural network embedding
ADDUCT_VOCAB = {
    '[M+H]+': 0,
    '[M-H]-': 1,
    '[M+Na]+': 2,
    '[M+K]+': 3,
    '[M+NH4]+': 4,
    '[M+CH2O2-H]-': 5,
    '[M+HCOO]-': 5,
    '[M+Cl]-': 6,
    '[2M+H]+': 7,
    '[2M+Na]+': 8,
    '[2M-H]-': 9,
    '[M-H2O+H]+': 10,
    'UNKNOWN': 11,
}
