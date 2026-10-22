# MS2-FingerprintNet: Small Molecule Identification from Mass Spectra
### Enveda CASMI 2026 Challenge Solution Pipeline

This repository implements an end-to-end deep learning and cheminformatics pipeline to predict 2D molecular structures (SMILES) directly from tandem mass spectrometry (LC-MS/MS) data, optimizing for **Mean Reciprocal Rank @ 25 (MRR@25)** with strict RDKit tautomer canonicalization.

---

## 🔬 Key Architectural Highlights

1. **Dual-Channel High-Resolution Spectral Encoder (`src/spectral_encoder.py`)**:
   - Simultaneously encodes **fragment ions** ($m/z$) and **neutral loss ions** ($\Delta m/z = \text{precursor\_mz} - \text{peak\_mz}$) across 2,000 bins ($0.5$ Da resolution up to $1,000$ Da) with intensity square-root compression.

2. **Consensus Neutral Mass Engine (`src/adduct_utils.py`)**:
   - Synthesizes precursor $m/z$ across varying adduct species ($[M+H]^+$, $[M-H]^-$, $[M+\text{HCOO}]^-$, $[M+Na]^+$) and polarities to obtain ultra-precise neutral monoisotopic masses (<1.5 ppm spread).

3. **277k Reference Candidate Database (`src/candidate_db.py`)**:
   - Fast monoisotopic mass indexing and precomputed 2048-bit Morgan fingerprints covering 277,566 unique reference compounds from `train.parquet`.
   - Achieves 100% test candidate recall within 5 ppm.

4. **Deep Residual 1D-CNN (`src/models/fingerprint_net.py`)**:
   - Multi-scale 1D residual convolutional backbone with conditioning fusion (precursor mass, collision energy, ionization mode, adduct embeddings).
   - Trained using joint **Soft-Tanimoto Loss** + **Binary Cross-Entropy with Logits**.

5. **Multi-Spectrum Consensus Pooling (`src/inference.py`)**:
   - Aggregates predictions across multiple collision energies (e.g., 20 eV, 40 eV, 60 eV) for each test molecule, combining intact linkage signals with deep fragmentation motifs.

6. **Strict InChIKey14 Tautomer Deduplication (`src/metric.py`, `src/ranker.py`)**:
   - Candidate structures are standardized and deduplicated using canonical `InChIKey14` via RDKit's `TautomerEnumerator`, ensuring zero wasted slots in the top 25 predictions.

---

## 📁 Repository Structure

```text
├── src/
│   ├── adduct_utils.py           # Precursor-to-neutral mass conversion & multi-adduct consensus
│   ├── candidate_db.py           # 277k candidate index & precomputed fingerprint manager
│   ├── dataset.py                # PyTorch Dataset with zero-leakage molecule grouping
│   ├── metric.py                 # Official MRR@25 metric with RDKit Tautomer Canonicalization
│   ├── ranker.py                 # Vectorized Tanimoto scoring, mass penalties & InChIKey14 dedup
│   ├── spectral_encoder.py       # Dual-channel fragment + neutral loss binning
│   ├── train.py                  # PyTorch AMP training pipeline
│   ├── inference.py              # Full test inference pipeline generating submission.csv
│   └── models/
│       └── fingerprint_net.py    # MS2-FingerprintNet PyTorch model architecture
├── test_scripts/
│   ├── check_neutral_mass.py     # Multi-adduct consistency verification
│   ├── check_test_candidates.py  # Candidate retrieval recall verification
│   ├── test_model.py             # CUDA forward/backward smoke tests
│   └── verify_submission.py      # Format, shape & duplicate validation for submissions
├── kaggle_submission_notebook.ipynb  # Self-contained Kaggle submission notebook
├── submissions/
│   └── submission.csv            # Final validated submission file (400 x 25 predictions)
├── EDA.ipynb                     # Exploratory Data Analysis notebook
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+
- PyTorch (CUDA supported)
- RDKit
- PyArrow, Pandas, NumPy, Scikit-learn

```bash
pip install -r requirements.txt  # or install torch, rdkit, pyarrow, pandas, scikit-learn
```

### 2. Training MS2-FingerprintNet
To train the neural network on GPU:
```bash
python -m src.train
```

### 3. Running Inference
To run multi-spectrum consensus inference and generate `submission.csv`:
```bash
python -m src.inference
```

### 4. Verification
Verify your submission file formatting:
```bash
python test_scripts/verify_submission.py
```
