import pandas as pd

test = pd.read_parquet('data_set/test.parquet')

adduct_shifts = {
    '[M+H]+': -1.007276,
    '[M-H]-': +1.007276,
    '[M+Na]+': -22.989218,
    '[M+K]+': -38.963158,
    '[M+NH4]+': -18.033823,
    '[M+CH2O2-H]-': -44.998201,
    '[M+Cl]-': -34.969402,
}

for mol_id in ['m_005e53', 'm_006153', 'm_0259d4', 'm_068354']:
    sub = test[test['molecule_id'] == mol_id][['spectrum_id', 'adduct', 'ionization_mode', 'precursor_mz', 'collision_energy_ev']]
    print(f"\n--- Molecule: {mol_id} ---")
    estimates = []
    for _, r in sub.iterrows():
        add = r['adduct']
        shift = adduct_shifts.get(add, 0)
        neutral = r['precursor_mz'] + shift
        estimates.append(neutral)
        print(f"  {r['spectrum_id']} | {add:<14} | {r['ionization_mode']:<8} | mz={r['precursor_mz']:.5f} -> neutral={neutral:.5f}")
    diff = max(estimates) - min(estimates)
    print(f"  Max neutral mass difference: {diff:.6f} Da ({diff / estimates[0] * 1e6:.2f} ppm)")
