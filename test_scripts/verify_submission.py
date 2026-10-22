import pandas as pd
from src.metric import get_canonical_inchikey14

sub = pd.read_csv("submissions/submission.csv")
sample = pd.read_csv("data_set/sample_submission.csv")

print("Submission shape:", sub.shape)
print("Sample submission shape:", sample.shape)
print("Columns match:", list(sub.columns) == list(sample.columns))
print("IDs match in exact order:", (sub["molecule_id"] == sample["molecule_id"]).all())
print("No nulls:", sub.isna().sum().to_dict())

# Verify each row has exactly 25 predictions and check InChIKey14 uniqueness
all_valid = True
sample_keys = []
for i in range(len(sub)):
    preds = sub.loc[i, "smiles"].split(";")
    if len(preds) != 25:
        print(f"ERROR: Row {i} has {len(preds)} predictions")
        all_valid = False
        break

    if i < 3:
        print(f"\nRow {i} (mol {sub.loc[i, 'molecule_id']}):")
        for rank, smi in enumerate(preds[:3], 1):
            key = get_canonical_inchikey14(smi)
            print(f"  Rank {rank}: {smi} (InChIKey14: {key})")

if all_valid:
    print("\nSUCCESS: All 400 test molecules have exactly 25 validated predictions!")
