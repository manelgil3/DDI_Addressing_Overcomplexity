import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

n_bits = 2048

# File paths
csv_files = [
    "../data/unseen_ddi/tr_dataset.csv",
    "../data/unseen_ddi/tst_dataset.csv",
    "../data/unseen_ddi/val_dataset.csv",
    "../data/unseen_drugs/tr_dataset.csv",
    "../data/unseen_drugs/val_dataset_unseen_onedrug.csv",
    "../data/unseen_drugs/val_dataset_unseen_twodrugs.csv"
]

output_path = "../embeddings/ddi_mfps_embeddings.pt"

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string to a Morgan fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
    return None

# Generate Morgan fingerprints
drug_fingerprints = {}
smiles_set = set()

# Collect all unique SMILES from the CSV files
for file in csv_files:
    df = pd.read_csv(file, usecols=['drug_A', 'drug_B'])
    smiles_set.update(df['drug_A'].dropna().unique())
    smiles_set.update(df['drug_B'].dropna().unique())

# Create fingerprints for each unique SMILES
for smiles in smiles_set:
    fingerprint = smiles_to_morgan(smiles)
    if fingerprint is not None:
        # Convert the fingerprint to a tensor
        fingerprint_tensor = torch.tensor(list(fingerprint), dtype=torch.float32)
        drug_fingerprints[smiles] = fingerprint_tensor
    else:
        # Placeholder for invalid SMILES
        print(f"Invalid SMILES: {smiles}")
        drug_fingerprints[smiles] = torch.zeros(n_bits, dtype=torch.float32)

# Save the fingerprints
torch.save(drug_fingerprints, output_path)
print(f"Drug fingerprints saved to {output_path}")
