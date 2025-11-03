import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem

# File paths
csv_files = [
    "../data/unseen_ddi/tr_dataset.csv",
    "../data/unseen_ddi/tst_dataset.csv",
    "../data/unseen_ddi/val_dataset.csv",
    "../data/unseen_drugs/tr_dataset.csv",
    "../data/unseen_drugs/val_dataset_unseen_onedrug.csv",
    "../data/unseen_drugs/val_dataset_unseen_twodrugs.csv"
]

output_path = "../embeddings/ddi_molformer_embeddings.pt"

# Load tokenizer and model from Hugging Face
model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
model.eval()

# Function to validate SMILES
def is_valid_smiles(smiles):
    """Validate if a SMILES string can be parsed by RDKit."""
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception as e:
        print(f"Error validating SMILES '{smiles}': {e}")
        return False

# Collect all unique SMILES from the CSV files
smiles_set = set()
for file in csv_files:
    df = pd.read_csv(file, usecols=['drug_A', 'drug_B'])
    smiles_set.update(df['drug_A'].dropna().unique())
    smiles_set.update(df['drug_B'].dropna().unique())

# Filter out invalid SMILES
valid_smiles = {smiles for smiles in smiles_set if is_valid_smiles(smiles)}

# Dictionary to store embeddings
embeddings = {}

# Process each SMILES
for smiles in valid_smiles:
    # Tokenize and get embeddings
    inputs = tokenizer(smiles, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output.squeeze().cpu()  # Use `pooler_output` for sentence-level embedding

    embeddings[smiles] = embedding

# Save embeddings dictionary
torch.save(embeddings, output_path)
print(f"Embeddings saved to {output_path}")
