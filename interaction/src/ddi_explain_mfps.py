import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import os

# Dataset Definition
class DDIDataset(Dataset):
    def __init__(self, dataframe, drug_embeddings):
        self.data = dataframe
        self.drug_embeddings = drug_embeddings
        # Filter dataset to include only rows with available embeddings
        self.data = self.data[
            self.data['drug_A'].isin(drug_embeddings.keys()) &
            self.data['drug_B'].isin(drug_embeddings.keys())
        ]
        self.pairs = self.data[['drug_A', 'drug_B']].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_a = row['drug_A']
        drug_b = row['drug_B']
        label = torch.tensor(row['DDI'], dtype=torch.long)
        return (
            torch.tensor(self.drug_embeddings[drug_a], dtype=torch.float32),
            torch.tensor(self.drug_embeddings[drug_b], dtype=torch.float32),
            label
        )

# Model Definition
class DDINetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_classes, dropout_rate=0.3):
        super(DDINetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, drug_a_embedding, drug_b_embedding):
        encoded_a = self.encoder(drug_a_embedding)
        encoded_b = self.encoder(drug_b_embedding)
        combined = torch.cat((encoded_a, encoded_b), dim=1)
        logits = self.classifier(combined)
        return logits

    def get_relevant_bits(self, drug_a_embedding, drug_b_embedding):
        drug_a_embedding.requires_grad = True
        drug_b_embedding.requires_grad = True
        logits = self.forward(drug_a_embedding, drug_b_embedding).sum()
        logits.backward()
        return torch.abs(drug_a_embedding.grad), torch.abs(drug_b_embedding.grad)

    def trace_to_structure(self, mol, importance_scores, percentile=80):
        radius = 2
        n_bits = importance_scores.shape[1]
        fp_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=fp_info)
        scores = importance_scores.cpu().detach().numpy()[0]
        
        # Calculate threshold for importance
        bit_threshold = np.percentile(scores, percentile)
        important_bits = np.where(scores >= bit_threshold)[0]
        
        highlight_atoms = set()
        highlight_bonds = set()
        motifs = []
        
        for bit in important_bits:
            if bit in fp_info:
                for (center_atom, radius) in fp_info[bit]:
                    try:
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                        atoms_in_env = {mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx() for bond_idx in env}
                        atoms_in_env.update({mol.GetBondWithIdx(bond_idx).GetEndAtomIdx() for bond_idx in env})
                        
                        highlight_atoms.update(atoms_in_env)
                        highlight_bonds.update(env)  # Highlight bonds too
                        
                        submol = Chem.PathToSubmol(mol, env)
                        motifs.append(Chem.MolToSmiles(submol))
                    except Exception as e:
                        print(f"Error processing bit {bit}: {e}")
        
        return list(highlight_atoms), list(highlight_bonds), motifs



# Collate Function for DataLoader
def collate_fn(batch):
    drug_a, drug_b, labels = zip(*batch)
    return torch.stack(drug_a), torch.stack(drug_b), torch.tensor(labels)

# Training Function
def train_model(model, train_loader, num_epochs, criterion, optimizer, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for drug_a, drug_b, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            drug_a, drug_b, labels = drug_a.to(device), drug_b.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(drug_a, drug_b)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

    return model

# Explainability Results
def save_explainability_results(model, drug_embeddings, pairs, output_path):
    model.eval()
    results = {"ritonavir": {}, "cobicistat": {}}
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    for drug_name, smiles in zip(
        ["ritonavir", "cobicistat"],
        ["CC(C)[C@H](NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)N[C@H](C[C@H](O)[C@H](CC1=CC=CC=C1)NC(=O)OCC1=CN=CS1)CC1=CC=CC=C1",
         "CC(C)C1=NC(CN(C)C(=O)N[C@@H](CCN2CCOCC2)C(=O)N[C@H](CC[C@H](CC2=CC=CC=C2)NC(=O)OCC2=CN=CS2)CC2=CC=CC=C2)=CS1"]):
        drug_images = []
        motifs = {}
        for drug_a_id, drug_b_id in pairs:
            if drug_b_id != smiles:
                continue
            drug_a_embedding = torch.tensor(drug_embeddings[drug_a_id], dtype=torch.float32).unsqueeze(0).requires_grad_()
            drug_b_embedding = torch.tensor(drug_embeddings[drug_b_id], dtype=torch.float32).unsqueeze(0).requires_grad_()
            importance_a, importance_b = model.get_relevant_bits(drug_a_embedding, drug_b_embedding)
            mol = Chem.MolFromSmiles(drug_b_id)
            highlight_atoms, highlight_bonds, extracted_motifs = model.trace_to_structure(mol, importance_b, percentile=80)
            img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds, size=(400, 400))
            drug_images.append(img)
            for motif in extracted_motifs:
                motifs[motif] = motifs.get(motif, 0) + 1
        results[drug_name]["images"] = drug_images
        results[drug_name]["motifs"] = motifs

    # Save Images and Statistics
    for drug_name, data in results.items():
        img_path = os.path.join(output_path, "images", drug_name)
        os.makedirs(img_path, exist_ok=True)
        for idx, img in enumerate(data["images"]):
            img.save(os.path.join(img_path, f"{drug_name}_highlighted_{idx}.png"))
        with open(os.path.join(output_path, f"{drug_name}_motifs.txt"), "w") as f:
            for motif, count in sorted(data["motifs"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"{motif}: {count}\n")

# Main Function
def main():
    data_path = '../data/motifs/all_data.csv'
    embeddings_path = '../embeddings/ddi_mfps_embeddings.pt'
    output_path = '../resultats/mfps'

    print("Loading embeddings...")
    drug_embeddings = torch.load(embeddings_path, map_location='cpu')

    print("Loading datasets...")
    data = pd.read_csv(data_path)
    dataset = DDIDataset(data, drug_embeddings)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDINetwork(embedding_dim=2048, hidden_dim=512, output_dim=256, num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training model...")
    trained_model = train_model(model, train_loader, num_epochs=200, criterion=criterion, optimizer=optimizer, device=device)

    print("Saving explainability results...")
    save_explainability_results(trained_model, drug_embeddings, dataset.pairs, output_path)

if __name__ == "__main__":
    main()
