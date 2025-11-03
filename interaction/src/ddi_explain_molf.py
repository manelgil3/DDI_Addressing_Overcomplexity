import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from rdkit import Chem
from rdkit.Chem import Draw

# Dataset Definition
class DDIDataset(Dataset):
    def __init__(self, dataframe, drug_embeddings):
        self.data = dataframe
        self.drug_embeddings = drug_embeddings
        print(f"Reading dataset: {len(dataframe)} rows")
        self.data = self.data[
            self.data['drug_A'].isin(drug_embeddings.keys()) & 
            self.data['drug_B'].isin(drug_embeddings.keys())
        ]
        self.pairs = self.data[['drug_A', 'drug_B']].values.tolist()
        print("Filtered dataset pairs:", len(self.pairs))

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
        
        logits = self.forward(drug_a_embedding, drug_b_embedding)
        predicted_class = logits.argmax(dim=1)
        selected_logit = logits[0, predicted_class]
        
        selected_logit.backward()
        
        return torch.abs(drug_a_embedding.grad), torch.abs(drug_b_embedding.grad)

# Explainability Visualization
def trace_and_highlight(mol, importance_scores, percentile=80):
    importance_array = importance_scores.cpu().detach().numpy().flatten()
    valid_atoms = [i for i in range(mol.GetNumAtoms())]

    min_length = min(len(importance_array), len(valid_atoms))
    importance_array = importance_array[:min_length]

    # Compute threshold using percentile
    threshold = np.percentile(importance_array, percentile)
    highlight_atoms = np.where(importance_array >= threshold)[0]
    highlight_atoms = [int(idx) for idx in highlight_atoms if idx in valid_atoms]

    if not highlight_atoms:
        highlight_atoms = valid_atoms  # Default to all atoms if no highlights

    # Find bonds between highlighted atoms
    highlight_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in highlight_atoms and bond.GetEndAtomIdx() in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())

    img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds, size=(500, 500))
    return img, highlight_atoms


def save_explainability_results(model, drug_embeddings, pairs, output_path, ritonavir_smiles, cobicistat_smiles):
    model.eval()

    # Define molecule folders
    images_path = os.path.join(output_path, "images")
    ritonavir_folder = os.path.join(images_path, "ritonavir")
    cobicistat_folder = os.path.join(images_path, "cobicistat")
    os.makedirs(ritonavir_folder, exist_ok=True)
    os.makedirs(cobicistat_folder, exist_ok=True)

    motif_counts = {'ritonavir': {}, 'cobicistat': {}}
    count_tracker = {'ritonavir': 1, 'cobicistat': 1}

    for drug_a_id, drug_b_id in pairs:
        if drug_b_id not in [ritonavir_smiles, cobicistat_smiles]:
            continue

        try:
            drug_a_embedding = torch.tensor(drug_embeddings[drug_a_id], dtype=torch.float32).unsqueeze(0).requires_grad_()
            drug_b_embedding = torch.tensor(drug_embeddings[drug_b_id], dtype=torch.float32).unsqueeze(0).requires_grad_()

            importance_a, importance_b = model.get_relevant_bits(drug_a_embedding, drug_b_embedding)

            mol = Chem.MolFromSmiles(drug_b_id)
            if mol:
                img_b, highlighted = trace_and_highlight(mol, importance_b, percentile=80)

                # Determine correct save directory
                molecule_name = "ritonavir" if drug_b_id == ritonavir_smiles else "cobicistat"
                molecule_folder = ritonavir_folder if molecule_name == "ritonavir" else cobicistat_folder
                save_path = os.path.join(molecule_folder, f"{molecule_name}_{count_tracker[molecule_name]}.png")

                img_b.save(save_path)
                print(f"✅ Saved: {save_path}")
                count_tracker[molecule_name] += 1

                # Extract highlighted motifs
                all_substructs = []
                for atom_idx in highlighted[:3]:
                    try:
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius=1, rootedAtAtom=int(atom_idx))
                        substruct = Chem.PathToSubmol(mol, env)
                        if substruct:
                            smiles = Chem.MolToSmiles(substruct, canonical=True)
                            all_substructs.append(smiles)
                    except:
                        continue

                unique_substructs = set(all_substructs)
                for smiles in unique_substructs:
                    motif_counts[molecule_name][smiles] = motif_counts[molecule_name].get(smiles, 0) + 1

        except Exception as e:
            print(f"Error processing pair {drug_a_id}-{drug_b_id}: {str(e)}")

    with open(os.path.join(output_path, "motif_statistics.txt"), "w") as f:
        for drug_type in ['ritonavir', 'cobicistat']:
            f.write(f"\n{drug_type.capitalize()} Common Motifs:\n")
            sorted_motifs = sorted(motif_counts[drug_type].items(), key=lambda x: x[1], reverse=True)[:20]
            for motif, count in sorted_motifs:
                f.write(f"{motif}: {count} occurrences\n")

# Collate Function for DataLoader
def collate_fn(batch):
    drug_a, drug_b, labels = zip(*batch)
    return torch.stack(drug_a), torch.stack(drug_b), torch.tensor(labels)

# Training Function
def train_model(model, train_loader, num_epochs, criterion, optimizer, device):
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

    return model.state_dict()


# Main Function
def main():
    all_data_path = '../data/motifs/all_data.csv'
    embeddings_path = '../embeddings/ddi_molformer_embeddings.pt'
    results_path = '../resultats/molf'
    ritonavir_smiles = "CC(C)[C@H](NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)N[C@H](C[C@H](O)[C@H](CC1=CC=CC=C1)NC(=O)OCC1=CN=CS1)CC1=CC=CC=C1"
    cobicistat_smiles = "CC(C)C1=NC(CN(C)C(=O)N[C@@H](CCN2CCOCC2)C(=O)N[C@H](CC[C@H](CC2=CC=CC=C2)NC(=O)OCC2=CN=CS2)CC2=CC=CC=C2)=CS1"

    drug_embeddings = torch.load(embeddings_path, map_location='cpu')
    all_data = pd.read_csv(all_data_path)

    dataset = DDIDataset(all_data, drug_embeddings)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDINetwork(768, 512, 256, 4, 0.3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_model = train_model(model, data_loader, 200, criterion, optimizer, device)
    model.load_state_dict(best_model)

    save_explainability_results(model, drug_embeddings, dataset.pairs, results_path, ritonavir_smiles, cobicistat_smiles)

if __name__ == "__main__":
    main()
