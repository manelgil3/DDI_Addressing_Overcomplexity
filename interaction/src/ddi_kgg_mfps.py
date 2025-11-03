import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import random

# Set device (Single GPU or CPU fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Dataset Definition
class DDIDataset(Dataset):
    def __init__(self, dataframe, drug_embeddings):
        self.data = dataframe
        self.drug_embeddings = drug_embeddings
        self.data = self.data[
            self.data['drug_A_smiles'].isin(drug_embeddings.keys()) & 
            self.data['drug_B_smiles'].isin(drug_embeddings.keys())
        ]
        self.pairs = self.data[['drug_A_smiles', 'drug_B_smiles']].values.tolist()
        print("Filtered dataset pairs:", len(self.pairs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_a = row['drug_A_smiles']
        drug_b = row['drug_B_smiles']
        label = torch.tensor(row['label'], dtype=torch.long)

        if drug_a not in self.drug_embeddings or drug_b not in self.drug_embeddings:
            return None

        drug_a_embedding = self.drug_embeddings[drug_a]
        drug_b_embedding = self.drug_embeddings[drug_b]

        return torch.tensor(drug_a_embedding, dtype=torch.float32), \
               torch.tensor(drug_b_embedding, dtype=torch.float32), label


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


# Collate Function for DataLoader
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    drug_a, drug_b, labels = zip(*batch)
    return torch.stack(drug_a), torch.stack(drug_b), torch.tensor(labels)


def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    best_model = None
    best_val_accuracy = 0

    model.to(device)

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
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

        val_accuracy, val_f1, val_kappa, val_loss = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Cohen’s κ: {val_kappa:.4f}, Loss: {val_loss:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()

    return best_model


# Evaluation Function
def evaluate_model(model, loader, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0

    model.to(device)

    with torch.no_grad():
        for drug_a, drug_b, labels in loader:
            drug_a, drug_b, labels = drug_a.to(device), drug_b.to(device), labels.to(device)
            logits = model(drug_a, drug_b)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    kappa = cohen_kappa_score(all_labels, all_preds)

    return accuracy, f1, kappa, avg_loss


# Main Function (Run 5 Times)
def main():
    # File Paths
    train_path = '../data/data_kgg/split_kgg/train_smiles.csv'
    val_path = '../data/data_kgg/split_kgg/valid_smiles.csv'
    test_path = '../data/data_kgg/split_kgg/test_smiles.csv'
    embeddings_path = '../embeddings/ddi_mfps_embeddings.pt'
    results_path = '../resultats_kgg/mfps'
    os.makedirs(results_path, exist_ok=True)

    # Load Embeddings
    print("Loading embeddings...")
    drug_embeddings = torch.load(embeddings_path, map_location='cpu')

    # Load Data
    train_data = pd.read_csv(train_path, names=["drug_A_smiles", "drug_B_smiles", "label"], header=None)
    val_data = pd.read_csv(val_path, names=["drug_A_smiles", "drug_B_smiles", "label"], header=None)
    test_data = pd.read_csv(test_path, names=["drug_A_smiles", "drug_B_smiles", "label"], header=None)

    train_dataset = DDIDataset(train_data, drug_embeddings)
    val_dataset = DDIDataset(val_data, drug_embeddings)
    test_dataset = DDIDataset(test_data, drug_embeddings)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    num_epochs = 100
    num_runs = 5
    accuracy_list, f1_list, kappa_list = [], [], []

    for run in range(num_runs):
        print(f"\n==== Run {run + 1}/{num_runs} ====")
        set_seed(run)

        model = DDINetwork(2048, 512, 256, 85, 0.3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_model = train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer)
        model.load_state_dict(best_model)

        acc, f1, kappa, _ = evaluate_model(model, test_loader, criterion)
        accuracy_list.append(acc)
        f1_list.append(f1)
        kappa_list.append(kappa)

    # Compute Mean & Std
    mean_std_results = f"Accuracy: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}\n" \
                       f"F1 Score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}\n" \
                       f"Cohen’s κ: {np.mean(kappa_list):.4f} ± {np.std(kappa_list):.4f}\n"

    with open(os.path.join(results_path, "final_results.txt"), "w") as f:
        f.write(mean_std_results)

    print(mean_std_results)


if __name__ == "__main__":
    main()
