import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset Definition
class DDIDatasetRegression(Dataset):
    def __init__(self, dataframe, drug_embeddings):
        self.data = dataframe
        self.drug_embeddings = drug_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_a = row['drug_A']
        drug_b = row['drug_B']
        label = torch.tensor(np.log2(row['AUC FC']), dtype=torch.float32)  # Use log2(AUC FC) as target

        if drug_a not in self.drug_embeddings or drug_b not in self.drug_embeddings:
            return None

        drug_a_embedding = self.drug_embeddings[drug_a]
        drug_b_embedding = self.drug_embeddings[drug_b]

        return torch.tensor(drug_a_embedding, dtype=torch.float32), \
               torch.tensor(drug_b_embedding, dtype=torch.float32), label

# Model Definition
class DDINetworkRegression(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(DDINetworkRegression, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(2 * output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, drug_a_embedding, drug_b_embedding):
        encoded_a = self.encoder(drug_a_embedding)
        encoded_b = self.encoder(drug_b_embedding)
        combined = torch.cat((encoded_a, encoded_b), dim=1)
        output = self.regressor(combined)
        return output

# Collate Function for DataLoader
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    drug_a, drug_b, labels = zip(*batch)
    return torch.stack(drug_a), torch.stack(drug_b), torch.tensor(labels)

def train_model(model, train_loader, val_loader, num_epochs, results_path, criterion, optimizer, device, name):
    best_model = None
    best_val_loss = float('inf')

    # Tracking variables
    epoch_numbers = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for drug_a, drug_b, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            drug_a, drug_b, labels = drug_a.to(device), drug_b.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(drug_a, drug_b).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        epoch_numbers.append(epoch + 1)

        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

        # Validate
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    # Save plots
    save_plot(epoch_numbers, train_losses, "Epoch", "Loss", "Training Loss", os.path.join(results_path, f"training_loss_{name}.png"))
    save_plot(epoch_numbers, val_losses, "Epoch", "Loss", "Validation Loss", os.path.join(results_path, f"validation_loss_{name}.png"))

    return best_model

# Evaluation Function
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for drug_a, drug_b, labels in loader:
            drug_a, drug_b, labels = drug_a.to(device), drug_b.to(device), labels.to(device)
            predictions = model(drug_a, drug_b).squeeze()
            loss = criterion(predictions, labels)
            total_loss += loss.item()

    return total_loss / len(loader)

# Test Metrics Calculation
def test_metrics(model, loader, device, results_path):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for drug_a, drug_b, labels in loader:
            drug_a, drug_b, labels = drug_a.to(device), drug_b.to(device), labels.to(device)
            predictions = model(drug_a, drug_b).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    pearson_corr, _ = pearsonr(all_labels, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))

    # Plot Predicted vs Ground Truth
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_predictions, alpha=0.5, label="Predictions")
    plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', label="Ideal Fit")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Ground Truth")
    plt.xlim(-6,5)
    plt.ylim(-6,5)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, "predicted_vs_ground_truth.png"))
    plt.close()

    return pearson_corr, rmse

# Save plot function
def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Main Function
def main():
    # File Paths
    train_path = '../data/tr_datset_regression_fold1.csv'
    val_path = '../data/val_datset_regression_fold1.csv'
    test_path = '../data/PKDDI_info_2023_FDA_external.csv'
    embeddings_path = '../embeddings/dda_molformer_embeddings.pt'
    results_path = '../results/molf'

    # Load Embeddings
    print("Loading embeddings...")
    drug_embeddings = torch.load(embeddings_path, map_location='cpu')

    # Load Data
    print("Loading datasets...")
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    train_dataset = DDIDatasetRegression(train_data, drug_embeddings)
    val_dataset = DDIDatasetRegression(val_data, drug_embeddings)
    test_dataset = DDIDatasetRegression(test_data, drug_embeddings)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    # Define Number of Runs
    num_runs = 5
    pcc_scores = []
    rmse_scores = []

    for run in range(num_runs):
        print(f"\n=== Run {run + 1} / {num_runs} ===")

        # Model Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DDINetworkRegression(embedding_dim=768, hidden_dim=1024, output_dim=512, dropout_rate=0.1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.MSELoss()

        num_epochs = 200

        # Train and Evaluate
        print("Training...")
        best_model = train_model(model, train_loader, val_loader, num_epochs, results_path, criterion, optimizer, device, f'regression_run{run + 1}')
        model.load_state_dict(best_model)

        print("Evaluating on Test Set...")
        pearson_corr, rmse = test_metrics(model, test_loader, device, results_path)
        
        pcc_scores.append(pearson_corr)
        rmse_scores.append(rmse)

        print(f"Run {run + 1} - Pearson Correlation: {pearson_corr:.4f}, RMSE: {rmse:.4f}")

    # Compute Mean ± Std of Metrics
    pcc_mean, pcc_std = np.mean(pcc_scores), np.std(pcc_scores)
    rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)

    # Print Final Results
    print("\n=== Final Results Over 5 Runs ===")
    print(f"Pearson Correlation: {pcc_mean:.4f} ± {pcc_std:.4f}")
    print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")

    # Save Results to File
    with open(os.path.join(results_path, "final_metrics.txt"), "w") as f:
        f.write(f"Pearson Correlation: {pcc_mean:.4f} ± {pcc_std:.4f}\n")
        f.write(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}\n")

if __name__ == "__main__":
    main()

