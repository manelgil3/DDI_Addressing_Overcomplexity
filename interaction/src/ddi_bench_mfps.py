import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F  # Add this if not already imported
from sklearn.preprocessing import label_binarize  # Add this for multiclass binarization
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset Definition
class DDIDataset(Dataset):
    def __init__(self, dataframe, drug_embeddings):
        self.data = dataframe
        self.drug_embeddings = drug_embeddings
        # Filter dataset to include only rows with available embeddings
        self.data = self.data[self.data['drug_A'].isin(drug_embeddings.keys()) & self.data['drug_B'].isin(drug_embeddings.keys())]
        self.pairs = self.data[['drug_A', 'drug_B']].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_a = row['drug_A']
        drug_b = row['drug_B']
        label = torch.tensor(row['DDI'], dtype=torch.long)

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


def train_model(model, train_loader, val_loader, num_epochs, results_path, criterion, optimizer, name):
    model.to(device)  # Move model to GPU

    best_model = None
    best_val_accuracy = 0
    train_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for drug_a, drug_b, labels in train_loader:
            drug_a, drug_b, labels = drug_a.to(device), drug_b.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(drug_a, drug_b)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_accuracy = evaluate_model(model, val_loader)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()

            # **Save the best model after each epoch**
            model_filename = os.path.join(results_path, f"best_model_{name}.pt")
            torch.save(best_model, model_filename)
            print(f"✅ Saved Best Model: {model_filename}")

    return best_model


# Evaluation Function
def evaluate_model(model, loader):
    model.eval()
    model.to(device)  # Ensure model is on GPU
    all_labels, all_preds = [], []

    with torch.no_grad():
        for drug_a, drug_b, labels in loader:
            drug_a, drug_b, labels = drug_a.to(device), drug_b.to(device), labels.to(device)
            logits = model(drug_a, drug_b)
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

def calculate_metrics(model, loader, results_path, num_classes):
    model.eval()
    model.to(device)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for drug_a, drug_b, labels in loader:
            drug_a, drug_b, labels = drug_a.to(device), drug_b.to(device), labels.to(device)
            logits = model(drug_a, drug_b)
            probabilities = F.softmax(logits, dim=1)
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)

    auroc = roc_auc_score(label_binarize(all_labels, classes=list(range(num_classes))), all_probs, multi_class='ovr')

    aupr_list = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(
            label_binarize(all_labels, classes=list(range(num_classes)))[:, i],
            all_probs[:, i]
        )
        aupr_list.append(auc(recall, precision))

    return auroc, np.mean(aupr_list)



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

def main():
    # File Paths
    train_path_1 = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/data/unseen_ddi/tr_dataset.csv'
    val_path_1 = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/data/unseen_ddi/val_dataset.csv'
    test_path_1 = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/data/unseen_ddi/tst_dataset.csv'
    train_path_2 = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/data/unseen_drugs/tr_dataset.csv'
    val_path_2a = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/data/unseen_drugs/val_dataset_unseen_onedrug.csv'
    val_path_2b = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/data/unseen_drugs/val_dataset_unseen_twodrugs.csv'
    
    # NEW — Scaffold split paths
    scaffold_dir = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/data/scaffold_0p7'
    scaf_train_path = os.path.join(scaffold_dir, 'train.csv')
    scaf_val_path   = os.path.join(scaffold_dir, 'val.csv')
    scaf_test_path  = os.path.join(scaffold_dir, 'test.csv')
    
    embeddings_path = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/embeddings/ddi_mfps_embeddings.pt'
    results_path = '/home/mgil/ARTIBAND/data/ddi_bu/interaction/resultats_ddi/mfps/new_split'
    os.makedirs(results_path, exist_ok=True)

    num_runs = 5  # Number of times to run training & evaluation
    acc_scores, auroc_scores, aupr_scores = [], [], []
    acc_scores_2a, auroc_scores_2a, aupr_scores_2a = [], [], []
    acc_scores_2b, auroc_scores_2b, aupr_scores_2b = [], [], []
    # NEW — Accumulators for scaffold split
    acc_scores_scaf, auroc_scores_scaf, aupr_scores_scaf = [], [], []


    for run in range(num_runs):
        print(f"\n=== Run {run + 1} / {num_runs} ===")

        # Load Embeddings
        print("Loading embeddings...")
        drug_embeddings = torch.load(embeddings_path, map_location='cpu')

        
        # Load Data
        print("Loading datasets...")
        train_data_1 = pd.read_csv(train_path_1)
        val_data_1 = pd.read_csv(val_path_1)
        test_data_1 = pd.read_csv(test_path_1)

        train_data_2 = pd.read_csv(train_path_2)
        val_data_2a = pd.read_csv(val_path_2a)
        val_data_2b = pd.read_csv(val_path_2b)
        
        
        # NEW — Load scaffold split datasets
        scaf_train_df = pd.read_csv(scaf_train_path)
        scaf_val_df   = pd.read_csv(scaf_val_path)
        scaf_test_df  = pd.read_csv(scaf_test_path)

        
        train_dataset_1 = DDIDataset(train_data_1, drug_embeddings)
        val_dataset_1 = DDIDataset(val_data_1, drug_embeddings)
        test_dataset_1 = DDIDataset(test_data_1, drug_embeddings)

        train_dataset_2 = DDIDataset(train_data_2, drug_embeddings)
        val_dataset_2a = DDIDataset(val_data_2a, drug_embeddings)
        val_dataset_2b = DDIDataset(val_data_2b, drug_embeddings)
        
        
        scaf_train_dataset = DDIDataset(scaf_train_df, drug_embeddings)
        scaf_val_dataset   = DDIDataset(scaf_val_df,   drug_embeddings)
        scaf_test_dataset  = DDIDataset(scaf_test_df,  drug_embeddings)

        
        # DataLoaders
        train_loader_1 = DataLoader(train_dataset_1, batch_size=512, shuffle=True, collate_fn=collate_fn)
        val_loader_1 = DataLoader(val_dataset_1, batch_size=512, shuffle=False, collate_fn=collate_fn)
        test_loader_1 = DataLoader(test_dataset_1, batch_size=512, shuffle=False, collate_fn=collate_fn)

        train_loader_2 = DataLoader(train_dataset_2, batch_size=512, shuffle=True, collate_fn=collate_fn)
        val_loader_2a = DataLoader(val_dataset_2a, batch_size=512, shuffle=False, collate_fn=collate_fn)
        val_loader_2b = DataLoader(val_dataset_2b, batch_size=512, shuffle=False, collate_fn=collate_fn)
        
        
        scaf_train_loader = DataLoader(scaf_train_dataset, batch_size=512, shuffle=True,  collate_fn=collate_fn)
        scaf_val_loader   = DataLoader(scaf_val_dataset,   batch_size=512, shuffle=False, collate_fn=collate_fn)
        scaf_test_loader  = DataLoader(scaf_test_dataset,  batch_size=512, shuffle=False, collate_fn=collate_fn)

        # Model Setup
        num_classes = 4  # Number of classes
        model = DDINetwork(embedding_dim=2048, hidden_dim=512, output_dim=256, num_classes=num_classes, dropout_rate=0.3)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Move model to GPU
        model.to(device)

        num_epochs = 200

        
        # Train on Set 1
        print(f"Training on Set 1... (Run {run + 1})")
        best_model_1 = train_model(model, train_loader_1, val_loader_1, num_epochs, results_path, criterion, optimizer, f'set1_run{run + 1}')
        model.load_state_dict(best_model_1)

        # Save trained model for Set 1
        model_save_path_set1 = os.path.join(results_path, f"final_model_set1_run{run + 1}.pt")
        torch.save(model.state_dict(), model_save_path_set1)
        print(f"✅ Saved Final Model (Set 1): {model_save_path_set1}")

        # Evaluate on Test Set 1
        print(f"Evaluating on Test Set 1... (Run {run + 1})")
        accuracy_1 = evaluate_model(model, test_loader_1)
        auroc_1, aupr_1 = calculate_metrics(model, test_loader_1, results_path, num_classes=num_classes)

        acc_scores.append(accuracy_1)
        auroc_scores.append(auroc_1)
        aupr_scores.append(aupr_1)

        print(f"Run {run + 1} - Set 1 Accuracy: {accuracy_1:.4f}, AUROC: {auroc_1:.4f}, AUPR: {aupr_1:.4f}")

        # ------------------------------------------------------------------------------------------------------------------------

        model = DDINetwork(embedding_dim=2048, hidden_dim=512, output_dim=256, num_classes=num_classes, dropout_rate=0.3)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # ------------------------------------------------------------------------------------------------------------------------

        # Train on Set 2
        print(f"Training on Set 2... (Run {run + 1})")
        best_model_2 = train_model(model, train_loader_2, val_loader_2a, num_epochs, results_path, criterion, optimizer, f'set2_run{run + 1}')
        model.load_state_dict(best_model_2)

        # Save trained model for Set 2
        model_save_path_set2 = os.path.join(results_path, f"final_model_set2_run{run + 1}.pt")
        torch.save(model.state_dict(), model_save_path_set2)
        print(f"✅ Saved Final Model (Set 2): {model_save_path_set2}")

        # Evaluate on Validation Sets 2
        print(f"Evaluating on Validation Sets 2... (Run {run + 1})")
        auroc_2a, aupr_2a = calculate_metrics(model, val_loader_2a, results_path, num_classes=num_classes)
        accuracy_2 = evaluate_model(model, val_loader_2a)

        auroc_2b, aupr_2b = calculate_metrics(model, val_loader_2b, results_path, num_classes=num_classes)
        accuracy_3 = evaluate_model(model, val_loader_2b)

        print(f"Run {run + 1} - Set 2 Accuracy (One Drug): {accuracy_2:.4f}, AUROC: {auroc_2a:.4f}, AUPR: {aupr_2a:.4f}")
        print(f"Run {run + 1} - Set 2 Accuracy (Two Drugs): {accuracy_3:.4f}, AUROC: {auroc_2b:.4f}, AUPR: {aupr_2b:.4f}")
        
        acc_scores_2a.append(accuracy_2)
        auroc_scores_2a.append(auroc_2a)
        aupr_scores_2a.append(aupr_2a)

        acc_scores_2b.append(accuracy_3)
        auroc_scores_2b.append(auroc_2b)
        aupr_scores_2b.append(aupr_2b)
        
        
        # ------------------------------------------------------------------------------------------------------------------------
        
        # NEW — Train on Scaffold split
        model = DDINetwork(embedding_dim=2048, hidden_dim=512, output_dim=256, num_classes=num_classes, dropout_rate=0.3)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # ------------------------------------------------------------------------------------------------------------------------

        print(f"Training on Scaffold split... (Run {run + 1})")
        best_model_scaf = train_model(model, scaf_train_loader, scaf_val_loader, num_epochs, results_path, criterion, optimizer, f'scaffold_run{run + 1}')
        model.load_state_dict(best_model_scaf)

        # Save trained model
        model_save_path_scaf = os.path.join(results_path, f"final_model_scaffold_run{run + 1}.pt")
        torch.save(model.state_dict(), model_save_path_scaf)
        print(f"✅ Saved Final Model (Scaffold): {model_save_path_scaf}")

        # Evaluate on Scaffold test
        acc_scaf = evaluate_model(model, scaf_test_loader)
        auroc_scaf, aupr_scaf = calculate_metrics(model, scaf_test_loader, results_path, num_classes=num_classes)
        print(f"Run {run + 1} - Scaffold Test Accuracy: {acc_scaf:.4f}, AUROC: {auroc_scaf:.4f}, AUPR: {aupr_scaf:.4f}")
        
        acc_scores_scaf.append(acc_scaf)
        auroc_scores_scaf.append(auroc_scaf)
        aupr_scores_scaf.append(aupr_scaf)

    
    # Compute Mean & Std of Metrics
    acc_mean, acc_std = np.mean(acc_scores), np.std(acc_scores)
    auroc_mean, auroc_std = np.mean(auroc_scores), np.std(auroc_scores)
    aupr_mean, aupr_std = np.mean(aupr_scores), np.std(aupr_scores)

    acc_mean_2a, acc_std_2a = np.mean(acc_scores_2a), np.std(acc_scores_2a)
    auroc_mean_2a, auroc_std_2a = np.mean(auroc_scores_2a), np.std(auroc_scores_2a)
    aupr_mean_2a, aupr_std_2a = np.mean(aupr_scores_2a), np.std(aupr_scores_2a)

    acc_mean_2b, acc_std_2b = np.mean(acc_scores_2b), np.std(acc_scores_2b)
    auroc_mean_2b, auroc_std_2b = np.mean(auroc_scores_2b), np.std(auroc_scores_2b)
    aupr_mean_2b, aupr_std_2b = np.mean(aupr_scores_2b), np.std(aupr_scores_2b)
    
    
    # NEW — Scaffold split summary
    acc_mean_scaf, acc_std_scaf = np.mean(acc_scores_scaf), np.std(acc_scores_scaf)
    auroc_mean_scaf, auroc_std_scaf = np.mean(auroc_scores_scaf), np.std(auroc_scores_scaf)
    aupr_mean_scaf, aupr_std_scaf = np.mean(aupr_scores_scaf), np.std(aupr_scores_scaf)

    print(f"Scaffold Test - Accuracy: {acc_mean_scaf:.4f} ± {acc_std_scaf:.4f}, "
          f"AUROC: {auroc_mean_scaf:.4f} ± {auroc_std_scaf:.4f}, "
          f"AUPR: {aupr_mean_scaf:.4f} ± {aupr_std_scaf:.4f}")

    
    print("\n=== Final Results Over 5 Runs ===")
    print(f"Test Set 1 - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}, AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}")
    print(f"Validation Set 2 (One Drug) - Accuracy: {acc_mean_2a:.4f} ± {acc_std_2a:.4f}, AUROC: {auroc_mean_2a:.4f} ± {auroc_std_2a:.4f}, AUPR: {aupr_mean_2a:.4f} ± {aupr_std_2a:.4f}")
    print(f"Validation Set 2 (Two Drugs) - Accuracy: {acc_mean_2b:.4f} ± {acc_std_2b:.4f}, AUROC: {auroc_mean_2b:.4f} ± {auroc_std_2b:.4f}, AUPR: {aupr_mean_2b:.4f} ± {aupr_std_2b:.4f}")
    
    
    # Save Results to File
    with open(os.path.join(results_path, "final_metrics.txt"), "w") as f:
        f.write(f"=== Final Results Over 5 Runs ===\n")
        
        f.write(f"Test Set 1 - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}, AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}\n")
        f.write(f"Validation Set 2 (One Drug) - Accuracy: {acc_mean_2a:.4f} ± {acc_std_2a:.4f}, AUROC: {auroc_mean_2a:.4f} ± {auroc_std_2a:.4f}, AUPR: {aupr_mean_2a:.4f} ± {aupr_std_2a:.4f}\n")
        f.write(f"Validation Set 2 (Two Drugs) - Accuracy: {acc_mean_2b:.4f} ± {acc_std_2b:.4f}, AUROC: {auroc_mean_2b:.4f} ± {auroc_std_2b:.4f}, AUPR: {aupr_mean_2b:.4f} ± {aupr_std_2b:.4f}\n")
        
        f.write(f"Scaffold Test - Accuracy: {acc_mean_scaf:.4f} ± {acc_std_scaf:.4f}, "
                f"AUROC: {auroc_mean_scaf:.4f} ± {auroc_std_scaf:.4f}, "
                f"AUPR: {aupr_mean_scaf:.4f} ± {aupr_std_scaf:.4f}\n")

    print(f"✅ Final metrics saved to {os.path.join(results_path, 'final_metrics.txt')}")



if __name__ == "__main__":
    main()

