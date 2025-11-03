import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import numpy as np

# File paths
csv_files = [
    "../data/unseen_ddi/tr_dataset.csv",
    "../data/unseen_ddi/tst_dataset.csv",
    "../data/unseen_ddi/val_dataset.csv",
    "../data/unseen_drugs/tr_dataset.csv",
    "../data/unseen_drugs/val_dataset_unseen_onedrug.csv",
    "../data/unseen_drugs/val_dataset_unseen_twodrugs.csv"
]

output_path = "../embeddings/ddi_gcn_embeddings.pt"

# Model parameters
model_path = '../models/gcn_model_alldata_train_50e.pth'
in_feats = 74
hidden_feats = [64, 64]
out_feats = 1328
batch_size = 256

# GCN Model class
def define_gcn():
    class GCN(nn.Module):
        def __init__(self, in_feats, hidden_feats, out_feats):
            super(GCN, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(dgl.nn.GraphConv(in_feats, hidden_feats[0], activation=F.relu))
            for i in range(1, len(hidden_feats)):
                self.layers.append(dgl.nn.GraphConv(hidden_feats[i - 1], hidden_feats[i], activation=F.relu))
            self.layers.append(dgl.nn.GraphConv(hidden_feats[-1], out_feats))
            self.readout = dgl.nn.SumPooling()

        def forward(self, batched_graph):
            features = batched_graph.ndata['h']
            h = features
            for i, layer in enumerate(self.layers):
                h = layer(batched_graph, h)
                if i == len(self.layers) - 2:
                    last_hidden_layer_output = h
            hg = self.readout(batched_graph, last_hidden_layer_output)
            return hg

    return GCN

# Load the model
def load_model(model_path, in_feats, hidden_feats, out_feats):
    GCN = define_gcn()
    model = GCN(in_feats=in_feats, hidden_feats=hidden_feats, out_feats=out_feats)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Prepare graphs from SMILES
def prepare_graphs(smiles_list):
    graphs = []
    valid_smiles = []
    failed_smiles = []

    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            graph = smiles_to_bigraph(smiles=smiles, node_featurizer=CanonicalAtomFeaturizer())
            graph = dgl.add_self_loop(graph)
            graphs.append(graph)
            valid_smiles.append(smiles)
        except Exception as e:
            failed_smiles.append((smiles, str(e)))

    print(f"Processed {len(graphs)} SMILES. Failed to process {len(failed_smiles)} SMILES.")
    return graphs, valid_smiles, failed_smiles

# Generate embeddings
def generate_embeddings(model, graphs, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(graphs), batch_size), desc="Batch Processing"):
        batch_graphs = dgl.batch(graphs[i:i + batch_size])
        with torch.no_grad():
            batch_embeddings = model(batch_graphs)
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

# Main processing
if __name__ == "__main__":
    # Load and combine unique SMILES from all files
    smiles_set = set()
    for file in csv_files:
        df = pd.read_csv(file, usecols=['drug_A', 'drug_B'])
        smiles_set.update(df['drug_A'].dropna().unique())
        smiles_set.update(df['drug_B'].dropna().unique())

    print(f"Number of unique SMILES: {len(smiles_set)}")

    # Prepare graphs
    graphs, valid_smiles, failed_smiles = prepare_graphs(list(smiles_set))

    # Load the model
    model = load_model(model_path, in_feats, hidden_feats, out_feats)

    # Generate embeddings
    embeddings = generate_embeddings(model, graphs, batch_size)

    # Create a dictionary of SMILES to embeddings
    embeddings_dict = {smiles: embedding for smiles, embedding in zip(valid_smiles, embeddings)}

    # Save embeddings
    torch.save(embeddings_dict, output_path)
    print(f"Embeddings saved to {output_path}")

    # Report failed SMILES
    if failed_smiles:
        print("Failed SMILES:")
        for smiles, error in failed_smiles:
            print(f"{smiles}: {error}")
