import torch
import torch.nn.functional as F
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from rdkit import Chem
from dgl import add_self_loop
import pandas as pd

# File paths
csv_files = [
    "../../data/PKDDI_info_2023_FDA_external.csv",
    "../../data/tr_datset_regression_fold1.csv",
    "../../data/val_datset_regression_fold1.csv",
    "../../data/Case_study/itraconazole_case_with_AUCFC.csv",
    "../../data/Case_study/Itraconazole.csv",
    "../../data/Case_study/paroxetine_case_with_AUCFC.csv",
    "../../data/Case_study/Paroxetine.csv"
]

output_path = "../../embeddings/dda_gcn_no_train_embeddings.pt"

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(dgl.nn.GraphConv(in_feats, hidden_feats[0], activation=F.relu))
        for i in range(1, len(hidden_feats)):
            self.layers.append(dgl.nn.GraphConv(hidden_feats[i-1], hidden_feats[i], activation=F.relu))
        self.layers.append(dgl.nn.GraphConv(hidden_feats[-1], out_feats, activation=F.relu))
        self.readout = dgl.nn.SumPooling()

    def forward(self, g):
        h = g.ndata['h']
        for layer in self.layers:
            h = layer(g, h)
        hg = self.readout(g, h)
        return hg

def process_smiles_batch(smiles_data, model):
    graphs = []
    smiles_keys = []
    for smiles in smiles_data:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("RDKit cannot parse SMILES: " + smiles)
            graph = smiles_to_bigraph(smiles=smiles, node_featurizer=CanonicalAtomFeaturizer())
            graph = add_self_loop(graph)
            graphs.append(graph)
            smiles_keys.append(smiles)
        except Exception as e:
            print(f"Failed to process SMILES: {smiles}, Error: {e}")
    
    if not graphs:  # Check if the list is empty
        return {}
    
    bg = dgl.batch(graphs)
    with torch.no_grad():
        embeddings = model(bg).detach().cpu()
    return {smiles_keys[i]: embeddings[i] for i in range(len(smiles_keys))}

# Initialize the model
model = GCN(in_feats=74, hidden_feats=[64, 64], out_feats=64)
model.eval()

# Collect all unique SMILES from the CSV files
smiles_set = set()
for file in csv_files:
    df = pd.read_csv(file, usecols=['drug_A', 'drug_B'])
    smiles_set.update(df['drug_A'].dropna().unique())
    smiles_set.update(df['drug_B'].dropna().unique())

# Batch processing
batch_size = 128  # Adjust batch size according to your system's capability
embedding_dict = {}
smiles_list = list(smiles_set)
for start in range(0, len(smiles_list), batch_size):
    end = start + batch_size
    batch_smiles = smiles_list[start:end]
    batch_embeddings = process_smiles_batch(batch_smiles, model)
    embedding_dict.update(batch_embeddings)

# Save embeddings
torch.save(embedding_dict, output_path)
print(f"Embeddings saved to {output_path}")

# Save the model
#model_path = '/home/mgil/ARTIBAND/models/gcn_conplex_chembl_all.pth'
#torch.save(model.state_dict(), model_path)
#print(f"Model saved to {model_path}")
