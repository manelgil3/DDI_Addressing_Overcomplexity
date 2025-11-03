import pandas as pd


def count_pairs_with_ritonavir_and_cobicistat(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Define SMILES strings for ritonavir and cobicistat
    ritonavir_smiles = "CC(C)[C@H](NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)N[C@H](C[C@H](O)[C@H](CC1=CC=CC=C1)NC(=O)OCC1=CN=CS1)CC1=CC=CC=C1"
    cobicistat_smiles = "CC(C)C1=NC(CN(C)C(=O)N[C@@H](CCN2CCOCC2)C(=O)N[C@H](CC[C@H](CC2=CC=CC=C2)NC(=O)OCC2=CN=CS2)CC2=CC=CC=C2)=CS1"

    # Filter and count pairs where drug_B is ritonavir or cobicistat using isin
    target_smiles = [ritonavir_smiles, cobicistat_smiles]
    filtered_pairs = df[df['drug_B'].isin(target_smiles)]

    ritonavir_pairs = filtered_pairs[filtered_pairs['drug_B'] == ritonavir_smiles]
    cobicistat_pairs = filtered_pairs[filtered_pairs['drug_B'] == cobicistat_smiles]

    print(f"Number of pairs with ritonavir in drug_B: {len(ritonavir_pairs)}")
    print(ritonavir_pairs)

    print(f"Number of pairs with cobicistat in drug_B: {len(cobicistat_pairs)}")
    print(cobicistat_pairs)


# Example usage
csv_path = '../data/motifs/all_data.csv'
count_pairs_with_ritonavir_and_cobicistat(csv_path)