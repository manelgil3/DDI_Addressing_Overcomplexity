import pandas as pd

# File path
csv_file = 'all_data.csv'

# Strings to search for
ritonavir = 'CC(C)[C@H](NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)N[C@H](C[C@H](O)[C@H](CC1=CC=CC=C1)NC(=O)OCC1=CN=CS1)CC1=CC=CC=C1'
cobiscistat = 'CC(C)C1=NC(CN(C)C(=O)N[C@@H](CCN2CCOCC2)C(=O)N[C@H](CC[C@H](CC2=CC=CC=C2)NC(=O)OCC2=CN=CS2)CC2=CC=CC=C2)=CS1'

# Load the CSV
data = pd.read_csv(csv_file)

# Normalize drug_B column (strip spaces)
data['drug_B'] = data['drug_B'].str.strip()

# Count exact matches
ritonavir_matches = (data['drug_B'] == ritonavir).sum()
cobiscistat_matches = (data['drug_B'] == cobiscistat).sum()

# Print results
print(f"Ritonavir matches: {ritonavir_matches}")
print(f"Cobiscistat matches: {cobiscistat_matches}")
