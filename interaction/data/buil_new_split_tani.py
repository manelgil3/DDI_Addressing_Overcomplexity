#!/usr/bin/env python3
import argparse, json, os, glob, random
from collections import defaultdict
import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

# ───────────────────────────── RDKit helpers ───────────────────────────── #
def mol_from_smiles(smi: str):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None

def canonical_smiles(smi: str):
    m = mol_from_smiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None

def murcko_scaffold_smiles_from_mol(m):
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(scaf) if scaf is not None else None

def ecfp4_fp_from_mol(m, nBits=2048, radius=2):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
    except Exception:
        return None

def max_tanimoto_to_set(fp, fp_list):
    if fp is None or not fp_list:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, fp_list)
    return max(sims) if sims else 0.0

# ───────────────────────────── I/O helpers ───────────────────────────── #
def collect_csvs(inputs):
    paths = []
    for p in inputs:
        if os.path.isdir(p):
            paths += glob.glob(os.path.join(p, "*.csv"))
        elif os.path.isfile(p) and p.endswith(".csv"):
            paths.append(p)
        else:
            # allow globs too
            paths += glob.glob(p)
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No CSV files found in provided --inputs.")
    return paths

def load_pairs(csv_paths):
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        required = {"drug_A", "drug_B", "DDI"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"{p} missing columns: {missing}")
        frames.append(df)
    all_pairs = pd.concat(frames, ignore_index=True)
    # de-duplicate exact duplicates if present
    all_pairs = all_pairs.drop_duplicates()
    return all_pairs

# ─────────────────────────── Split construction ────────────────────────── #
def assign_molecule_splits(unique_smiles,
                           val_frac=0.10, test_frac=0.10,
                           threshold=0.70, enforce_for_val=True, seed=42):
    """
    unique_smiles: iterable of canonical SMILES strings (IDs)
    Returns:
      assign: dict[canon_smi] -> split ("train"/"val"/"test")
      scaf:   dict[canon_smi] -> murcko scaffold smiles
      fps:    dict[canon_smi] -> ECFP4 bitvect
    """
    rng = random.Random(seed)

    # Build mols, scaffolds, fps
    mols = {s: mol_from_smiles(s) for s in unique_smiles}
    scaf = {s: (murcko_scaffold_smiles_from_mol(m) if m is not None else None) for s, m in mols.items()}
    fps  = {s: (ecfp4_fp_from_mol(m) if m is not None else None) for s, m in mols.items()}

    # Bucket by scaffold (None gets its bucket)
    scaf_to_mols = defaultdict(list)
    for s, sc in scaf.items():
        scaf_to_mols[sc].append(s)

    scaf_keys = list(scaf_to_mols.keys())
    rng.shuffle(scaf_keys)

    n = len(unique_smiles)
    target_train = int(round((1.0 - val_frac - test_frac) * n))
    target_val   = int(round(val_frac * n))
    target_test  = n - target_train - target_val

    assign = {}
    train_fps, val_fps = [], []

    for sc in scaf_keys:
        group = scaf_to_mols[sc]
        group_fps = [fps[s] for s in group if fps[s] is not None]

        n_train = sum(1 for v in assign.values() if v == "train")
        n_val   = sum(1 for v in assign.values() if v == "val")
        n_test  = sum(1 for v in assign.values() if v == "test")

        placed = False

        # 1) Fill train first
        if n_train < target_train:
            for s in group: assign[s] = "train"
            train_fps.extend(group_fps)
            placed = True

        # 2) Then val (optionally enforce dissimilar to train)
        elif n_val < target_val:
            ok_vs_train = True
            if enforce_for_val and group_fps and train_fps:
                ok_vs_train = max(max_tanimoto_to_set(fp, train_fps) for fp in group_fps) <= threshold
            if ok_vs_train:
                for s in group: assign[s] = "val"
                val_fps.extend(group_fps)
                placed = True

        # 3) Finally test, enforce strict vs train (and optionally vs val)
        if not placed:
            ok_vs_train = True
            ok_vs_val = True
            if group_fps and train_fps:
                ok_vs_train = max(max_tanimoto_to_set(fp, train_fps) for fp in group_fps) <= threshold
            if enforce_for_val and group_fps and val_fps:
                ok_vs_val = max(max_tanimoto_to_set(fp, val_fps) for fp in group_fps) <= threshold

            if ok_vs_train and ok_vs_val:
                for s in group: assign[s] = "test"
            else:
                # fallback to train to satisfy constraint
                for s in group: assign[s] = "train"
                train_fps.extend(group_fps)

    return assign, scaf, fps

def build_pair_splits(all_pairs, canon_map, assign):
    """
    Keep pairs where BOTH drugs (by canonical) are in the same split.
    Returns train_df, val_df, test_df (with original columns, original SMILES).
    """
    # map original → canonical once
    def split_of_row(row):
        a = canon_map.get(row["drug_A"])
        b = canon_map.get(row["drug_B"])
        if a is None or b is None:
            return None
        sa, sb = assign.get(a), assign.get(b)
        return sa if (sa is not None and sa == sb) else None

    split_series = all_pairs.apply(split_of_row, axis=1)
    kept = all_pairs.loc[split_series.notna()].copy()
    kept["split"] = split_series[split_series.notna()].values

    train_df = kept[kept["split"]=="train"].drop(columns=["split"])
    val_df   = kept[kept["split"]=="val"].drop(columns=["split"])
    test_df  = kept[kept["split"]=="test"].drop(columns=["split"])
    return train_df, val_df, test_df

# ───────────────────────────────── Main ───────────────────────────────── #
def main():
    ap = argparse.ArgumentParser(description="Scaffold split (Murcko + ECFP4 Tanimoto ≤ 0.7) for DDI using SMILES as IDs.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="CSV files and/or directories containing CSVs with columns: drug_A, drug_B, DDI")
    ap.add_argument("--out_dir", required=True, help="Output directory for the scaffold split.")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enforce_for_val", action="store_true",
                    help="Also enforce Tanimoto ≤ threshold for VAL vs TRAIN.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csvs = collect_csvs(args.inputs)
    pairs = load_pairs(csvs)

    # Build original→canonical map for ALL distinct SMILES found
    unique_orig = pd.unique(pd.concat([pairs["drug_A"], pairs["drug_B"]], ignore_index=True))
    canon_map = {}
    invalid = []
    for s in unique_orig:
        c = canonical_smiles(str(s))
        if c is None:
            invalid.append(s)
        canon_map[s] = c

    # Drop rows with invalid SMILES
    if invalid:
        before = len(pairs)
        pairs = pairs[pairs["drug_A"].map(canon_map).notna() & pairs["drug_B"].map(canon_map).notna()].copy()
        print(f"Filtered out {before - len(pairs)} pairs due to invalid SMILES.")

    # Unique canonical SMILES set
    unique_canon = set([canon_map[s] for s in pd.unique(pd.concat([pairs["drug_A"], pairs["drug_B"]], ignore_index=True))])

    # Assign molecules to splits
    assign, scaf, fps = assign_molecule_splits(
        unique_canon,
        val_frac=args.val_frac, test_frac=args.test_frac,
        threshold=args.threshold, enforce_for_val=args.enforce_for_val, seed=args.seed
    )

    # Build pair splits with original strings preserved
    train_df, val_df, test_df = build_pair_splits(pairs, canon_map, assign)

    # Save CSVs
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    # Diagnostics: per-SMILES assignment (canonical)
    pd.DataFrame(
        [{"canonical_smiles": s,
          "split": assign.get(s, None),
          "scaffold_smiles": scaf.get(s, None)} for s in sorted(unique_canon)]
    ).to_csv(os.path.join(args.out_dir, "drug_assignments.csv"), index=False)

    stats = {
        "n_pairs_input": int(len(pairs)),
        "n_unique_canonical_smiles": int(len(unique_canon)),
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "threshold": args.threshold,
        "seed": args.seed,
        "enforce_for_val": bool(args.enforce_for_val),
        "n_pairs_train": int(len(train_df)),
        "n_pairs_val": int(len(val_df)),
        "n_pairs_test": int(len(test_df)),
        "n_mols_train": int(sum(1 for v in assign.values() if v=="train")),
        "n_mols_val":   int(sum(1 for v in assign.values() if v=="val")),
        "n_mols_test":  int(sum(1 for v in assign.values() if v=="test")),
    }
    with open(os.path.join(args.out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("✅ Saved scaffold split to:", args.out_dir)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
