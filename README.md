# DDI Benchmarks — Reproducibility & How-To

This repository contains code to reproduce **Drug–Drug Interaction (DDI)** experiments on DrugBank-based splits (MeTDDI/KnowDDI) and our **Scaffold-OOD** split, plus attribution scripts for motif analyses (ritonavir / cobicistat).

---

## 1) System Requirements

- Linux (Ubuntu 20.04+ tested)
- Python **3.11**
- (Recommended) NVIDIA GPU with **CUDA 12.1** runtime (Driver ≥ 535)

All experiments can run on CPU (slower).

---

## 2) Environment Setup

### Option A — Conda (GPU)

Create `environment.yml` with:
```yaml
name: ddi
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy=1.26.*
  - pandas=2.2.*
  - scikit-learn=1.4.*
  - matplotlib=3.8.*
  - tqdm=4.66.*
  - rdkit=2023.09.5
  - pytorch=2.2.*
  - pytorch-cuda=12.1
  - cudatoolkit=12.1
  - pip:
      - dgl-cu121==2.2.1
      - dgllife==0.3.2
```

Then:
```bash
mamba env create -f environment.yml  # or: conda env create -f environment.yml
conda activate ddi
```

### Option B — Conda (CPU only)

Create `environment.cpu.yml` with:
```yaml
name: ddi-cpu
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy=1.26.*
  - pandas=2.2.*
  - scikit-learn=1.4.*
  - matplotlib=3.8.*
  - tqdm=4.66.*
  - rdkit=2023.09.5
  - pytorch=2.2.*   # CPU build
  - pip:
      - dgl==2.2.1
      - dgllife==0.3.2
```
Then:
```bash
mamba env create -f environment.cpu.yml
conda activate ddi-cpu
```

### Option C — Docker (GPU)

Create `Dockerfile` with:
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p $CONDA_DIR && rm /tmp/mc.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install -y -n base -c conda-forge mamba && conda clean -afy
WORKDIR /workspace
COPY environment.yml /workspace/environment.yml
RUN mamba env create -f /workspace/environment.yml && conda clean -afy
SHELL ["bash","-lc"]
RUN echo "conda activate ddi" >> ~/.bashrc
CMD ["/bin/bash"]
```

Build & run:
```bash
docker build -t ddi-repro .
docker run --gpus all --rm -it -v $PWD:/workspace -w /workspace ddi-repro bash
# inside:
conda activate ddi
```

---

## 3) Data

We **do not redistribute** DrugBank / FDA derivatives. Place CSVs as:

```
interaction/data/
  drugbank/
    unseen_ddi/              # MeTDDI splits (pairs + labels)
      tr_dataset.csv
      val_dataset.csv
      tst_dataset.csv
    unseen_drugs/
      tr_dataset.csv
      val_dataset_unseen_onedrug.csv
      val_dataset_unseen_twodrugs.csv
    data_kgg/                 # KnowDDI split
      train.txt
      val.txt
      test.txt
    scaffold_0p7/
      train.csv
      val.csv
      test.csv
    motifs/
      all_data.csv

regression/data/                   # FDA DDA (from motifsddi release)
    tr_datset_regression_fold1.csv
    val_datset_regression_fold1.csv
    PKDDI_info_2023_FDA_external.csv
```

> Column expectation for DDI CSVs: typically `drug_A`, `drug_B`, `DDI` (or `label`).

---

## 4) Scaffold-OOD Split (ours)

We provide a split script (place at `data/make_scaffold_split.py`) that:
- Groups molecules by **Bemis–Murcko scaffold**;
- Assigns entire scaffold groups to **train/val/test** (no scaffold sharing);
- Enforces **ECFP4 Tanimoto ≤ 0.70** between every test/val molecule and **all train** molecules;
- Keeps pairs only when **both** drugs fall into the same split.

**Create the split** (from the *Unseen DDI pool*):
```bash
python data/make_scaffold_split.py \
  --inputs 'data/drugbank/unseen_ddi/*.csv' \
  --out_dir data/splits/scaffold_ood \
  --val_frac 0.10 --test_frac 0.10 \
  --threshold 0.70 --seed 1337 --enforce_for_val
```

This writes:
```
data/scaffold_0p7/
  train.csv  val.csv  test.csv
  drug_assignments.csv
  split_stats.json
```

The stats we report in the paper correspond to:
```json
{
  "n_pairs_input": 343036,
  "n_unique_canonical_smiles": 1409,
  "val_frac": 0.1,
  "test_frac": 0.1,
  "threshold": 0.7,
  "seed": 1337,
  "enforce_for_val": true,
  "n_pairs_train": 244108,
  "n_pairs_val": 3264,
  "n_pairs_test": 1114,
  "n_mols_train": 1192,
  "n_mols_val": 141,
  "n_mols_test": 76
}
```

---

## 5) Running Experiments


### 5.1 MeTDDI (DrugBank) — Benchmarks

**MFPS**
```bash
python src/ddi_bench_mfps.py
```

**GCN (untrained)**
```bash
python src/ddi_bench_gcn.py
```

**GCN\_T (pretrained)**
```bash
python src/ddi_bench_gcn_t.py
```

**MoLFormer**
```bash
python src/ddi_bench_molf.py
```

> For all **Unseen Pairs** / **Unseen 1 Drug** / **Unseen 2 Drugs** / **Split created in §4** in the same script.

### 5.2 KnowDDI (Knowledge Graph) Split

MFPS:
```bash
python src/ddi_kgg_mfps.py
```

GCN / GCN\_T / MoLF:
```bash
python src/ddi_kgg_gcn.py
python src/ddi_kgg_gcn_t.py
python src/ddi_kgg_gcn_molf.py
```


## 6) Attribution / Motif Analyses

We provide per-embedding scripts. Examples below assume the MeTDDI pairs and will generate substructure diagrams and highlight images for **ritonavir** and **cobicistat**.

### MFPS
```bash
python src/ddi_explain_mfps.py
```

### MoLFormer
```bash
python src/ddi_explain_molf.py
```

### Pretrained GCN
```bash
python src/ddi_explain_gcn_t.py
```

Outputs:
- Motif panels (PNG)
- Highlighted molecules (`ritonavir.png`, `cobicistat.png`)

> RDKit note: fragments that cannot be drawn in isolation are rendered using the closest minimal core; counts are always computed from the exact motif definition.

---

## 7) Tips & Troubleshooting

- **ImportError: `rdMolDraw2D`** — use pinned `rdkit=2023.09.5` from conda-forge (as in envs).
- **DGL CUDA mismatch** — if you are on CUDA 11.8, change `pytorch-cuda=11.8` and `dgl-cu118==2.2.1`.
- **Determinism** — we enable deterministic ops; disable for speed if you only need quick checks.