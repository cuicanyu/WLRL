# WLRL: Wasserstein Latent Representation Learning for Protein-Ligand Binding Affinity Prediction

This repository contains the implementation of "WLRL: Wasserstein latent representation learning for protein-ligand binding affinity prediction". WLRL models each protein-ligand complex as a latent distribution and aligns these distributions to a shared set of learnable reference distributions through the Wasserstein Barycentric Representation Operator (WBRO) with reference-usage regularization, aiming to reduce heterogeneity in protein-ligand complexes.



## System requirements

The reference environment uses:

- Linux (Ubuntu 22.04 or 24.04 recommended);
- Python 3.10.8;
- PyTorch 2.0.1;
- CUDA 11.7;
- PyTorch Geometric 2.5.2;
- NumPy 1.26.4;
- scikit-learn 1.3.2;
- Matplotlib 3.8.0;
- Weights & Biases 0.15-0.17.

An NVIDIA GPU with at least 24 GB of memory is recommended for training.



## Installation

Install Miniconda or Anaconda, then create the supplied environment:

```bash
conda env create -f environment.yml
conda activate wlrl
```


## Data


The precomputed graph datasets follow the GEMS graph construction procedure. GEMS datasets and preprocessing information are available from the [GEMS data archive](https://github.com/camlab-ethz/GEMS).



## Training

Place the precomputed PDBbind CleanSplit training dataset at, for example:

```text
preprocessed_data/B6AEPL_train_cleansplit.pt
```

Train one fold with:

```bash
mkdir -p runs/seed0

python train.py \
  --dataset_path preprocessed_data/B6AEPL_train_cleansplit.pt \
  --run_name wlrl_seed0 \
  --save_dir runs/seed0 \
  --fold_to_train 0 \
  --n_folds 5 \
  --train_seed 0 \
  --random_seed 0 \
  --num_epochs 600 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --weight_decay 0.001 \
  --optim SGD \
  --early_stopping true
```



## Evaluation

```bash
mkdir -p results/casf2016

python test.py \
  --dataset_path preprocessed_data/B6AEPL_casf2016.pt \
  --stdicts f0_best_stdict.pt,f1_best_stdict.pt,f2_best_stdict.pt,f3_best_stdict.pt,f4_best_stdict.pt \
  --model_arch WLRL18 \
  --save_path results/casf2016
```
