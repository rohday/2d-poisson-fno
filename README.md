# SciML-FNO

Fourier Neural Operator for solving the 2D heterogeneous Poisson equation.

## Quick Start

```bash
# Generate data
cd data_generation
python3 generate_dataset.py --train_samples 1000 --val_samples 200 --test_samples 200 --output ../data/

# Train
python3 model/train.py --data_path data/train.npz --epochs 100 --output_dir checkpoints/

# Evaluate
python3 model/evaluate.py --checkpoint checkpoints/ --data_path data/test.npz
```

## Structure

```
├── model/           # FNO implementation, training, evaluation
├── data_generation/ # Poisson solver and dataset generation
├── benchmarks/      # Throughput stress tests
└── allmodels/       # Pre-trained models (z1k, z5k, z8k)
```

## Requirements

```
torch numpy scipy matplotlib tqdm h5py pyyaml psutil
```
