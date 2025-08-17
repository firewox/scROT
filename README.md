# scROT: Single-Cell RNA-seq Optimal Transport

## Introduction
scROT is a computational tool designed to integrate single-cell RNA sequencing (scRNA-seq) data with bulk RNA sequencing data using optimal transport (OT) theory. This method aims to address challenges in data integration, such as batch effects and differences in data modalities, by finding an optimal mapping between the two datasets.

## Purpose
The primary purpose of scROT is to facilitate the joint analysis of scRNA-seq and bulk RNA-seq data, enabling more comprehensive biological insights. It helps in:
- **Batch Effect Correction**: Aligning data from different experiments or platforms.
- **Cell Type Deconvolution**: Inferring cell type proportions in bulk samples using single-cell references.
- **Cross-Modality Integration**: Bridging the gap between single-cell resolution and bulk population-level measurements.

## Key Features
- **Optimal Transport-based Integration**: Utilizes optimal transport to find a robust and biologically meaningful mapping between scRNA-seq and bulk RNA-seq data.
- **Variational Autoencoder (VAE) Integration**: Incorporates VAEs for dimensionality reduction and learning latent representations, enhancing the integration process.
- **Flexible Data Handling**: Supports various data loading and sampling strategies, including batch-aware sampling and oversampling techniques.
- **Configurable Model Parameters**: Allows users to adjust key parameters such as regularization strength, learning rates, and encoder dimensions to optimize integration for specific datasets.
- **Performance Evaluation**: Includes mechanisms for evaluating integration quality using metrics like AUC and APR.

## Components
The scROT project is structured into several key components:
- `MainRun.py`: The main entry point for running the scROT model, handling argument parsing, data loading, model initialization, training, and result saving.
- `scot/data_loader.py`: Contains classes and functions for loading and preprocessing scRNA-seq and bulk RNA-seq data, including `BatchSampler`, `BatchSampler_balance`, `SingleCellDataset`, and `load_data` functions.
- `scot/function.py`: Implements the core `Run` function, which orchestrates the training process, manages model parameters, handles GPU device selection, and logs progress.
- `scot/model/vae.py`: Defines the Variational Autoencoder (VAE) architecture, including encoder and decoder components, and methods for inference and performance evaluation.
- `scot/model/loss.py`: Contains custom loss functions used in the VAE and optimal transport framework.
- `scot/model/layer.py`: Defines custom neural network layers used within the VAE model.
- `scot/model/mmd.py`: Implements Maximum Mean Discrepancy (MMD) for measuring distribution similarity.
- `scot/model/utils.py`: Provides utility functions for model training and evaluation.
- `scot/logger.py`: Handles logging of training progress and results.

## Installation

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Clone the Repository
```bash
git clone https://github.com/your_username/scROT.git
cd scROT
```

### Install Dependencies
It is recommended to create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

Then install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Model
To run the scROT model, execute `MainRun.py` with the necessary arguments. Here's an example:

```bash
python MainRun.py \
    --bulk_path data/bulk_data.h5ad \
    --sc_path data/sc_data.h5ad \
    --output_dir results/ \
    --lambda_value 0.1 \
    --encoder_dim 128 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --sampler smote \
    --gpu 0
```

### Command-line Arguments
- `--bulk_path`: Path to the bulk RNA-seq data file (e.g., H5AD format).
- `--sc_path`: Path to the single-cell RNA-seq data file (e.g., H5AD format).
- `--output_dir`: Directory to save results (OT plan, logs, etc.).
- `--lambda_value`: Regularization parameter for optimal transport.
- `--encoder_dim`: Dimension of the VAE encoder's latent space.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.
- `--lr`: Learning rate for the optimizer.
- `--sampler`: Sampling strategy (e.g., `smote`, `weight`, `balance`).
- `--gpu`: GPU device ID to use (e.g., `0`, `1`). Set to `-1` for CPU.
- `--seed`: Random seed for reproducibility.
- `--ot_method`: Optimal transport method (e.g., `sinkhorn`).
- `--reg`: Regularization strength for VAE.
- `--ot_iter`: Number of optimal transport iterations.
- `--ot_reg`: Optimal transport regularization.
- `--ot_reg_m`: Optimal transport regularization for marginals.
- `--ot_eps`: Epsilon for Sinkhorn algorithm.
- `--ot_tau`: Tau for Sinkhorn algorithm.
- `--ot_rho`: Rho for Sinkhorn algorithm.
- `--ot_eta`: Eta for Sinkhorn algorithm.
- `--ot_solver`: Optimal transport solver.
- `--ot_norm`: Normalization for optimal transport.
- `--ot_scale`: Scaling for optimal transport.
- `--ot_cost`: Cost function for optimal transport.
- `--ot_metric`: Metric for optimal transport.
- `--ot_init`: Initialization for optimal transport.
- `--ot_warmup`: Warmup epochs for optimal transport.
- `--ot_plan_save`: Whether to save the optimal transport plan.

### Output
The model will output various files to the specified `output_dir`, including:
- Training logs.
- Model checkpoints.
- Optimal transport plan (if `--ot_plan_save` is enabled).
- Performance metrics (AUC, APR) for bulk and single-cell data.

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation
If you use scROT in your research, please cite our work (citation details will be provided upon publication).

   


