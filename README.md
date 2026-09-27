# GenAI_HTTF

This repository contains the code and results associated with the work:

**Spatially Aware Generative Models for Full-Field Sensor Reconstruction in Safety-critical Digital Twins**

The study investigates **generative AI (GenAI) for sparse-sensor reconstruction** in the **High Temperature Test Facility (HTTF)** at Oregon State University, with the goal of supporting digital twins when only part of the sensor network is available.

## Overview

### High Temperature Test Facility (HTTF)

The HTTF is a reduced-scale integral thermal-hydraulic facility designed to reproduce important behavior of modular high-temperature gas-cooled reactors.

The facility contains spatially distributed thermocouples measuring:

- **TS:** solid temperatures
- **TF:** fluid temperatures

The HTTF includes a graphite-based core, upper and lower plena, reflectors, coolant channels, heater channels, and a reactor cavity cooling system. Electric heaters are used to reproduce reactor-like thermal conditions and decay-heat behavior.

The present work primarily uses two HTTF experiments:

- **PG-29:** Depressurized Conduction Cooldown (DCC)
- **PG-27:** Pressurized Conduction Cooldown (PCC)

---

Digital twins rely on sensor measurements to maintain an estimate of the physical system state.

In practice, however, complete sensor availability cannot always be guaranteed. Sensors may:

- fail,
- become intermittently unavailable,
- experience degradation,
- be excluded during data-quality screening, or
- be deliberately reduced to decrease sensing and computational requirements.

This creates a **partial-observability problem**.

The goal of this repository is therefore to reconstruct the unavailable sensor measurements using the remaining reliable measurements together with spatial sensor information and transient history.

The general problem is to reconstruct missing sensor measurements from partial sensor observations by combining available sensor values, spatial sensor locations, and transient history to recover a more complete digital-twin state.

## Reconstruction Cases

Two complementary reconstruction problems are considered.

### Case 1: Intra-Experiment Reconstruction

The intra-experiment case reconstructs missing sensors within the same experiment.

### Case 2: Inter-Experiment Reconstruction

The inter-experiment case evaluates whether reconstruction relationships learned from one HTTF transient can transfer to another physically different transient.

## Models

Six reconstruction approaches are evaluated.

### Generative Models

1. **Diffusion Model (DM)**
2. **Variational Autoencoder (VAE)**
3. **Generative Adversarial Network (GAN)**

### Deterministic Models

4. **GNN-GRU**
5. **GRU**
6. **Ridge Regression**

For intra-experiment reconstruction, the generative models use spatial and temporal information through a common GNN-GRU conditioning framework.

## Environment Installation

To set up the environment for this project, follow these steps:

```bash
# 1. Create a new conda environment with Python 3.11
conda create -n torchgpu python=3.11

# 2. Activate the environment
conda activate torchgpu

# 3. Install PyTorch and related libraries
pip install torch torchvision torchaudio

# 4. Install other relevant packages
pip install pandas matplotlib scikit-learn seaborn
```

## Example: Intra-Experiment Diffusion Model

One representative workflow is the GNN-GRU diffusion model for PG-29 intra-experiment reconstruction.

Run the corresponding DM script:

```bash
python models/DM/gnn_gru_dm.py
```

Typical results may be stored under:

```text
results/DM/
```

with files such as:

```text
global_metrics.csv
per_sensor_metrics.csv
predictions.csv
reconstruction_timeseries.png
sensor_rmse.png
```

## Citation

```bibtex
@article{nabila2026spatial,
  title   = {Spatially Aware Generative Models for Full-Field Sensor Reconstruction in Safety-critical Digital Twins},
  author  = {Nabila, Umme Mahbuba and Lin, Linyu and Radaideh, Majdi I.},
  year    = {2026}
}
```
