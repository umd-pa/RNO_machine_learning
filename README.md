# RNO Machine Learning - Research Project
## Research Project under Prof. Brian Clark
University of Maryland  
Department of Physics  

A collection of machine learning pipelines for radio neutrino observatories, supporting trigger classification, station-level event classification, and multi-station vertex reconstruction.

---

## Table of Contents

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Classifier Workflows](#5-classifier-workflows)
6. [Vertex Reconstruction Workflow](#6-vertex-reconstruction-workflow)
7. [Data Generation Flow](#7-data-generation-flow)
8. [Training Flow](#8-training-flow)
9. [Evaluation and Validation](#9-evaluation-and-validation)
10. [Notebooks and Debugging](#10-notebooks-and-debugging)
11. [License](#11-license)

---

## 1. Project Overview

This repository documents the research and development of machine learning pipelines—specifically Convolutional Neural Networks (CNNs)—to reconstruct and classify neutrino events using simulated data from the Radio Neutrino Observatory in Greenland (RNO-G). 

The repository focuses on two main machine learning pipelines:

| Pipeline | Description |
|---|---|
| `RNO_classifier/` | Classification workflows for trigger and station-level event classification |
| `RNO_vertex_reconstruction/` | Multi-station vertex reconstruction using a CNN-based pipeline |

The repo supports data generation, training, evaluation, and debugging for detector simulation and ML workflows. It is designed for use with Python and optional HTCondor job submission.

---

## 2. Repository Structure

```text
RNO_machine_learning/
├── LICENSE
├── README.md
├── complete_requirements.txt
├── requirements.txt
├── RNO_classifier/
│   ├── benchmark_ARA_trigger_classification/
│   │   ├── generate/
│   │   │   ├── config_noise.yaml
│   │   │   ├── config_signal.yaml
│   │   │   ├── go_generate.sh
│   │   │   ├── station.json
│   │   │   ├── step1_make_lists.py
│   │   │   ├── step2_run_sims.py
│   │   │   ├── step3_plot_traces.py
│   │   │   └── submit/
│   │   └── train/
│   │       ├── go_extract.sh
│   │       ├── go_train.sh
│   │       ├── step1_extract.py
│   │       ├── step1b_plot_dataset.py
│   │       ├── step2_train.py
│   │       └── step3_evaluate.py
│   ├── benchmark_station_classification/ # Generate and train pipelines are the same as other classification experiments
│   │   ├── generate/
│   │   └── train/
│   ├── RNO_station_classification/ # Generate and train pipelines are the same as other classification experiments
│   │   ├── generate/
│   └── └── train/
└── RNO_vertex_reconstruction/
    ├── generate/
    │   ├── jobs/
    │   │   ├── create_dagman.py
    │   │   ├── create_dagman_config.yaml
    │   │   ├── job_logs/
    │   │   └── submissions/
    │   └── simulation/
    └── train/
        ├── train_model.py
        ├── training_config.yaml
        ├── manifest_builder/
        ├── model_experiments/
        ├── model_validation.ipynb
        └── utils_dir/
```

---

## 3. Prerequisites

- Python 3.x
- Packages listed in `requirements.txt`

**Optional:**

- **CUDA** — for GPU-accelerated training
- **HTCondor** — for distributed simulation and generation jobs
- **Weights & Biases (WandB)** — for experiment tracking

---

## 4. Installation

Clone the repository:

```bash
git clone <repo-url>
cd RNO_machine_learning
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

Or install the full dependency set:

```bash
pip install -r complete_requirements.txt
```

---

## 5. Classifier Workflows

The classifier workflows support data generation, feature extraction, and model training for event classification. The standard structure for these pipelines is divided into two primary subdirectories: `generate/` and `train/`.

#### `generate/`

This directory handles the configuration, simulation, and data generation processes.

| File / Directory | Description |
|---|---|
| `config_noise.yaml` | Noise simulation configuration |
| `config_signal.yaml` | Signal simulation configuration |
| `station.json` / `RNO_single_station.json` | Station hardware configuration |
| `go_generate.sh` | Shell helper script to execute the generation pipeline |
| `step1_make_lists.py` | Creates event lists for simulation |
| `step2_run_sims.py` | Runs the simulation jobs |
| `step3_plot_traces.py` | Plots sample waveforms to verify outputs |
| `data/` | Directory where generated simulation datasets are stored |
| `submit/` | Scripts for job submission to computing clusters |

#### `train/`

This directory handles data extraction, dataset inspection, and the training and evaluation of the classification models.

| File | Description |
|---|---|
| `go_extract.sh` | Shell helper to extract features and build datasets |
| `go_train.sh` | Shell helper to launch model training |
| `requirements.txt` | Training-specific Python dependencies |
| `step1_extract.py` | Extracts training data from raw simulations |
| `step1b_plot_dataset.py` | Inspects and visualizes the structured training dataset |
| `step2_train.py` | Main script to define and train the model |
| `step2_train_old.py` | Legacy model training flow |
| `step3_evaluate.py` | Evaluates trained models and computes performance metrics |

These standard `generate/` and `train/` structures are located inside of each of their respective pipeline directories, which are designed for different types of stations and hardware responses:

- **`benchmark_ARA_trigger_classification/`**: Benchmarks trigger classification using ARA-style (Askaryan Radio Array) simulated signal and noise data.
- **`benchmark_station_classification/`**: Benchmarks broader station-level classification performance using generic station simulation data.
- **`RNO_station_classification/`**: Performs single-station classification explicitly configured for the RNO-G (Radio Neutrino Observatory in Greenland) hardware response.

---

## 6. Vertex Reconstruction Workflow

End-to-end multi-station vertex reconstruction using a CNN pipeline.

#### `generate/jobs/`

| File | Description |
|---|---|
| `create_dagman.py` | Generate HTCondor DAG configs |
| `create_dagman_config.yaml` | DAG creation configuration |
| `job_logs/` | Logs from submitted jobs |
| `submissions/` | Submitted job files |

#### `generate/simulation/`

Simulation and data generation scripts.

#### `train/`

| File | Description |
|---|---|
| `train_model.py` | Main training script |
| `training_config.yaml` | Training configuration |
| `manifest_builder/` | Create train/test/val manifests |
| `utils_dir/` | Training utilities |
| `model_experiments/` | Experiment checkpoints and outputs |
| `model_validation.ipynb` | Notebook for validation and analysis |

---

## 7. Data Generation Flow

### Classifier Pipelines

```
step1_make_lists.py  →  step2_run_sims.py  →  step3_plot_traces.py
```

1. Use `step1_make_lists.py` to build simulation/run lists.
2. Run `step2_run_sims.py` to generate simulated waveforms.
3. Use `step3_plot_traces.py` to inspect example traces.

> For ARA/benchmark tasks, `go_generate.sh` wraps this workflow into a single command.

### Vertex Reconstruction

1. Generate HTCondor job descriptions with `create_dagman.py`.
2. Run the generated DAG jobs to produce simulation shards.
3. Build train/val/test manifests using `manifest_builder/`.

---

## 8. Training Flow

### Classifier Pipelines

```
step1_extract.py  →  (step1b_plot_dataset.py)  →  step2_train.py  →  step3_evaluate.py
```

1. Run `go_extract.sh` or `step1_extract.py` to convert raw simulation outputs into dataset inputs.
2. *(Optional)* Use `step1b_plot_dataset.py` to visualize dataset structure and quality.
3. Train models with `step2_train.py`.
4. Evaluate with `step3_evaluate.py`.

### Vertex Reconstruction

1. Configure training parameters in `training_config.yaml`.
2. Train with `train_model.py`, using manifests from `manifest_builder/` to control splits.
3. Compare experiments in `model_experiments/`.

---

## 9. Evaluation and Validation

- **Classifiers:** Use the `eval/` folder under `benchmark_station_classification/` for evaluation utilities, or run `step3_evaluate.py` in any classifier training folder to compute model metrics.
- **Vertex Reconstruction:** Use `model_validation.ipynb` and the utilities in `train/` to inspect model performance.

---

## 10. Notebooks and Debugging

| Notebook / Folder | Description |
|---|---|
| `RNO_classifier/debug.ipynb` | Exploration and debugging for classifier pipelines |
| `RNO_classifier/debugging_folder/` | Scratch files and debug datasets |
| `RNO_vertex_reconstruction/train/model_validation.ipynb` | Vertex reconstruction validation notebook |

---

## 11. License

This repository is released under the terms of the [LICENSE](LICENSE) file.
