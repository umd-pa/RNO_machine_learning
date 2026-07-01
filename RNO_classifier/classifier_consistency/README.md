# Classifier Consistency Workflow

This directory is designed to train a classifier on specific ice/attenuation/gain configurations and evaluate its robustness (consistency) against minor perturbations in benchmark data.

## Pipeline Overview
1. **Neutrino Generation**: Create a base set of neutrinos.
2. **Dataset Preparation**: Generate noise and neutrino datasets using standard benchmark settings.
3. **Benchmark Training**: Train the primary benchmark classifier.
4. **Organization**: Establish a structured directory within `data_dir/classifier_consistency` to organize various comparison datasets for analysis.

## Usage Instructions

Navigate to the directory:
`cd RNO_classifier/classifier_consistency`

### Step 0: Neutrino Generation
`python step0_neutrino_dist.py [data_dir]`

Generate the neutrinos used for all subsequent simulations. These remain static across all comparisons.
- `data_dir`: The root directory where all datasets will be stored.
- This script creates a `classifier_consistency/` subdirectory and generates a `user_config.yaml` file containing your `data_dir` and `python_path`.

### Step 1.1: Benchmark Data Generation
`python step1_benchmark_model/step1.1_generate_benchmark.sh`

This script generates the Dagman file required for large-scale data production.
- **Submission**: Once generated, submit using: `condor_submit_dag [path_to_dag]`
- **Output**: Upon completion, `nu_extracted.hdf5` and `noise_extracted.hdf5` will be available in `data_dir/classifier_consistency/ARZ2020/benchmark`.

### Step 1.2: Benchmark Training
`python step1_benchmark_model/step1.2_train_benchmark.sh`

Train the classifier on the prepared data. Metrics are saved in `data_dir/classifier_consistency/ARZ2020/benchmark/runs`.

You can customize the training using the following parameters:
```bash
python step2_train.py \
    --signal signal.h5 \
    --noise  noise.h5 \
    --out    runs/exp01 \
    --arch tiny \
    --crop-samples 512 \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3 \
    --dropout 0.2 \
    --weight-decay 1e-4 \
    --val-frac 0.15 \
    --seed 42
```

**Parameter Descriptions:**
- `--signal`: Path to the signal HDF5 file.
- `--noise`: Path to the noise HDF5 file.
- `--out`: Directory where results (models, metrics, plots) are saved.

NOTE: Leave `signal`, `noise` and `out` empty for benchmark training!!

- `--arch`: Model architecture choice (`baseline`, `tiny`, `fpga`, `resnet`, `resnet+`, or `resnet++`).
- `--crop_samples`: Number of samples to keep after center-cropping the waveforms.
- `--epochs`: Total number of training epochs.
- `--batch_size`: Number of samples per batch.
- `--lr`: Initial learning rate for the optimizer.
- `--dropout`: Dropout probability in the model's head.
- `--weight_decay`: Weight decay coefficient for regularization.
- `--val_frac`: Fraction of data used for validation.
- `--seed`: Random seed for reproducibility.

### Step 2:
*Work in Progress*