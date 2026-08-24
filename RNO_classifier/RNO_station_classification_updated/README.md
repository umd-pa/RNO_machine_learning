# Classifier Consistency Workflow

This directory is designed to train a classifier on specific ice/attenuation/gain configurations and evaluate its robustness (consistency) against minor perturbations in benchmark data.
## Prerequisites

RNOG detector set-up uses an antenna not defaultly included with NuRadioMC. Run the following in your virtual environment terminal to get the necessary antenna model.

'
python - <<'PY'
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
ap = AntennaPatternProvider()
ap.load_antenna_pattern("RNOG_vpol_v3_5inch_center_n1.74")
print("cached")
PY
'

Afterwards, delete the .lock file (unnecessary if you already have the antenna model downloaded)

'rm -f /path/to/lock'


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

We are going to create a dagman file to manage the generation of simulated data.

`bash step1_benchmark_model/step1.1_generate_benchmark.sh`

This script generates the Dagman file required for large-scale data production.
- **How to Submit**: Once generated, submit using: `condor_submit_dag -F [path_to_dag]`
- **Output**: Upon completion, `nu_extracted.hdf5` and `noise_extracted.hdf5` will be available in `data_dir/classifier_consistency/ARZ2020/benchmark`.

### Step 1.2: Benchmark Training
`bash step1_benchmark_model/step1.2_train_benchmark.sh`

Train the classifier on the prepared data. Metrics are saved in `data_dir/classifier_consistency/ARZ2020/benchmark/runs`.

You can customize the training using the following parameters:
```bash
python step2_train.py \
    --signal signal.h5 \
    --noise  noise.h5 \
    --out    runs/exp01 \
    --arch tiny \
    --crop_samples 512 \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-3 \
    --dropout 0.2 \
    --weight_decay 1e-4 \
    --val_frac 0.15 \
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

### Step 2: Compare benchmark on different models

### Step 2.1: Comparison Data Generation

We are going to create a dagman file to manage the generation of simulated data.

```bash
bash step2_compare/step2.1_generate_comparison.sh \
    --ice_model   ice_model \
    --att_model   att_model \
    --hw_resp     None \
    --signal_model ARZ2020 \
    --data_dir    /path/to/data_dir \
    --python_path /path/to/venv/bin/python
```

This script generates the Dagman file required for large-scale data production.

**Parameter Descriptions:**
- `--ice_model`: Ice model to use. One of `greenland_simple`, `greenland_simple_plus_nsigma`, `greenland_simple_minus_nsigma`, `greenland_simple_plus_z0sigma`, `greenland_simple_minus_z0sigma` (default: `greenland_simple`).
- `--att_model`: Attenuation model. One of `GL3`, `GL3_plus_sigma`, `GL3_minus_sigma` (default: `GL3`).
- `--hw_resp`: Hardware gain response. One of `gain_plus_sigma`, `gain_minus_sigma`, or default if none provided (default: `None`).
- `--signal_model`: Askaryan pulse model to use (default: `ARZ2020`).
- `--data_dir`: Top-level data directory. Defaults to the `data_dir` set in `user_config.yaml` from step 0 if not provided.
- `--python_path`: Path to your Python interpreter. Defaults to the `python_path` set in `user_config.yaml` from step 0 if not provided.

**Note:** leaving all four sim-parameter flags (`--ice_model`, `--signal_model`, `--att_model`, `--hw_resp`) at their defaults generates the benchmark dataset. Changing exactly one of them creates a single-systematic comparison dataset (e.g. `.../att_model/GL3_plus_sigma/`). Changing more than one at once creates a combined comparison dataset under `.../other/<ice_model>-<att_model>-<hw_resp>/`.

- **How to Submit**: Once generated, submit using: `condor_submit_dag -F [path_to_dag]`
- **Output**: Upon completion, `nu_extracted.hdf5` and `noise_extracted.hdf5` will be available in the corresponding dataset directory under `data_dir/classifier_consistency/<signal_model>/...`.

### Step 2.2: Model comparison

Once the comparison dataset has finished generating, evaluate the benchmark
classifier's performance on it (e.g. different ice model, attenuation, gain).

```bash
bash step2_compare/step2.2_eval_comparison.sh \
    --checkpoint runs/exp01/best_model.pt \
    --signal     signal.h5 \
    --noise      noise.h5 \
    --target_fpr 1e-5 \
    --out        runs/exp01/eval \
    --batch_size 512 \
    --compare
```

**Parameter Descriptions:**
- `--checkpoint`: Path to the trained model checkpoint (from step 1.2).
- `--signal`: Path to the signal HDF5 file to evaluate on.
- `--noise`: Path to the noise HDF5 file to evaluate on.
- `--target_fpr`: False positive rate to calibrate the operating threshold at (default: `1e-3`).
- `--out`: Directory where plots and metrics are saved (default: `<checkpoint_dir>/eval`).
- `--batch_size`: Number of waveforms processed at once (default: `512`).
- `--compare`: Required when evaluating on a dataset different from the one the model was trained on — uses all events in `--signal`/`--noise` rather than the val-split indices stored in the checkpoint.

**Output:** `roc_curve.png`, `confusion_matrix.png`, `score_histogram.png`, `efficiency_vs_snr.png`, `eff_vs_snr.csv`, `scores.npz`, and `summary.txt` are written to `--out`.
