# RNO-G Multi-station Vertex Reconstruction with CNN Pipeline
## Repo Structure
```text
RNO_vertex_reconstruction_ml/
├── RNO_album_maker/                                                   # Directory for data creation
|    ├── jobs/                                                         # Directory for managing jobs in HTCondor
|    |    ├── create_dagman.py                                         # Script for creating master.dag
|    |    ├── create_dagman_config.yaml                                # Config file for create_dagman.py
|    |    └── master.dag (EXAMPLE)                                     # Example master.dag
|    └── simulation/                                                   # Directory containing simulation and datacreation scripts
|        ├── simulation_data/                                          # Data stored while simulating
|        |    ├── RNO_four_stations.json                               # Station that is simulated
|        |    └── config.yaml                                          # Config file for 2_run_simulation.py
|        ├── simulation_steps/                                         # Directory that contains simulation steps
|        |    ├── 1_generate_neutrinos.py                              # Generates neutrinos
|        |    ├── 2_run_simulation.py                                  # Runs simulation
|        |    ├── 3_generate_hdf5_shards.py                            # Generates shard albums
|        |    ├── F_cleanup.py                                         # Cleans up simulation intermediate files
|        |    └── album_simulation_utils.py                            # Utilities for simulations
|        |    
|        └── simulation_utils/                                         # Directory for other simulation utilities
├── RNO_machine_learning/                                              # Saved models from training
|    ├── train_model.py                                                # Script for training models 
|    ├── training_config.yaml                                          # Config for train_model.py 
|    ├── manifest_builder                                              # Directory for generating .json manifests
|    |    ├── build_shards_manifest.py                                 # Script to create manifest from shard dirs
|    |    ├── manifest_config.yaml                                     # Config for build_shards_maifest.py
|    |    └── manifest/ (EXAMPLE)                                      # Example of what manifest should look like
|    ├── utils_dir/                                                    # Utils dir for machine learning
|    └── model_experiments/ (EXAMPLE)                                  # Directory containing checkpoints and experiments.
├── complete_requirements.txt                                          # Complete requirements with all module dependencies (NOT CHECKED)
└── requirements.txt                                                   # Requirements with important module dependencies (NOT CHECKED)
```

## Getting Started

### Prerequisites
* Python
* HTCondor (if you want to run multiple simulations in parallel)
* WanDB account (Not necessary but recommended)
* Cuda

### Installation
1. Clone repository
2. cd into the repository
3. `pip install -r requirements.txt` or `pip install -r complete_requirements.txt`. The second one contains all specific modules + dependencies which may break down for some users

## Creating Data
To create data you must run the 3 simulation steps in sequence. Please take a look at the argparse inside of the scripts to understand what each script expects.

TODO

## Splitting Data
Once *multiple shards* have been created, run manifest_builder on the shards to create a .json separating the shards into train/test/val splits. 
> There must be multiple shards for this to work well!


## Training
Once manifest is created, train any model with train_model.py. Pass the path of the manifest you just created to training_config.yaml.

