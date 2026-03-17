import os
os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"

import torch
import wandb
import yaml

from torch.utils.data import DataLoader
from utils_dir.train_test import train_test
from utils_dir import dataset, models

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def get_abs_path(rel_path):
    # Helper function to convert relative paths from THIS file to absolute paths
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))

# ====================================================================
# CONFIG
# ====================================================================
with open(get_abs_path('training_config.yaml')) as f:
    config = yaml.safe_load(f)

MANIFEST_PATH   = config['data']['manifest_path']
CACHE_DIR       = config['data']['cache_dir']
BATCH_SIZE      = config['training']['batch_size']
NUM_EPOCHS      = config['training']['num_epochs']
LEARNING_RATE   = config['training']['learning_rate']
WEIGHT_DECAY    = config['training']['weight_decay']
CHECKPOINT_FREQ = config['training']['checkpoint_freq']
WANDB_ID        = config['resume']['wandb_id']
CHECKPOINT_PATH = config['resume']['checkpoint_path']
PROJECT_NAME    = config['wandb']['project']
WANDB_ENABLED   = config['wandb']['enabled']
NOTES = config['wandb']['notes']
TAGS = config['wandb']['tags']

# ====================================================================
# DATA
# ====================================================================
print("Staging data to scratch...")
manifest   = dataset.stage_manifest_to_scratch(MANIFEST_PATH, cache_dir=CACHE_DIR)
train_data = manifest['splits']['train']['files']
test_data  = manifest['splits']['test']['files']

print("Initializing datasets...")
train_album = dataset.ShardStreamIterableDataset(
    shard_file_list = train_data,
    manifest_path   = MANIFEST_PATH,
    batch_size      = BATCH_SIZE,
    is_train        = True
)

test_album = dataset.ShardStreamIterableDataset(
    shard_file_list = test_data,
    manifest_path   = MANIFEST_PATH,
    batch_size      = BATCH_SIZE,
    is_train        = False,
    label_mean      = train_album.label_mean,
    label_std       = train_album.label_std
)

# ========================================================================================
# Calculate Optimal Prefetch Factor (Necessary because of custom iterable dataset class)
# ========================================================================================
TOT_IMGS = manifest['metadata']['total_images_all_splits']
TOT_SHARDS = manifest['metadata']['total_shards_all_splits']
avg_imgs_per_shard = TOT_IMGS / TOT_SHARDS

TRAIN_PREFETCH_FACTOR = int(round(avg_imgs_per_shard / BATCH_SIZE) + 1)
TEST_PREFETCH_FACTOR = TRAIN_PREFETCH_FACTOR * 2  # Test set is smaller, so can afford more prefetching to speed it up
print(f"Calculated train prefetch factor: {TRAIN_PREFETCH_FACTOR} based on average images per shard and batch size.")
print(f"Calculated test prefetch factor: {TEST_PREFETCH_FACTOR} always double the train prefetch factor")
# ========================================================================================
train_data_loader = DataLoader(
    dataset            = train_album,
    batch_size         = None,
    shuffle            = False,
    num_workers        = 2,
    prefetch_factor    = TRAIN_PREFETCH_FACTOR,
    pin_memory         = True,
    persistent_workers = True
)

test_data_loader = DataLoader(
    dataset            = test_album,
    batch_size         = None,
    shuffle            = False,
    num_workers        = 4,
    prefetch_factor    = TEST_PREFETCH_FACTOR,
    pin_memory         = True,
    persistent_workers = True
)

# ====================================================================
# MODEL PARAMS — change these when switching architectures
# ====================================================================
model_params = {
    'hidden_units' : 32,
    'leak_factor'  : 0.1,
    'dropout_rate' : 0.0,
    'temporal_res' : 512
}

# ====================================================================
# MODEL
# ====================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    print("WARNING: GPU not available, training on CPU!")

model = models.RNO_four_late_non_linear_merge(
    input_shape  = 1,
    output_shape = 3,
    label_mean   = train_album.label_mean,
    label_std    = train_album.label_std,
    **model_params
)

# ====================================================================
# OPTIMIZER & LOSS
# ====================================================================
optimizer = torch.optim.AdamW(
    params       = model.parameters(),
    lr           = LEARNING_RATE,
    weight_decay = WEIGHT_DECAY,
    fused        = True
)
loss_fn = torch.nn.MSELoss()

# ====================================================================
# EXPERIMENT NAME
# ====================================================================
param_str       = '_'.join(f'{k}-{v}' for k, v in model_params.items())
experiment_name = (
    f'exp_{model.__class__.__name__}'
    f'_{param_str}'
    f'_lr-{LEARNING_RATE:.2e}'
    f'_wdcay-{WEIGHT_DECAY:.1e}'
)

# ====================================================================
# COMPILE MODEL
# ====================================================================
# Move model to device then compile — compiling after .to() avoids
# recompilation when tensors arrive on the expected device
model.to(DEVICE)
model = torch.compile(model)  # type: ignore

# ====================================================================
# WANDB — must init before building checkpoint_dir (needs run.id)
# ====================================================================
if WANDB_ENABLED:
    wandb.init(
        project = PROJECT_NAME,
        name    = experiment_name,
        id      = WANDB_ID,
        resume  = 'allow',
        notes = NOTES,
        tags = TAGS,
        config  = {
            'batch_size'      : BATCH_SIZE,
            'num_epochs'      : NUM_EPOCHS,
            'learning_rate'   : LEARNING_RATE,
            'weight_decay'    : WEIGHT_DECAY,
            'manifest_path'   : MANIFEST_PATH,
            'n_train_batches' : len(train_data_loader),
            'n_test_batches'  : len(test_data_loader),
            'optimizer'       : optimizer.__class__.__name__,
            'loss_fn'         : loss_fn.__class__.__name__,
            **model_params
        }
    )

    wandb.watch(model, log='all', log_freq=len(train_data_loader)) #type: ignore Log every epoch
    wandb.define_metric('test_loss', summary='min')
    wandb.define_metric('train_loss', summary='min')

    manifest_artifact = wandb.Artifact('dataset_manifest', type='dataset')
    manifest_artifact.add_file(MANIFEST_PATH)
    wandb.log_artifact(manifest_artifact)

# ====================================================================
# CHECKPOINT DIR — built after wandb.init so run.id is available
# ====================================================================
run_id = wandb.run.id if WANDB_ENABLED else experiment_name #type: ignore (If wandb disabled, just use experiment name for checkpoint dir)
checkpoint_dir = os.path.join(
    get_abs_path('model_experiments'),
    PROJECT_NAME,
    'experiments',
    run_id,
    'checkpoints'
)
os.makedirs(checkpoint_dir, exist_ok=True)

# ====================================================================
# STARTUP SUMMARY
# ====================================================================
print(f"Experiment : {experiment_name}")
print(f"Device     : {DEVICE}")
print(f"Batches    : {len(train_data_loader)} train | {len(test_data_loader)} test")
print(f"Checkpoints: {checkpoint_dir}")
if WANDB_ENABLED and wandb.run:
    print(f"WandB      : {wandb.run.url}")

# ====================================================================
# TRAIN
# ====================================================================
train_test(
    model            = model, # type:ignore
    train_dataloader = train_data_loader,
    test_dataloader  = test_data_loader,
    optimizer        = optimizer,
    loss_fn          = loss_fn,
    device           = DEVICE,
    epochs           = NUM_EPOCHS,
    checkpoint_dir   = checkpoint_dir,
    checkpoint_freq  = CHECKPOINT_FREQ,
    checkpoint_path  = CHECKPOINT_PATH
)

if WANDB_ENABLED:
    wandb.finish()