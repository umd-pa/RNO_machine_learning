import os
os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"

import torch
import wandb
import yaml

from torch.utils.data import DataLoader
from utils_dir.train_test import train_test
from utils_dir import dataset, models
from utils_dir import my_utils

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


def get_abs_path(rel_path):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))


# ====================================================================
# CONFIG
# ====================================================================
with open(get_abs_path('training_config.yaml')) as f:
    config = yaml.safe_load(f)

MANIFEST_PATH    = config['data']['manifest_path']
CACHE_DIR        = config['data']['cache_dir']
MIN_STATION_HITS = config['data']['min_station_hits']
BATCH_SIZE       = config['training']['batch_size']
NUM_EPOCHS       = config['training']['num_epochs']
LEARNING_RATE    = config['training']['learning_rate']
WEIGHT_DECAY     = config['training']['weight_decay']
CHECKPOINT_FREQ  = config['training']['checkpoint_freq']
RADIUS_WEIGHT    = config['training'].get('radius_weight', 1.0)
DELTA            = config['training'].get('delta', 1)
LEAK_FACTOR      = config['training'].get('leak_factor', 0)
EARLY_STOPPING   = config['training'].get('early_stopping_patience', None)
WANDB_ID         = config['resume']['wandb_id']
CHECKPOINT_PATH  = config['resume']['checkpoint_path']
PROJECT_NAME     = config['wandb']['project']
WANDB_ENABLED    = config['wandb']['enabled']
NOTES            = config['wandb']['notes']
TAGS             = config['wandb']['tags']
GAMMA            = config['training'].get('gamma', 0.8)

model_params = {
    'hidden_units' : config['training']['hidden_units'],
    'leak_factor'  : LEAK_FACTOR,
    'dropout_rate' : config['training']['dropout_rate'],
    'temporal_res' : config['training'].get('temporal_res', 128),
}

if WANDB_ENABLED:
    wandb.init(
        project = PROJECT_NAME,
        id      = WANDB_ID,
        resume  = 'allow',
        notes   = NOTES,
        tags    = TAGS,
        config  = {
            'batch_size'    : BATCH_SIZE,
            'num_epochs'    : NUM_EPOCHS,
            'learning_rate' : LEARNING_RATE,
            'weight_decay'  : WEIGHT_DECAY,
            'radius_weight' : RADIUS_WEIGHT,
            'delta'         : DELTA,
            'manifest_path' : MANIFEST_PATH,
            **model_params
        }
    )

    cfg = dict(wandb.config)

    LEARNING_RATE = cfg.get('learning_rate', LEARNING_RATE)
    WEIGHT_DECAY  = cfg.get('weight_decay',  WEIGHT_DECAY)
    RADIUS_WEIGHT = cfg.get('radius_weight', RADIUS_WEIGHT)
    DELTA         = cfg.get('delta', DELTA)
    BATCH_SIZE    = cfg.get('batch_size',    BATCH_SIZE)
    NUM_EPOCHS    = cfg.get('num_epochs',    NUM_EPOCHS)
    LEAK_FACTOR   = cfg.get('leak_factor',   LEAK_FACTOR)

    for key in model_params:
        if key in cfg:
            model_params[key] = cfg[key]

# ====================================================================
# DATA
# ====================================================================
print("Staging data to scratch...")
manifest   = dataset.stage_manifest_to_scratch(MANIFEST_PATH, cache_dir=CACHE_DIR)
train_data = manifest['splits']['train']['files']
test_data  = manifest['splits']['test']['files']

print("Initializing datasets...")
train_album = dataset.ShardStreamIterableDataset(
    shard_file_list  = train_data,
    manifest_path    = MANIFEST_PATH,
    batch_size       = BATCH_SIZE,
    is_train         = True,
    min_station_hits = MIN_STATION_HITS,
    debug            = False
)

test_album = dataset.ShardStreamIterableDataset(
    shard_file_list  = test_data,
    manifest_path    = MANIFEST_PATH,
    batch_size       = BATCH_SIZE,
    is_train         = False,
    label_mean       = train_album.label_mean,
    label_std        = train_album.label_std,
    min_station_hits = MIN_STATION_HITS,
    debug            = False
)

TOT_IMGS              = manifest['metadata']['total_images_all_splits']
TOT_SHARDS            = manifest['metadata']['total_shards_all_splits']
avg_imgs_per_shard    = TOT_IMGS / TOT_SHARDS
TRAIN_PREFETCH_FACTOR = int(round(avg_imgs_per_shard / BATCH_SIZE) + 1)
TEST_PREFETCH_FACTOR  = TRAIN_PREFETCH_FACTOR * 2

print(f"Calculated train prefetch factor: {TRAIN_PREFETCH_FACTOR}")
print(f"Calculated test prefetch factor:  {TEST_PREFETCH_FACTOR}")

train_data_loader = DataLoader(
    dataset            = train_album,
    batch_size         = None,
    shuffle            = False,
    num_workers        = 4,
    prefetch_factor    = TRAIN_PREFETCH_FACTOR,
    pin_memory         = True,
    persistent_workers = True
)

test_data_loader = DataLoader(
    dataset            = test_album,
    batch_size         = None,
    shuffle            = False,
    num_workers        = 6,
    prefetch_factor    = TEST_PREFETCH_FACTOR,
    pin_memory         = True,
    persistent_workers = True
)

# ====================================================================
# MODEL
# ====================================================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    print("WARNING: GPU not available, training on CPU!")

model = models.RNO_four_branch_resnet(
    input_shape  = 1,
    output_shape = 3,
    label_mean   = train_album.label_mean,
    label_std    = train_album.label_std,
    **model_params
)

# ====================================================================
# OPTIMIZER / LOSS / SCHEDULER
# ====================================================================
optimizer = torch.optim.AdamW(
    params       = model.parameters(),
    lr           = LEARNING_RATE,
    weight_decay = WEIGHT_DECAY,
    fused        = True
)

# loss_fn = torch.nn.HuberLoss(delta=DELTA) # 1 sigma for regularization
loss_fn = torch.nn.MSELoss()
# loss_fn = models.RadiusWeightedMSELoss()
# loss_fn = my_utils.RadiusAwareHuberLoss(label_std=train_album.label_std, hardcoded_crossover=1.0475, delta=DELTA, radius_weight=RADIUS_WEIGHT)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode     = 'min',
#     factor   = 0.5,
#     patience = 5,
#     min_lr   = 1e-6,
# )

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,        # The number of epochs before the first restart
    T_mult=2,      # Multiplier. 1st cycle=10 epochs, 2nd=20, 3rd=40...
    eta_min=1e-5   # The lowest the learning rate will go before a restart
)

# scheduler = torch.optim.lr_scheduler.ExponentialLR(
#     optimizer,
#     gamma=GAMMA,
# )

# scheduler = None # No scheduler training

# ====================================================================
# MODEL CONFIG
# ====================================================================
model_config = {
    'model_class'     : model.__class__.__name__,
    'input_shape'     : 1,
    'output_shape'    : 3,
    'hidden_units'    : model_params['hidden_units'],
    'leak_factor'     : model_params['leak_factor'],
    'temporal_res'    : model_params['temporal_res'],
    'manifest_path'   : MANIFEST_PATH,
    'min_station_hits': MIN_STATION_HITS,
    'batch_size'      : BATCH_SIZE,
    'learning_rate'   : LEARNING_RATE,
    'weight_decay'    : WEIGHT_DECAY,
    'loss_fn'         : loss_fn.__class__.__name__,
    'notes'           : NOTES,
}

if any(isinstance(m, torch.nn.Dropout) for m in model.modules()):
    model_config['dropout_rate'] = model_params['dropout_rate']

if isinstance(loss_fn, my_utils.RadiusAwareHuberLoss):
    model_config['radius_weight'] = RADIUS_WEIGHT
    model_config['crossover_R']   = float(loss_fn.crossover_R)  # type: ignore
    model_config['relu_slope']    = loss_fn.relu_slope
    model_config['delta']         = loss_fn.huber.delta
elif isinstance(loss_fn, torch.nn.HuberLoss):
    model_config['delta'] = DELTA

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

# Prefix with SWEEP if this run is part of a W&B sweep
if WANDB_ENABLED and wandb.run and wandb.run.sweep_id:
    experiment_name = f'SWEEP_{experiment_name}'

# ====================================================================
# COMPILE MODEL
# ====================================================================
model.to(DEVICE)
model = torch.compile(model)  # type: ignore

# ====================================================================
# WANDB RUN CONFIG
# ====================================================================
if WANDB_ENABLED:
    wandb.run.name = experiment_name  # type: ignore

    wandb.config.update({
        'n_train_batches': len(train_data_loader),
        'n_test_batches' : len(test_data_loader),
        'optimizer'      : optimizer.__class__.__name__,
        'loss_fn'        : loss_fn.__class__.__name__,
    })

    wandb.watch(model, log='all', log_freq=len(train_data_loader)//4)  # type: ignore
    wandb.define_metric('Test Loss',      summary='min')
    wandb.define_metric('Train Loss',     summary='min')
    wandb.define_metric('best_test_loss', summary='min')

    manifest_artifact = wandb.Artifact('dataset_manifest', type='dataset')
    manifest_artifact.add_file(MANIFEST_PATH)
    wandb.log_artifact(manifest_artifact)

# ====================================================================
# CHECKPOINT DIR
# ====================================================================
run_id         = wandb.run.id if WANDB_ENABLED else experiment_name  # type: ignore
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
    model                   = model,  # type: ignore
    train_dataloader        = train_data_loader,
    test_dataloader         = test_data_loader,
    optimizer               = optimizer,
    loss_fn                 = loss_fn,
    scheduler               = scheduler,
    device                  = DEVICE,
    epochs                  = NUM_EPOCHS,
    checkpoint_dir          = checkpoint_dir,
    checkpoint_freq         = CHECKPOINT_FREQ,
    checkpoint_path         = CHECKPOINT_PATH,
    early_stopping_patience = EARLY_STOPPING,
    model_config            = model_config
)
