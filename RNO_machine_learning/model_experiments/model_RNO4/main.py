import os
os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"

from torch.utils.data import DataLoader
from torch import Tensor
import logging
import torch
import time
import sys

# Enable TF32 for faster matrix multiplications
torch.set_float32_matmul_precision('high')

# Enable benchmarking (runs a race of convolution algorithms to determine which one is better)
torch.backends.cudnn.benchmark = True # Use for static input size and GPUs

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all functions from utils_dir (handled by __init__.py)
from utils_dir.train_test import train_test # noqa: E402
from utils_dir.dataset import ShardAlbumDataset # noqa: E402
from utils_dir import models #noqa: E402

# Only function if current working directory is model dir:
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

if 'model_' not in os.path.basename(current_dir):
    raise ValueError("Must run from a directory containing 'model_' in the name")
    
if 'experiments' in os.path.basename(current_dir):
    raise ValueError("Cannot run from within experiments directory")
    
print('✅ Inside correct folder')

# ====================================================================
# PARAMS
# ====================================================================
BATCH_SIZE = 256
NUM_EPOCHS = int(100_000)
checkpoint_path = None
CHECKPOINT_FREQ = 50
NORMALIZE_INPUTS = True
LEAK_FACTOR = 0.1 
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.001
HIDDEN_UNITS = 32
WANDB_ID = None
TEMPORAL_RES = 256
CARTESIAN = True # <-- Added to prevent NameError in experiment_name. But deprecated since all data should be in Cartesian
# ====================================================================

if checkpoint_path is not None:
    print('<<<<<<<<<<WARNING: UTILIZING CHECKPOINT>>>>>>>>>>')
if WANDB_ID is not None:
    print(f'<<<<<<<<<<WARNING: UTILIZING RUN_ID: {WANDB_ID}>>>>>>>>>>')

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = "cpu"

if device=="cpu":
    print(device)
    print('ERROR: Not utilizing GPU!')
    # sys.exit(1)
    
print(torch.__version__)

# ====================================================================
# PATHS
# ====================================================================
train_vds_path = '/data/i3store/users/ssued/albums/sharded/rno_sim_shards_v2_max/train.vds'
test_vds_path  = '/data/i3store/users/ssued/albums/sharded/rno_sim_shards_v2_max/test.vds'

# ====================================================================
# INITIALIZE DATASETS & EXTRACT NORMALIZATION STATS
# ====================================================================
print('Initializing album datasets...')

# By setting is_train=True, the dataset automatically computes the stats for us!
train_album = ShardAlbumDataset(train_vds_path, is_train=True)

if not isinstance(train_album.label_mean, Tensor)or not isinstance(train_album.label_std, Tensor):
    raise TypeError("Expected statistics to be a torch.Tensor!")

# Extract the freshly computed stats to share with the test set and model
label_mean = train_album.label_mean.numpy()
label_std = train_album.label_std.numpy()

# Initialize test set with the train stats so they share the same normalized space
test_album = ShardAlbumDataset(test_vds_path, is_train=False, label_mean=label_mean, label_std=label_std)

print(f'Train album size: {len(train_album)} | Test album size: {len(test_album)}')

# ====================================================================
# LOAD DATALOADERS
# ====================================================================
print('Initializing album data loaders...')
train_data_loader = DataLoader(dataset = train_album,
                               batch_size = BATCH_SIZE,
                               shuffle = True,
                               num_workers = 12,
                               prefetch_factor=16, # 256 X 12 X 8 is 24576 images given to the GPU
                               pin_memory=True,
                               persistent_workers=True, # Important!
                               drop_last=True) # Will drop the last batch which may not have 256 elements!

test_data_loader = DataLoader(dataset = test_album,
                              batch_size = BATCH_SIZE,
                              shuffle = False,
                              num_workers = 8, # Had 4 but now prefetch went from 2 -> 1
                              prefetch_factor = 4, # From 
                              pin_memory=True,
                              persistent_workers=True)

print(f'Number of train batches: {len(train_data_loader)} | Number of test batches: {len(test_data_loader)}')

# ====================================================================
# INITIALIZE MODEL (Passing the normalization stats for PyTorch Buffers)
# ====================================================================
print('Initializing model...')
model = models.RNO_four_late_non_linear_merge( #
                          input_shape=1,
                          hidden_units=HIDDEN_UNITS,
                          output_shape=3,
                          num_epochs=NUM_EPOCHS,
                          batch_size=BATCH_SIZE,
                          num_train_batches=len(train_data_loader),
                          leak_factor=LEAK_FACTOR,
                          dropout_rate=DROPOUT_RATE,
                          temporal_res=TEMPORAL_RES,
                          label_mean=label_mean,
                          label_std=label_std
                          )

model = model.to(device)

# ====================================================================
# SETUP OPTIMIZER & LOGGING
# ====================================================================
optimizer = torch.optim.Adam(params=model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, fused = True)
optimizer_name = optimizer.__class__.__name__

loss_fn = torch.nn.MSELoss()
loss_fn_name = loss_fn.__class__.__name__

experiment_name = (f'exp_{model.__class__.__name__}' +
                  f'_b-{BATCH_SIZE}' +
                  f'_tr-{len(train_data_loader)}' +
                  f'_lfn-{loss_fn_name}' +
                  f'_opt-{optimizer_name}' +
                  f'_wdcay-{WEIGHT_DECAY:.1e}'
                  f'_hiddenu-{HIDDEN_UNITS}' +
                  f'_lr-{LEARNING_RATE:.2e}' +
                  f'_leak-{LEAK_FACTOR}' +
                  f'_tempRes-{TEMPORAL_RES}' +
                  f'_dpout-{DROPOUT_RATE}' +
                  '_bin_mode=MAX'
                  '_debug'
                 )

# Create experiments directory if it doesn't exist
os.makedirs('experiments', exist_ok=True)
experiment_path = os.path.join('experiments', experiment_name)

# Warn user if experiment already exists and wait for confirmation
if os.path.exists(experiment_path):
    print('WARNING: Experiment with this name already exists. Run data will be saved to this existing directory. ' \
    'To continue create a file titled "y" in the experiment directory',flush=True)
    start = time.time()
    timeout = 60  # seconds
    while True:
        if os.path.exists(os.path.join(experiment_path, 'y')):
            print('"y" file spotted, continuing...',flush=True)
            os.remove(os.path.join(experiment_path, 'y'))
            break
        if time.time() - start >= timeout:
            print('Timeout reached (60s), exiting program...')
            sys.exit(1)
        time.sleep(1)
else:
    os.makedirs(experiment_path, exist_ok=True)

# Setup logging
logger = logging.getLogger('experiment_log') # Setup logging
logging.basicConfig(filename=f'{experiment_path}/experiment.log',
                    filemode='w',
                    level=logging.INFO,
                    format='[%(levelname)s: %(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger.info(f"Starting experiment: {experiment_name}")
logger.info(f"Device: {device}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Model: {type(model).__name__}")
logger.info(f"Optimizer: {optimizer_name}")
logger.info(f"Loss function: {loss_fn_name}")

# ====================================================================
# RUN TRAINING LOOP
# ====================================================================
train_test(model = model, 
           train_dataloader = train_data_loader, 
           test_dataloader = test_data_loader, 
           optimizer = optimizer,
           scheduler = None,
           loss_fn = loss_fn,
           device = device,
           experiment_name = experiment_name,
           epochs = NUM_EPOCHS,
           checkpoint_freq = CHECKPOINT_FREQ,
           checkpoint_path = checkpoint_path,
           loss_file = 'losses.txt',
           logger = logger,
           wandb_id=WANDB_ID)
