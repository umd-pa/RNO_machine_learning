from collections import OrderedDict
from torch import nn
import torch
import os

class RadiusAwareHuberLoss(nn.Module):
    """
    Huber loss with a soft radius penalty that activates beyond a physically
    motivated crossover radius. Self-consistent — crossover is computed
    internally from label_std and crossover_physical, requiring no external
    calibration from previous models.

    The ReLU threshold activates at crossover_physical meters (physical space),
    converted to normalized space using the geometric mean of label_std.
    Beyond the crossover, the penalty grows linearly with distance at rate
    relu_slope, then is scaled by radius_weight relative to the Cartesian
    Huber loss.

    Args:
        label_std          : Per-coordinate std from training set (3,) tensor.
                             Used to convert crossover_physical to normalized
                             space via geometric mean.
        crossover_physical : Physical radius (meters) at which penalty activates.
                             From show_error_vs_radius — where rel_R crosses zero.
                             Default: 750m.
        hardcoded_crossover: If provided, skips the geometric mean calculation
                             and uses this value directly as crossover_R in
                             normalized space. Use the empirical value from
                             show_error_vs_radius (e.g. 1.0475 for this dataset).
        radius_weight      : Overall weight of radius penalty vs Huber loss.
                             Tune using cartesian_loss / radius_loss ratio at
                             initialization. Default: 0.5.
        delta              : Huber loss transition point. Default: 1.0.
        relu_slope         : Controls how fast penalty grows beyond crossover.
                             1.0 = linear, 2.0 = more aggressive for distant
                             events. Default: 1.0.
    """
    def __init__(self,
                 label_std,
                 crossover_physical: float = 750.0,
                 hardcoded_crossover: float | None = None,
                 radius_weight: float = 0.5,
                 delta: float = 1.0,
                 relu_slope: float = 1.0):
        super().__init__()
        self.huber         = nn.HuberLoss(delta=delta)
        self.radius_weight = radius_weight
        self.relu_slope    = relu_slope

        # Compute crossover in normalized space
        if hardcoded_crossover is not None:
            # Use empirical value directly — more accurate than any approximation
            # for datasets with large mean offsets (e.g. z mean = -782m)
            crossover_val = hardcoded_crossover
        else:
            # Geometric mean of stds — appropriate for Euclidean distance
            # across three axes. Better than arithmetic mean which is biased
            # toward the dominant x/y stds and underweights z.
            label_std_tensor = torch.as_tensor(label_std, dtype=torch.float32)
            geom_std         = torch.prod(label_std_tensor) ** (1/3)
            crossover_val    = crossover_physical / geom_std.item()

        # register_buffer creates self.crossover_R as a tensor attribute
        # that automatically moves to the correct device with loss_fn.to(device)
        self.register_buffer('crossover_R',
                             torch.tensor(crossover_val, dtype=torch.float32))

        print("RadiusAwareHuberLoss initialized:")
        print(f"  crossover_physical = {crossover_physical:.0f} m")
        print(f"  crossover_R (norm) = {crossover_val:.4f}")
        print(f"  radius_weight      = {radius_weight}")
        print(f"  relu_slope         = {relu_slope}")

    def forward(self, reco: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # Standard Huber loss on all Cartesian coordinates
        cartesian_loss = self.huber(reco, true)

        # Normalized radius for each event in the batch
        reco_R = torch.sqrt((reco**2).sum(dim=1))
        true_R = torch.sqrt((true**2).sum(dim=1))

        # Soft threshold via ReLU — zero below crossover_R, growing linearly
        # above it. relu_slope controls steepness: higher values penalize
        # distant events more aggressively relative to near-crossover events.
        # The weight naturally scales the penalty with distance beyond crossover,
        # so physically distant events contribute more to the radius loss.
        weight      = torch.relu(true_R - self.crossover_R) * self.relu_slope#type:ignore
        radius_loss = torch.mean(weight * (reco_R - true_R)**2)

        return cartesian_loss + self.radius_weight * radius_loss
def move_optimizer_to_device(optimizer, device):
    """Move optimizer state to specified device"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                # Keep step tensor on CPU for Adam with capturable=False
                # if k == 'step':
                #     state[k] = v.cpu() NOT GOOD FOR FUSED ADAM!
                # else:
                state[k] = v.to(device)

def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, test_loss, checkpoints_dir, test_loss_min=float('inf'), model_config=None, min=False) -> str:
    """
    Saves model, optimizer, and scheduler states to a .pth checkpoint file.

    Args:
        epoch          : Current epoch number, used in the filename.
        model          : The model to save.
        optimizer      : The optimizer to save.
        scheduler      : The LR scheduler to save, or None.
        train_loss     : Training loss at this epoch.
        test_loss      : Test loss at this epoch.
        checkpoints_dir: Directory to save the checkpoint in.
        min            : If True, saves as a 'best model' checkpoint.

    Returns:
        str: Path to the saved checkpoint file.
    """
    checkpoint = {
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss'          : train_loss,
        'test_loss'           : test_loss,
        'model_config'        : model_config,
        'test_loss_min'       : test_loss_min
    }

    os.makedirs(checkpoints_dir, exist_ok=True)

    if min:
        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_min_e{epoch}.pth')
    else:
        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_e{epoch}.pth')

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, device, checkpoint_path, model_config=None):
    """
    Loads model, optimizer, and scheduler states from a .pth checkpoint file.
    Handles both compiled and uncompiled model state dicts in both directions:
    - Checkpoint saved from compiled model, loading into uncompiled → strips _orig_mod. prefix
    - Checkpoint saved from uncompiled model, loading into compiled → loads via _orig_mod

    Optionally validates that the checkpoint's model_config matches the current
    run's model_config before loading weights. This catches accidental cross-architecture
    resumption early — before weights are loaded and training silently corrupts.
    Validation is skipped gracefully if either config is None (old checkpoints).
    
    Args:
        model          : The model to load state into (compiled or uncompiled).
        optimizer      : The optimizer to load state into.
        scheduler      : The LR scheduler to load state into, or None.
        device         : Device to move optimizer state to after loading.
        checkpoint_path: Path to the .pth checkpoint file.
        model_config   : The current run's model config dict from main.py.
                         If None, validation is skipped for backward compatibility.

    Returns:
        int: The next epoch to train from (saved epoch + 1).
    """
    try:
        # Load onto CPU first to avoid CUDA OOM on machines with limited VRAM
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle both nested and flat state dict formats
        state_dict     = loaded_checkpoint.get('model_state_dict',     loaded_checkpoint)
        optimizer_dict = loaded_checkpoint.get('optimizer_state_dict', loaded_checkpoint)

    except FileNotFoundError:
        print(f"ERROR: Could not find checkpoint: {checkpoint_path}")
        raise

    # Model configuration validation
    ckpt_config = loaded_checkpoint.get('model_config')
    if ckpt_config is not None and model_config is not None:
        critical_fields = ['model_class', 'hidden_units', 'input_shape', 'output_shape']

        mismatches = []
        for field in critical_fields:
            ckpt_val    = ckpt_config.get(field)
            current_val = model_config.get(field)
            if ckpt_val != current_val:
                mismatches.append(
                    f"  {field}: checkpoint={ckpt_val}, current={current_val}"
                )
        if mismatches:
            raise ValueError(
                "Model config mismatch between checkpoint and current run!\n"
                + '\n'.join(mismatches)
                + "\nFix your training_config.yaml or point to the correct checkpoint."
            )
        print("Config validation passed — checkpoint matches current architecture.")



    # Handle compiled vs uncompiled mismatch in both directions:
    # - If model is compiled (_orig_mod exists) but checkpoint is uncompiled → load via _orig_mod
    # - If checkpoint is compiled (_orig_mod. prefix in keys) but model is not → strip prefix
    if hasattr(model, '_orig_mod'):
        # Model is compiled — load directly into the underlying uncompiled module
        model._orig_mod.load_state_dict(_uncompile_keys(state_dict))
    else:
        # Model is not compiled — strip prefix if checkpoint was saved from compiled model
        model.load_state_dict(_uncompile_keys(state_dict))

    optimizer.load_state_dict(_uncompile_keys(optimizer_dict))

    # Optimizer state is loaded on CPU — move to correct device
    move_optimizer_to_device(optimizer, device)

    # Restore scheduler state if both exist
    if loaded_checkpoint.get('scheduler_state_dict') is not None and scheduler is not None:
        scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])

    # Return next epoch so training resumes from where it left off and the minimum test_loss
    saved_min = loaded_checkpoint.get('test_loss_min', float('inf'))
    return loaded_checkpoint['epoch'] + 1, saved_min
    
def _uncompile_keys(state_dict: dict) -> dict:
    """Helper to remove torch.compile prefix '_orig_mod.' from compiled state dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "") 
        new_state_dict[new_key] = v
    return new_state_dict

def auto_name(*layers):
    """Automatically names PyTorch layers for W&B logging (e.g., Conv3d_1, Conv3d_2)"""
    named_layers = OrderedDict()
    counts = {}
    for layer in layers:
        layer_type = layer.__class__.__name__
        counts[layer_type] = counts.get(layer_type, 0) + 1
        named_layers[f"{layer_type}_{counts[layer_type]}"] = layer
    return named_layers
