import torch
import os

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

def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, test_loss, checkpoints_dir, model_config=None, min=False) -> str:
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
        'model_config'        : model_config
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

    # Return next epoch so training resumes from where it left off
    return loaded_checkpoint['epoch'] + 1
    
def _uncompile_keys(state_dict: dict) -> dict:
    """Helper to remove torch.compile prefix '_orig_mod.' from compiled state dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "") 
        new_state_dict[new_key] = v
    return new_state_dict