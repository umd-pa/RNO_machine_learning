import torch
import os

def move_optimizer_to_device(optimizer, device):
    """Move optimizer state to specified device"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                # Keep step tensor on CPU for Adam with capturable=False
                if k == 'step':
                    state[k] = v.cpu()
                else:
                    state[k] = v.to(device)

def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, test_loss, checkpoints_dir, logger):
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                  'train_loss': train_loss,
                  'test_loss': test_loss,
                 }
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = f'{checkpoints_dir}/checkpoint_e{epoch}.pth'
    logger.info(f'Saving checkpoint at epoch {epoch} in: {checkpoint_path}')
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, scheduler, logger, device, checkpoint_path):
    try:
        # Load checkpoint onto CPU to avoid CUDA OOM during simple eval setup
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle potential state_dict nesting
        state_dict = loaded_checkpoint['model_state_dict'] if 'model_state_dict' in loaded_checkpoint else loaded_checkpoint
        optimizer_dict = loaded_checkpoint['optimizer_state_dict'] if 'optimizer_state_dict' in loaded_checkpoint else loaded_checkpoint

    except FileNotFoundError:
        logger.error(f'Could not find checkpoint: {checkpoint_path}')
        raise

    # load states
    model.load_state_dict(_uncompile_keys(state_dict))
    optimizer.load_state_dict(_uncompile_keys(optimizer_dict))
    move_optimizer_to_device(optimizer, device)
    if loaded_checkpoint['scheduler_state_dict'] is not None and scheduler is not None:
        scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])

    return loaded_checkpoint['epoch'] + 1
    
def _uncompile_keys(state_dict: dict) -> dict:
    """Helper to remove torch.compile prefix '_orig_mod.' from compiled state dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "") 
        new_state_dict[new_key] = v
    return new_state_dict