"""
Train/Test loop for RNO vertex reconstruction models.

Author: Santiago Sued
"""

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from .train_test_steps.train import train_step
from .train_test_steps.test import test_step
from .my_utils import save_checkpoint, load_checkpoint
import traceback
import wandb
import torch
import time
import os


def train_test(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               test_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: torch.device | str,
               epochs: int,
               checkpoint_dir: str,
               checkpoint_freq: int,
               checkpoint_path: str | None = None,
               scheduler: LRScheduler | ReduceLROnPlateau | None = None,
               min_test_loss_epoch: bool = True,
               model_config: dict | None = None):
    """
    Full training and evaluation loop with checkpointing and WandB logging.

    Runs train/test steps for each epoch. Checkpoints are saved periodically
    and whenever a new best test loss is achieved. All metrics are logged to
    WandB if an active run exists (wandb.run is not None).

    On any exit — clean finish, KeyboardInterrupt, or crash — a final
    checkpoint is saved if valid losses exist, and the WandB run is closed.

    Args:
        model            : PyTorch model to train. Moved to device and compiled internally.
        train_dataloader : DataLoader for training data.
        test_dataloader  : DataLoader for test/validation data.
        optimizer        : Optimizer (e.g., Adam).
        loss_fn          : Loss function (e.g., MSELoss).
        device           : Device to run on (CPU/CUDA).
        epochs           : Number of epochs to train for.
        checkpoint_dir   : Directory to save checkpoint .pth files.
        checkpoint_freq  : Save a periodic checkpoint every N epochs.
        checkpoint_path  : Path to a .pth file to resume from. None = train from scratch.
        scheduler        : Optional LR scheduler. ReduceLROnPlateau is stepped with
                           test_loss; all other schedulers are stepped blindly each epoch.
        min_test_loss_epoch: If True, saves a separate best-model checkpoint whenever
                           test loss improves. Previous best is deleted to save disk space.
    """

    train_loss    = None
    test_loss     = None
    test_loss_min = float('inf')

    # Load checkpoint if resuming — returns the next epoch to train from
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, device, checkpoint_path, model_config=model_config)
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Initialize final_epoch before the loop in case we exit before any epoch completes
    final_epoch = start_epoch

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            epoch_start = time.time()

            # --- Training step ---
            train_loss = train_step(
                model       = model,
                data_loader = train_dataloader,
                loss_fn     = loss_fn,
                optimizer   = optimizer,
                device      = device,
            )

            # --- Test step ---
            test_loss = test_step(
                model       = model,
                data_loader = test_dataloader,
                loss_fn     = loss_fn,
                device      = device,
            )

            epoch_time = time.time() - epoch_start

            # --- Epoch summary ---
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Time: {epoch_time:.1f}s")

            # --- WandB epoch-level logging ---
            # step=epoch ensures correct x-axis alignment when resuming mid-run
            if wandb.run:
                wandb.log({
                    'Train Loss'   : train_loss,
                    'Test Loss'    : test_loss,
                    'Epoch Time'   : epoch_time,
                    'Epoch'        : epoch,
                    'Learning Rate': optimizer.param_groups[0]['lr']
                })

            # --- Best model checkpoint ---
            if min_test_loss_epoch and test_loss < test_loss_min:
                test_loss_min = test_loss
                best_path = save_checkpoint(epoch, model, optimizer, scheduler,
                                            train_loss, test_loss, checkpoint_dir, model_config=model_config, min=True)

                # Upload best checkpoint to WandB — only best model, not every periodic
                # checkpoint, to avoid excessive artifact storage
                if wandb.run:
                    artifact = wandb.Artifact(
                        name     = f'best-checkpoint-{wandb.run.id}',
                        type     = 'model',
                        metadata = {'epoch': epoch, 'test_loss': test_loss}
                    )
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)

                # Remove previous best checkpoint to keep disk usage minimal
                for filename in os.listdir(checkpoint_dir):
                    if 'min_' in filename and f'min_e{epoch}' not in filename:
                        try:
                            os.remove(os.path.join(checkpoint_dir, filename))
                        except OSError as e:
                            print(f"WARNING: Could not remove old checkpoint {filename}: {e}")

            # --- Periodic checkpoint ---
            if epoch % checkpoint_freq == 0 and epoch != 0:
                save_checkpoint(epoch, model, optimizer, scheduler,
                                train_loss, test_loss, checkpoint_dir, model_config=model_config)

            # --- Scheduler step ---
            # ReduceLROnPlateau requires the monitored metric to decide whether
            # to reduce LR; all other schedulers advance on a fixed schedule
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            final_epoch = epoch

    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            print("Training interrupted by user (Ctrl+C)")
        elif isinstance(e, SystemExit):
            print("Training terminated by system signal")
        else:
            print(f"Training failed at epoch {final_epoch}")
            print(traceback.format_exc())

    finally:
        # Always attempt a final checkpoint on any exit — clean or otherwise
        if train_loss is not None and test_loss is not None:
            save_checkpoint(final_epoch, model, optimizer, scheduler,
                            train_loss, test_loss, checkpoint_dir, model_config=model_config)
            print(f"Final checkpoint saved at epoch {final_epoch} in {checkpoint_dir}")
        else:
            print(f"Exited at epoch {final_epoch} with no recorded losses — no checkpoint saved.")

        # Log final summary values and close WandB run cleanly
        if wandb.run:
            wandb.summary['final_epoch']    = final_epoch
            wandb.summary['best_test_loss'] = test_loss_min
            wandb.finish()