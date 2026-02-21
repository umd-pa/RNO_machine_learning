from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from .train_test_steps.train import train_step
from .train_test_steps.test import test_step
from .my_utils import save_checkpoint, load_checkpoint
import numpy as np
import traceback
import logging
import wandb
import torch
import time
import csv
import os

def train_test(model: torch.nn.Module, 
               train_dataloader: torch.utils.data.DataLoader, 
               test_dataloader: torch.utils.data.DataLoader, 
               optimizer: torch.optim.Optimizer,
               scheduler: LRScheduler | ReduceLROnPlateau | None,
               loss_fn: torch.nn.Module,
               device: torch.device | str,
               experiment_name: str,
               logger: logging.Logger,
               epochs: int = 5,
               checkpoint_freq: int = 100,
               checkpoint_path: str | None = None,
               loss_file: str = 'losses.txt',
               wandb_id: str | None = None,
               min_test_loss_epoch: bool = True):

    # Final epoch for logging
    final_epoch = 0

    # Initialize experiment_dir
    experiment_dir = os.path.join('experiments',experiment_name)

    # Initialize checkpoint_dir
    checkpoints_dir = os.path.join(experiment_dir,'checkpoints')
    
    # Use folder name for project name
    project_name = os.path.basename(os.getcwd())

    # Instantiate loss and optimizer name
    loss_fn_name = loss_fn.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    
    # Log into wandb and initialize
    wandb.login()
    wandb.init(project=project_name,
               name=experiment_name,
               id=wandb_id,
               resume='allow',
              config={"dataset": train_dataloader.dataset.path, # type: ignore (path is a field in my AlbumDataset class)
                      "epochs": epochs,
                      "loss_fn": loss_fn_name,
                      "optimizer_name": optimizer_name,
                     })
    # Utilize checkpoint if specified
    if checkpoint_path:
        logger.info(f'Checkpoint specified, extracting checkpoint state from: {checkpoint_path}')

        start_epoch = load_checkpoint(model, optimizer, scheduler, logger, device, checkpoint_path)
        
        logger.info('Checkpoint loaded successfully')
    else:
        start_epoch = 0

    # Move model parameters to the specified device after compiling to make more efficient
    model = model.to(device) # Move first
    model = torch.compile(model) # type: ignore (Ignore the fact that model is no longer a nn.Module model after being compiled!)

    # Track timing
    start_time = time.time()

    loss_file_path = os.path.join(experiment_dir,loss_file)
    file_exists = os.path.exists(loss_file_path)
    
    train_loss = None
    test_loss = None
    test_loss_min = float('inf')

    with open(loss_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If file already exists, skip headers
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])

        try:
            for epoch in range(start_epoch, start_epoch + epochs):
                # Save 0 epoch before any training step:
                if epoch == 0:
                    save_checkpoint(epoch,model,optimizer,scheduler,0,0,checkpoints_dir,logger)

                # Epoch start time
                epoch_start = time.time()

                # --- Training step ---
                train_loss = train_step(model=model,
                                            data_loader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device,
                                            logger=logger)
    
                # --- Testing step ---
                test_loss = test_step(model=model,
                                      data_loader=test_dataloader,
                                      loss_fn=loss_fn,
                                      device=device,
                                      logger=logger)
                
                if min_test_loss_epoch:
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        # Save new minimum
                        save_checkpoint(f'min_{epoch}',model,optimizer,scheduler,0,0,checkpoints_dir,logger)
                        # remove previous minimum
                        for filename in os.listdir(checkpoints_dir):
                            if 'min_' in filename and f'min_{epoch}' not in filename:
                                old_min_path = os.path.join(checkpoints_dir,filename)
                                try:
                                    os.remove(old_min_path)
                                    logger.info(f"Removed previous best checkpoint: {filename}")
                                except OSError as e:
                                    logger.warning(f"Failed to remove old checkpoint {filename}: {e}")

                # Scheduler step if present
                if scheduler:
                    pass # pass for now
                    #scheduler.step() 

                # Write losses to csv
                writer.writerow([epoch,train_loss,test_loss])
                csvfile.flush()

                # Log losses to logger
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                
                # Obtain time elapsed
                epoch_end = time.time()
                
                # Log losses to wandb with EXPLICIT STEP
                wandb.log({"Train Loss": train_loss,
                           "Test Loss": test_loss,
                           "Epoch Time": epoch_end-epoch_start,
                           "stop_file_exists": int(os.path.exists('STOP_TRAINING'))},
                           step=epoch) # Forces alignment with epoch number
                
                if epoch % checkpoint_freq == 0 and epoch != 0:
                    save_checkpoint(epoch,model,optimizer,scheduler,train_loss,test_loss,checkpoints_dir,logger)

                final_epoch = epoch

                # Stopping framework
                if os.path.exists(os.path.join(experiment_dir,'STOP_TRAINING')): # If STOP_TRAINING file exists in experiment directory
                    logger.warning(f"STOP FILE DETECTED AT {epoch}. Stopping training...")
                    os.remove(os.path.join(experiment_dir,'STOP_TRAINING'))  # Clean up the stop file
                    break

            logger.info('Training successful')
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                logger.info("Training interrupted by user (Ctrl+C)")
            elif isinstance(e, SystemExit):
                logger.info("Training terminated by system signal")
            else:
                logger.error(f"Training failed at epoch {final_epoch}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
        finally:
            if os.path.exists('STOP_TRAINING'):
                os.remove('STOP_TRAINING')
            # Only save final checkpoint if we have valid loss values
            if 'train_loss' in locals() and 'test_loss' in locals():
                save_checkpoint(final_epoch,model,optimizer,scheduler,train_loss,test_loss,checkpoints_dir,logger) # unbound: ignore
                logger.info(f'Code exited on epoch: {final_epoch} and a checkpoint was created in {checkpoints_dir}')
                wandb.finish()    
            else:
             logger.warning(f'Code exited on epoch: {final_epoch} but no losses were detected.')