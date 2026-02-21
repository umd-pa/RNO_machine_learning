from .train_test_steps.train import train_step
from .train_test_steps.test import test_step
from .utils import save_checkpoint, load_checkpoint
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
               scheduler: torch.optim.lr_scheduler,
               loss_fn: torch.nn.Module,
               device: torch.device,
               experiment_name: str,
               logger: logging.Logger,
               epochs: int = 5,
               checkpoint_freq: int = 100,
               checkpoint_name: str = None,
               loss_file: str = 'losses.txt'):

    # Final epoch for logging
    final_epoch = 0

    # Initialize experiment_dir
    experiment_dir = os.path.join('experiments',experiment_name)

    # Initialize checkpoint_dir
    checkpoints_dir = os.path.join(experiment_dir,'checkpoints')
    
    # Use model name for project name
    project_name = type(model).__name__

    # Instantiate loss and optimizer name
    loss_fn_name = loss_fn.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    
    # Log into wandb and initialize
    wandb.login()
    wandb.init(project=project_name,
               name=experiment_name, # Pass a runname
              config={"dataset": len(train_dataloader.dataset),
                      "epochs": epochs,
                      "loss_fn": loss_fn_name,
                      "optimizer_name": optimizer_name,
                     })
    # Utilize checkpoint if specified
    if checkpoint_name:
        logger.info(f'Checkpoint specified, extracting checkpoint state from: {checkpoint_name}')
        
        start_epoch = load_checkpoint(model, optimizer, scheduler, logger, device, checkpoints_dir, checkpoint_name)
        
        logger.info('Checkpoint loaded successfully')
    else:
        start_epoch = 0

    # Move model parameters to the specified device
    model.to(device)

    # Track timing
    start_time = time.time()

    loss_file_path = os.path.join(experiment_dir,loss_file)
    file_exists = os.path.exists(loss_file_path)
    
    with open(loss_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If file already exists, skip headers
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
    
        try:
            for epoch in range(start_epoch, start_epoch + epochs):
                # Epoch start time
                epoch_start = time.time()
                
                # --- Training step ---
                train_loss = train_step(model=model,
                                            data_loader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
    
                # --- Testing step ---
                test_loss = test_step(model=model,
                                      data_loader=test_dataloader,
                                      loss_fn=loss_fn,
                                      device=device)
                
                # Scheduler step if present
                if scheduler:
                    scheduler.step()

                # Write losses to csv
                writer.writerow([epoch,train_loss,test_loss])
                csvfile.flush()

                # Log losses to logger
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                
                # Obtain time elapsed
                epoch_end = time.time()
                
                # Log losses to wandb
                wandb.log({"Train Loss": train_loss,
                           "Test Loss": test_loss,
                           "Epoch Time": epoch_end-epoch_start,
                           "stop_file_exists": int(os.path.exists('STOP_TRAINING'))})
                
                if epoch % checkpoint_freq == 0 and epoch > 0:
                    save_checkpoint(epoch,model,optimizer,scheduler,train_loss,test_loss,checkpoints_dir,logger)

                final_epoch = epoch

                # Stopping framework
                if os.path.exists(os.path.join(experiment_dir,'STOP_TRAINING')):
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
                save_checkpoint(final_epoch,model,optimizer,scheduler,train_loss,test_loss,checkpoints_dir,logger)
                logger.info(f'Code exited on epoch: {final_epoch} and a checkpoint was created in {checkpoints_dir}')
                wandb.finish()    
            else:
             logger.warning(f'Code exited on epoch: {final_epoch} but no losses were detected.')