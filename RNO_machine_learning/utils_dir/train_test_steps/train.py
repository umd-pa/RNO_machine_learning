from torch.amp.grad_scaler import GradScaler
import logging
import torch
import time

def train_step(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    loss_fn: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device | str, 
    logger: logging.Logger, 
    scaler: GradScaler | None = None
):
    """
    Performs one training epoch over the entire batched dataset.
    
    Args:
        model: PyTorch neural network model to train
        dataloader: DataLoader containing training batches
        loss_fn: Loss function (e.g., CrossEntropyLoss, MSELoss)
        optimizer: Optimization algorithm (e.g., Adam, SGD)
        device: Device to run computations on (CPU/CUDA)
        logging: Logger for logging training progress
        scaler: GradScaler instance for adaptive gradient scaling in mixed precision
    
    Returns:
        float: Average training loss across all batches in the epoch
    """
    model.train()

    batch_train_losses = []
    total_imgs = 0
    start_time = time.time()

    for batch, (X, y) in enumerate(data_loader):
        # Data Preparation
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # Reset Gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward Pass with Mixed Precision
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            y_pred = model(X)
            y = y.squeeze(1)
            loss = loss_fn(y_pred, y)

        # Backward Pass (Scaled to prevent underflow)
        scaler.scale(loss).backward()

        # Optimizer Step (Unscales gradients and steps if they are valid)
        scaler.step(optimizer)

        # Update Scaler for the next batch
        scaler.update()

        # Metrics Tracking
        total_imgs += len(X)
        batch_train_losses.append(loss.item())

        # Logging based on specified batch interval
        if batch > 0 and batch % 50 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            avg_speed = total_imgs / elapsed
            logger.info(f"TRAIN: Batch {batch:4d} | Speed: {avg_speed:7.2f} img/s | Loss: {loss.item():.4f}")
            
    return sum(batch_train_losses) / len(batch_train_losses)