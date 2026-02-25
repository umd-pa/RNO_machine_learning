import logging
import torch
import time

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device | str, logger: logging.Logger):
    """
    Performs one testing epoch over the entire batched dataset.
    
    Args:
        model: PyTorch neural network model to train
        dataloader: DataLoader containing training batches
        optimizer: Optimization algorithm (e.g., Adam, SGD)
        device: Device to run computations on (CPU/CUDA)
        logging: Logger for logging testing progress
    
    Returns:
        float: Average training loss across all batches in the epoch
    """
    # Set model to evaluation mode - disables dropout, batch normalization training behavior
    model.eval()

    # Initialize list to store loss from each batch
    batch_test_losses = []

    total_imgs = 0
    start_time = time.time()

    # Stop tracking gradients and loop through all batches
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Move input features and labels to the specified device
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                # Forward pass
                y_pred = model(X)
                
                # Squeeze labels. Only matters if batch size = 1. Will convert y from y.shape = [1,3] to [3]
                y = y.squeeze(1)

                # if logger.isEnabledFor(logging.DEBUG):
                #     logger.debug(f'Batch {batch}:')
                #     logger.debug([(yp, y0) for yp, y0 in zip(y_pred[0], y[0])])
                
                # Calculate loss
                loss = loss_fn(y_pred, y)
            
            total_imgs += len(X)

            batch_test_losses.append(loss.item())

            if batch > 0 and batch % 10 == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                avg_speed = total_imgs / elapsed
                logger.info(f"TEST: Batch {batch:4d} | Speed: {avg_speed:7.2f} img/s | Loss: {loss.item():.4f}")

    # Return the average loss across all batches in this testing epoch
    return sum(batch_test_losses) / len(batch_test_losses)
