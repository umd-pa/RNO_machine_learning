import logging
import torch

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device | str, logger: logging.Logger):
    """
    Performs one training epoch over the entire batched dataset.
    
    Args:
        model: PyTorch neural network model to train
        dataloader: DataLoader containing training batches
        loss_fn: Loss function (e.g., CrossEntropyLoss, MSELoss)
        optimizer: Optimization algorithm (e.g., Adam, SGD)
        device: Device to run computations on (CPU/CUDA)
        logging: Logger for logging training progress
    
    Returns:
        float: Average training loss across all batches in the epoch
    """
    # Set model to training mode - enables dropout, batch normalization training behavior
    model.train()

    # Initialize list to store loss from each batch
    batch_train_losses = []

    # Loop through all batches
    for batch, (X, y) in enumerate(data_loader):
        # Move input features and labels to the specified device
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)

        # Squeeze labels. Only matters if batch size = 1. Will convert y from y.shape = [1,3] to [3]
        y = y.squeeze()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Batch {batch}:')
            logger.debug([(yp, y0) for yp, y0 in zip(y_pred[0], y[0])])

        # Calculate loss
        loss = loss_fn(y_pred, y)
        batch_train_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        # for name, param in model.named_parameters(): For debugging
        #     if param.grad is not None:
        #         print(f"{name} - grad norm: {param.grad.norm().item()}")
        #     else:
        #         print(f"{name} - NO GRADIENT")

    # Return the average loss across all batches in this training epoch
    return sum(batch_train_losses) / len(batch_train_losses)
