import torch

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim, device: torch.device):
    """
    Performs one training epoch over the entire batched dataset.
    
    Args:
        model: PyTorch neural network model to train
        dataloader: DataLoader containing training batches
        loss_fn: Loss function (e.g., CrossEntropyLoss, MSELoss)
        optimizer: Optimization algorithm (e.g., Adam, SGD)
        device: Device to run computations on (CPU/CUDA)
    
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
        
        # Calculate loss
        loss = loss_fn(y_pred, y)
        batch_train_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return the average loss across all batches in this training epoch
    return sum(batch_train_losses) / len(batch_train_losses)
