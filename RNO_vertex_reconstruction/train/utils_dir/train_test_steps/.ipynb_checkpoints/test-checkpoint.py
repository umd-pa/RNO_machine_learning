import torch

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device):
    """
    Performs one testing epoch over the entire batched dataset.
    
    Args:
        model: PyTorch neural network model to train
        dataloader: DataLoader containing training batches
        optimizer: Optimization algorithm (e.g., Adam, SGD)
        device: Device to run computations on (CPU/CUDA)
    
    Returns:
        float: Average training loss across all batches in the epoch
    """
    # Set model to evaluation mode - disables dropout, batch normalization training behavior
    model.eval()

    # Initialize list to store loss from each batch
    batch_test_losses = []

    # Stop tracking gradients and loop through all batches
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Move input features and labels to the specified device
            X, y = X.to(device), y.to(device)
    
            # Forward pass
            y_pred = model(X)
            
            # Calculate loss
            loss = loss_fn(y_pred, y)
            batch_test_losses.append(loss.item())

    # Return the average loss across all batches in this testing epoch
    return sum(batch_test_losses) / len(batch_test_losses)
