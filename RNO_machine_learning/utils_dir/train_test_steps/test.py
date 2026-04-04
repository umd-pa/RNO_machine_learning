import torch
import time


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device | str,
) -> tuple[float, float]:
    """
    Performs one full test epoch over the entire batched dataset.

    Uses torch.inference_mode() instead of no_grad() — stricter and faster
    as it disables both gradient tracking and version tracking. Combined with
    bfloat16 autocast for consistent precision with the training step.

    Loss and Euclidean error tensors are accumulated on the GPU and synced
    to CPU only once at the end of the epoch, matching the same pattern as
    train_step. Euclidean error is computed in physical space (meters) by
    unnormalizing predictions and targets using the model's registered
    label_mean and label_std buffers.

    Args:
        model      : PyTorch model to evaluate. Must already be on `device`.
                     Must have label_mean and label_std registered as buffers.
        data_loader: DataLoader providing (images, labels) batches.
        loss_fn    : Loss function (e.g., HuberLoss).
        device     : Device to run computations on (CPU/CUDA).

    Returns:
        tuple[float, float]: (mean_test_loss, mean_euclidean_error_meters)
    """
    model.eval()

    batch_test_losses  = []
    batch_euclidean    = []
    total_imgs         = 0
    start_time         = time.time()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):

            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_pred = model(X)
                loss   = loss_fn(y_pred, y.squeeze(1))

            # Unnormalize to physical space for Euclidean error
            y_squeezed   = y.squeeze(1)
            pred_m       = (y_pred     * model.label_std) + model.label_mean
            target_m     = (y_squeezed * model.label_std) + model.label_mean
            euclidean    = torch.sqrt(((pred_m - target_m) ** 2).sum(dim=-1)).mean()

            total_imgs += X.size(0)

            batch_test_losses.append(loss.detach())
            batch_euclidean.append(euclidean.detach())

            if batch > 0 and batch % 10 == 0:
                avg_speed = total_imgs / (time.time() - start_time)
                print(f"TEST: Batch {batch:4d} | Speed: {avg_speed:7.2f} img/s")

    mean_loss      = torch.stack(batch_test_losses).mean().item()
    mean_euclidean = torch.stack(batch_euclidean).mean().item()

    return mean_loss, mean_euclidean