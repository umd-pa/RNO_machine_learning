import wandb
import torch
import time


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device | str,
) -> float:
    """
    Performs one full test epoch over the entire batched dataset.

    Uses torch.inference_mode() instead of no_grad() — stricter and faster
    as it disables both gradient tracking and version tracking. Combined with
    bfloat16 autocast for consistent precision with the training step.

    Loss tensors are accumulated on the GPU and synced to CPU only once at
    the end of the epoch, matching the same pattern as train_step.

    WandB logging is conditional on an active run (wandb.run is not None)
    — no flag needs to be passed. If WandB is disabled, all logging is
    silently skipped.

    Args:
        model      : PyTorch model to evaluate. Must already be on `device`.
        data_loader: DataLoader providing (images, labels) batches.
        loss_fn    : Loss function (e.g., MSELoss).
        device     : Device to run computations on (CPU/CUDA).

    Returns:
        float: Mean test loss across all batches in the epoch.
    """
    model.eval()

    batch_test_losses = []
    total_imgs        = 0
    start_time        = time.time()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):

            # Async transfer — CPU does not wait for GPU transfer to complete
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # bfloat16 forward pass — consistent precision with train_step
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_pred = model(X)
                loss   = loss_fn(y_pred, y.squeeze(1))

            total_imgs += X.size(0)

            # Stay on GPU — no CPU sync until end of epoch
            batch_test_losses.append(loss.detach())

            if batch > 0 and batch % 10 == 0:
                avg_speed = total_imgs / (time.time() - start_time)
                print(f"TEST: Batch {batch:4d} | Speed: {avg_speed:7.2f} img/s")

    # Single CPU sync — stack all GPU loss tensors, mean, then .item()
    return torch.stack(batch_test_losses).mean().item()