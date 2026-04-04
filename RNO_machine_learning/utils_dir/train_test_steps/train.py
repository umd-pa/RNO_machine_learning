import wandb
import torch
import time


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """
    Performs one full training epoch over the entire batched dataset.

    Uses bfloat16 mixed precision autocast for faster computation. No
    GradScaler is needed — bfloat16 shares float32's exponent range and
    cannot underflow, unlike float16 which requires gradient scaling.

    Loss tensors are accumulated on the GPU throughout the epoch and
    synced to CPU only once at the end, avoiding per-batch CPU/GPU
    synchronization stalls that would otherwise bottleneck the GPU.

    Speed is reported as a cumulative average since epoch start rather
    than a per-batch instantaneous measurement — early batches appear
    slower due to GPU warmup and cold data cache, and the average
    stabilizes as the epoch progresses.

    WandB logging is conditional on an active run (wandb.run is not None)
    — no flag needs to be passed. If WandB is disabled in the config,
    wandb.run is None and all logging calls are silently skipped.

    Args:
        model      : PyTorch model to train. Must already be on `device`.
        data_loader: DataLoader providing (images, labels) batches.
        loss_fn    : Loss function (e.g., MSELoss).
        optimizer  : Optimizer (e.g., Adam).
        device     : Device to run computations on (CPU/CUDA).

    Returns:
        float: Mean training loss across all batches in the epoch.
    """
    model.train()

    batch_train_losses = []  # GPU tensors accumulated without CPU sync
    total_imgs         = 0
    start_time         = time.time()

    for batch, (X, y) in enumerate(data_loader):

        # Async transfer — CPU does not wait for GPU transfer to complete
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Frees gradient memory entirely rather than filling with zeros
        optimizer.zero_grad(set_to_none=True)

        # bfloat16 forward pass — no GradScaler needed
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            y_pred = model(X)
            loss   = loss_fn(y_pred, y.squeeze(1))

        loss.backward()
        optimizer.step()

        total_imgs += X.size(0)

        # detach() removes from computation graph without CPU sync —
        # tensor stays on GPU until stacked at end of epoch
        batch_train_losses.append(loss.detach())

        if batch % 100 == 0:
            avg_speed = total_imgs / (time.time() - start_time)
            print(f"TRAIN: Batch {batch:4d} | Speed: {avg_speed:7.2f} img/s | Loss: {loss.detach():.4f}")

    # Single CPU sync — stack all GPU loss tensors, mean, then .item()
    avg_loss = torch.stack(batch_train_losses).mean().item()
    return avg_loss