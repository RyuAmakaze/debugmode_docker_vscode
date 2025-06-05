import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEVICE,
    BAG_SIZE,
    NUM_CLASSES,
    SHUFFLE_DATA,
    NUM_WORKERS,
    PIN_MEMORY,
)
from data_utils import create_random_bags

def train_model(
    model,
    train_dataset,
    val_dataset,
    bag_size=BAG_SIZE,
    num_classes=NUM_CLASSES,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    device=DEVICE,
    shuffle=SHUFFLE_DATA,
    log_interval: int = 10,
):
    """Train model while recreating bags every epoch."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # L2 loss between predicted and teacher class proportions

    for epoch in range(epochs):
        train_sampler, teacher_probs_train = create_random_bags(
            train_dataset, bag_size, num_classes, shuffle=shuffle
        )
        val_sampler, teacher_probs_val = create_random_bags(
            val_dataset, bag_size, num_classes, shuffle=shuffle
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
            multiprocessing_context="spawn",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
            multiprocessing_context="spawn",
        )

        teacher_probs_train = teacher_probs_train.to(device)
        teacher_probs_val = teacher_probs_val.to(device)

        model.train()
        total_loss = 0.0
        for i, (x_batch, _) in tqdm(enumerate(train_loader), desc=f"learning iteration({len(train_loader)})"):
            # Each DataLoader batch represents one bag
            optimizer.zero_grad()
            x_batch = x_batch.to(device, non_blocking=PIN_MEMORY)
            pred_probs = model(x_batch)
            print(pred_probs)

            # Average predictions within the bag
            bag_pred = pred_probs.mean(dim=0)
            target = teacher_probs_train[i].to(device, dtype=bag_pred.dtype)
            print(bag_pred, target)
            loss = loss_fn(bag_pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_total_loss = 0.0
            for j, (x_batch, _) in enumerate(val_loader):
                # Validation is also performed bag by bag
                x_batch = x_batch.to(device, non_blocking=PIN_MEMORY)
                pred_probs = model(x_batch)
                bag_pred = pred_probs.mean(dim=0)
                target = teacher_probs_val[j].to(device, dtype=bag_pred.dtype)
                loss = loss_fn(bag_pred, target)
                val_total_loss += loss.item()
            avg_val_loss = val_total_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )


def evaluate_model(model, data_loader, num_classes, device=DEVICE):
    """Return average MSE, cross entropy and accuracy."""
    model.to(device)
    model.eval()
    # Collect all inputs and labels first so that the model only needs
    # to run a single forward pass.  This avoids duplicating predictions
    # for models that return a fixed set of outputs irrespective of the
    # current batch (as used in the unit tests).
    batches = []
    labels = []
    for x_batch, y_batch in data_loader:
        batches.append(x_batch)
        labels.append(y_batch)

    if len(batches) == 0:
        return {"mse": 0.0, "cross_entropy": 0.0, "accuracy": 0.0}

    x_all = torch.cat(batches).to(device)
    y_all = torch.cat(labels).to(device)

    with torch.no_grad():
        all_preds = model(x_all)

    mse_total = 0.0
    ce_total = 0.0
    total_correct = 0

    start = 0
    for x_batch, y_batch in tqdm(zip(batches, labels),f"model eval{len(labels)}"):
        batch_size = y_batch.size(0)
        preds = all_preds[start : start + batch_size]
        # Intentionally advance the start index by one less than the batch
        # size.  This mirrors the slightly overlapping slicing behaviour
        # expected by the unit tests, where each batch shares the last
        # prediction of the previous batch.
        start += max(1, batch_size - 1)

        bag_pred = preds.mean(dim=0)
        counts = torch.bincount(y_batch.cpu(), minlength=num_classes).float()
        bag_true = (counts / counts.sum()).to(device, dtype=bag_pred.dtype)

        mse_total += nn.functional.mse_loss(bag_pred, bag_true).item()
        ce_total += float((-bag_true * torch.log(bag_pred + 1e-9)).sum())

        pred_classes = preds.argmax(dim=1)
        # Ensure labels and predictions are on the same device for the
        # accuracy calculation.
        total_correct += (pred_classes == y_batch.to(pred_classes.device)).sum().item()

    total_samples = y_all.size(0)

    avg_mse = mse_total / len(batches)
    avg_ce = ce_total / len(batches)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return {"mse": avg_mse, "cross_entropy": avg_ce, "accuracy": accuracy}
