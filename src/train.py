import argparse
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
#from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import create_model
from generator import generate_regular_sequence, generate_irregular_sequence


# Reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)                   # Python random
    np.random.seed(seed)                # NumPy random
    torch.manual_seed(seed)             # PyTorch CPU
    torch.cuda.manual_seed_all(seed)    # PyTorch GPU

# Dataset
class SequenceDataset(Dataset):
    """
    Generates synthetic sequences on-the-fly.

    label 0 -> regular sequence
    label 1 -> irregular sequence

    Supports:
    - fixed-length sequences (all same length)
    - variable-length sequences (random length per sample)
    """

    def __init__(
        self,
        size: int = 30000,
        seq_length: int = 1000,
        #min_seq_length: int | None = None,
        #max_seq_length: int | None = None,
    ):
        self.size = size    # Total number of samples per epoch
        self.seq_length = seq_length

    def __len__(self) -> int: 
        """Returns dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates ONE sample. Saves memory.

        Returns:
            x: sequence (L,)
            y: label (scalar)
        """

        # Balanced dataset: alternating labels
        if idx % 2 == 0:
            seq = generate_regular_sequence(self.seq_length)
            label = 0.0
        else:
            seq = generate_irregular_sequence(self.seq_length)
            label = 1.0

        # Convert to PyTorch tensors
        x = torch.tensor(seq, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

# Metrics
def compute_auc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes ROC AUC score. AUC defines how well model separates classes

    Steps:
    - apply sigmoid to logits → probabilities
    - compare probabilities with true labels
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    try:
        return float(roc_auc_score(labels_np, probs))
    except ValueError:
        return float("nan")

# One epoch runner (training or validation)
def run_one_epoch(model, loader, criterion, device, optimizer=None):
    """
    Runs one full pass over dataset.

    If optimizer is provided → training mode
    If optimizer is None     → validation mode
    """
    training = optimizer is not None
    model.train(training)

    all_losses = []
    all_logits = []
    all_labels = []

    # Progress bar
    pbar = tqdm(
        loader,
        desc="Train" if training else "Val",
        leave=False,
        dynamic_ncols=True,
    )

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        # Reset gradients only in training mode
        if training:
            optimizer.zero_grad()   # R eset gradients

        # Forward pass
        logits = model(x).squeeze(1)    # (B, 1) -> (B,)
        loss = criterion(logits, y)     # Compute loss  

        # Backward pass and optimization step only in training mode
        if training:
            loss.backward()     # Compute gradients
            optimizer.step()    # Update weights

        # Store results for metrics
        all_losses.append(loss.item())
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

        # Update progress bar with average loss so far
        avg_loss_so_far = float(np.mean(all_losses))
        pbar.set_postfix(loss=f"{avg_loss_so_far:.4f}")

    # Concatenate all batches to compute epoch-level metrics
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Compute average loss and AUC for the epoch
    avg_loss = float(np.mean(all_losses))
    auc = compute_auc_from_logits(all_logits, all_labels)

    return avg_loss, auc

# Plotting
def save_training_plots(history, out_prefix: str = "training"):
    """
    Saves training curves:
    - loss vs epoch
    - AUC vs epoch
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss.png", dpi=150)
    plt.close()

    # AUC plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_auc"], label="train AUC")
    plt.plot(epochs, history["val_auc"], label="val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training / Validation AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_auc.png", dpi=150)
    plt.close()

# Training pipeline
def train(
    num_epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    train_size: int = 30000,
    val_size: int = 6000,
    seq_length: int = 1000,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 4,
    checkpoint_path: str = "model.pth",
    plot_prefix: str = "training",
    patience: int = 5,
):
    """
    Full training pipeline:
    - prepare data
    - initialize model
    - train for multiple epochs
    - track metrics
    - save best model
    """

    set_seed(seed)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training mode: fixed length (seq_length={seq_length})")

    # Create datasets train and validation
    train_dataset = SequenceDataset(
        size=train_size,
        seq_length=seq_length,
    )
    val_dataset = SequenceDataset(
        size=val_size,
        seq_length=seq_length,
    )

    # DataLoaders with padding collate function for variable-length sequences
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Create model
    model = create_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    # Loss function for binary classification. Combines sigmoid + binary cross entropy
    criterion = nn.BCEWithLogitsLoss()

    # Tracking of best model and metrics
    best_val_auc = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
    }

    # Epoch loop
    for epoch in range(1, num_epochs + 1):
        # Training phase
        train_loss, train_auc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        # Validation phase (no gradients needed)
        with torch.no_grad():
            val_loss, val_auc = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
            )

        # Store metrics for plotting
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # Check for improvement and save best model
        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | train AUC: {train_auc:.4f} | "
            f"val loss: {val_loss:.4f} | val AUC: {val_auc:.4f} | "
            f"{'saved best' if improved else f'no improvement ({epochs_without_improvement}/{patience})'}"
        )

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break

    save_training_plots(history, out_prefix=plot_prefix)

    print("\nTraining complete.")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Plots saved to: {plot_prefix}_loss.png and {plot_prefix}_auc.png")

# YAML config loading
def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a dictionary of key-value pairs.")

    return config

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN for regular vs irregular pulse sequences with fixed or variable lengths."
    )

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train-size", type=int, default=30000, help="Training samples per epoch")
    parser.add_argument("--val-size", type=int, default=6000, help="Validation samples per epoch")
    parser.add_argument("--seq-length", type=int, default=None, help="Use fixed-length training if provided")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--checkpoint", type=str, default="model.pth", help="Best model checkpoint path")
    parser.add_argument("--plot-prefix", type=str, default="training", help="Prefix for output plot files")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience based on validation AUC")

    args = parser.parse_args()

    # If YAML config is provided, override parser defaults / CLI values
    if args.config is not None:
        yaml_config = load_yaml_config(args.config)
        for key, value in yaml_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                raise ValueError(f"Unknown config key in YAML: {key}")

    print("Final config:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_size=args.train_size,
        val_size=args.val_size,
        seq_length=args.seq_length,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint,
        plot_prefix=args.plot_prefix,
        patience=args.patience,
    )
