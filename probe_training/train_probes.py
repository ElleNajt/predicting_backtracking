"""Train linear probes on activations to predict labels at different lags."""

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd


class LinearProbe(nn.Module):
    """Simple linear probe for binary classification."""

    def __init__(self, input_dim, l2_penalty=0.01):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.l2_penalty = l2_penalty

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def get_l2_loss(self):
        """Compute L2 regularization loss."""
        return self.l2_penalty * torch.sum(self.linear.weight ** 2)


def prepare_dataset_with_lag(activations_list, labels_list, layer, category, lag):
    """
    Prepare dataset for training with specified lag.

    Args:
        activations_list: List of dicts {layer: tensor of shape [seq_len, hidden_dim]}
        labels_list: List of dicts {category: array of shape [seq_len]}
        layer: Which layer to use
        category: Which category to predict
        lag: Time lag (negative = predict future, positive = predict past)
            lag = -4 means: activation[t] predicts label[t-4]

    Returns:
        X (features), y (labels)
    """
    X_list = []
    y_list = []

    for activations, labels in zip(activations_list, labels_list):
        if layer not in activations:
            continue
        if category not in labels:
            continue

        acts = activations[layer]  # [seq_len, hidden_dim]
        labs = labels[category]    # [seq_len]

        # Use minimum length to avoid index errors
        seq_len = min(len(acts), len(labs))

        for t in range(seq_len):
            target_idx = t + lag

            # Skip if target is out of bounds
            if target_idx < 0 or target_idx >= seq_len:
                continue

            X_list.append(acts[t])
            y_list.append(labs[target_idx])

    if len(X_list) == 0:
        return None, None

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)

    return X, y


def train_probe(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim=4096,
    l2_penalty=0.01,
    lr=1e-3,
    batch_size=64,
    epochs=50,
    early_stopping_patience=5,
    device='cuda'
):
    """
    Train a linear probe.

    Returns:
        probe, train_metrics, val_metrics
    """
    probe = LinearProbe(input_dim, l2_penalty).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Handle class imbalance
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pos_weight = torch.tensor([pos_weight.item()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0
    patience_counter = 0
    best_probe_state = None

    for epoch in range(epochs):
        probe.train()

        # Shuffle training data
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch training
        total_loss = 0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size].to(device)
            batch_y = y_train_shuffled[i:i+batch_size].to(device)

            optimizer.zero_grad()
            logits = probe(batch_X)
            loss = criterion(logits, batch_y) + probe.get_l2_loss()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val.to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)

            val_acc = accuracy_score(y_val.numpy(), val_preds)
            val_f1 = f1_score(y_val.numpy(), val_preds, zero_division=0)
            try:
                val_auc = roc_auc_score(y_val.numpy(), val_probs)
            except:
                val_auc = 0.0

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_probe_state = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    # Load best probe
    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        # Training metrics
        train_logits = probe(X_train.to(device))
        train_probs = torch.sigmoid(train_logits).cpu().numpy()
        train_preds = (train_probs > 0.5).astype(int)
        train_acc = accuracy_score(y_train.numpy(), train_preds)
        train_f1 = f1_score(y_train.numpy(), train_preds, zero_division=0)

        # Validation metrics
        val_logits = probe(X_val.to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
        val_acc = accuracy_score(y_val.numpy(), val_preds)
        val_f1 = f1_score(y_val.numpy(), val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(y_val.numpy(), val_probs)
        except:
            val_auc = 0.0

    train_metrics = {'accuracy': train_acc, 'f1': train_f1}
    val_metrics = {'accuracy': val_acc, 'f1': val_f1, 'auc': val_auc}

    return probe, train_metrics, val_metrics


def main():
    """Main training pipeline."""
    # Configuration
    DATA_DIR = "/workspace/probe_training/data"
    MODELS_DIR = "/workspace/probe_training/models"
    RESULTS_DIR = "/workspace/probe_training/results"

    LAYERS = [8, 10, 12, 14, 16]
    LAGS = [-16, -12, -8, -4, 0, 4, 8]
    CATEGORY = "backtracking"  # Primary category of interest
    HIDDEN_DIM = 4096

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("Loading activations and labels...")
    with open(os.path.join(DATA_DIR, "train_activations.pkl"), 'rb') as f:
        train_activations = pickle.load(f)

    with open(os.path.join(DATA_DIR, "train_labels.pkl"), 'rb') as f:
        train_data = pickle.load(f)
        train_labels = train_data['labels']

    with open(os.path.join(DATA_DIR, "val_activations.pkl"), 'rb') as f:
        val_activations = pickle.load(f)

    with open(os.path.join(DATA_DIR, "val_labels.pkl"), 'rb') as f:
        val_data = pickle.load(f)
        val_labels = val_data['labels']

    print(f"Train samples: {len(train_activations)}")
    print(f"Val samples: {len(val_activations)}")

    # Train probes for all combinations
    results = []

    total_configs = len(LAYERS) * len(LAGS)
    with tqdm(total=total_configs, desc="Training probes") as pbar:
        for layer in LAYERS:
            for lag in LAGS:
                config_name = f"layer{layer}_lag{lag:+d}"
                pbar.set_description(f"Training {config_name}")

                # Prepare dataset
                X_train, y_train = prepare_dataset_with_lag(
                    train_activations, train_labels, layer, CATEGORY, lag
                )
                X_val, y_val = prepare_dataset_with_lag(
                    val_activations, val_labels, layer, CATEGORY, lag
                )

                if X_train is None or X_val is None:
                    print(f"Skipping {config_name}: no data")
                    pbar.update(1)
                    continue

                # Check class balance
                pos_ratio_train = y_train.mean().item()
                pos_ratio_val = y_val.mean().item()

                if pos_ratio_train == 0 or pos_ratio_train == 1:
                    print(f"Skipping {config_name}: imbalanced training data")
                    pbar.update(1)
                    continue

                # Train probe
                probe, train_metrics, val_metrics = train_probe(
                    X_train, y_train, X_val, y_val,
                    input_dim=HIDDEN_DIM,
                    device=device
                )

                # Save probe
                probe_path = os.path.join(MODELS_DIR, f"probe_{config_name}.pt")
                torch.save(probe.state_dict(), probe_path)

                # Record results
                results.append({
                    'layer': layer,
                    'lag': lag,
                    'train_acc': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_acc': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_auc': val_metrics['auc'],
                    'n_train': len(X_train),
                    'n_val': len(X_val),
                    'pos_ratio_train': pos_ratio_train,
                    'pos_ratio_val': pos_ratio_val,
                })

                pbar.update(1)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, "probe_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n=== Top 10 Probes by Validation F1 ===")
    print(results_df.nlargest(10, 'val_f1')[['layer', 'lag', 'val_f1', 'val_acc', 'val_auc']])


if __name__ == "__main__":
    main()
