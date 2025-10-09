"""Compute PR AUC for existing trained probes."""

import os
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from train_probes import LinearProbe, prepare_dataset_with_lag


def main():
    """Compute PR AUC for all existing probes."""
    # Configuration
    DATA_DIR = "/workspace/probe_training/data"
    MODELS_DIR = "/workspace/probe_training/models"
    RESULTS_DIR = "/workspace/probe_training/results"

    LAYERS = [8, 10, 12, 14, 16]
    LAGS = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48]
    CATEGORY = "backtracking"
    HIDDEN_DIM = 4096

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load labels once
    print("Loading labels...")
    with open(os.path.join(DATA_DIR, "val_labels.pkl"), 'rb') as f:
        val_data = pickle.load(f)
        val_labels = val_data['labels']

    print(f"Val samples: {len(val_labels)}")

    # Compute PR AUC for each probe
    results = []

    total_configs = len(LAYERS) * len(LAGS)
    with tqdm(total=total_configs, desc="Computing PR AUC") as pbar:
        for layer in LAYERS:
            # Load this layer's activations
            print(f"\nLoading layer {layer} activations...")
            with open(os.path.join(DATA_DIR, f"val_layer{layer}_activations.pkl"), 'rb') as f:
                val_activations = pickle.load(f)

            for lag in LAGS:
                config_name = f"layer{layer}_lag+{lag}"
                probe_path = os.path.join(MODELS_DIR, f"probe_{config_name}.pt")

                pbar.set_description(f"Evaluating {config_name}")

                # Check if probe exists
                if not os.path.exists(probe_path):
                    print(f"Skipping {config_name}: probe not found")
                    pbar.update(1)
                    continue

                # Prepare validation dataset
                X_val, y_val = prepare_dataset_with_lag(
                    val_activations, val_labels, layer, CATEGORY, lag, single_layer=True
                )

                if X_val is None or y_val is None:
                    print(f"Skipping {config_name}: no data")
                    pbar.update(1)
                    continue

                # Load probe
                probe = LinearProbe(HIDDEN_DIM)
                probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
                probe = probe.to(device)
                probe.eval()

                # Compute predictions
                with torch.no_grad():
                    val_logits = probe(X_val.to(device))
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()

                # Compute PR AUC
                try:
                    pr_auc = average_precision_score(y_val.numpy(), val_probs)
                except:
                    pr_auc = 0.0

                # Record results
                results.append({
                    'layer': layer,
                    'lag': lag,
                    'pr_auc': pr_auc,
                })

                pbar.update(1)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, "pr_auc_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nPR AUC results saved to {results_path}")

    # Print summary
    print("\n=== Top 10 Probes by PR AUC ===")
    print(results_df.nlargest(10, 'pr_auc'))

    print("\n=== PR AUC by Layer (lag=0) ===")
    lag0 = results_df[results_df['lag'] == 0].sort_values('pr_auc', ascending=False)
    print(lag0)


if __name__ == "__main__":
    main()
