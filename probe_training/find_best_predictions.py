"""Find examples where the probe performs particularly well."""

import os
import pickle
import torch
import numpy as np
from train_probes import LinearProbe, prepare_dataset_with_lag
from utils.data_processing import load_annotated_chains, process_chain
from transformers import AutoTokenizer


def find_best_predictions(layer=12, lag=4, top_k=5):
    """
    Find chains where the probe has the highest precision for predicting backtracking.

    Returns chains where:
    1. There are actual backtracking events
    2. The probe makes strong predictions (>0.8) before those events
    3. The probe has low false positive rate
    """
    # Configuration
    DATA_DIR = "/workspace/probe_training/data"
    MODELS_DIR = "/workspace/probe_training/models"
    DATA_FILE = "/workspace/all_annotated_chains.json"
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    HIDDEN_DIM = 4096

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load probe
    probe_path = os.path.join(MODELS_DIR, f"probe_layer{layer}_lag+{lag}.pt")
    probe = LinearProbe(HIDDEN_DIM)
    probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
    probe = probe.to(device)
    probe.eval()
    print(f"Loaded probe: layer {layer}, lag +{lag}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load validation data
    with open(os.path.join(DATA_DIR, f"val_layer{layer}_activations.pkl"), 'rb') as f:
        val_activations = pickle.load(f)

    with open(os.path.join(DATA_DIR, "val_labels.pkl"), 'rb') as f:
        val_data = pickle.load(f)
        val_labels = val_data['labels']

    # Load annotated chains to get metadata
    all_chains = load_annotated_chains(DATA_FILE)
    val_chains = all_chains[800:]  # Validation split

    print(f"Analyzing {len(val_chains)} validation chains...")

    # Evaluate each chain
    chain_scores = []

    for idx, (activations, labels, chain) in enumerate(zip(val_activations, val_labels, val_chains)):
        # Prepare dataset for this chain
        X, y = prepare_dataset_with_lag(
            [activations], [labels], layer, "backtracking", lag, single_layer=True
        )

        if X is None or y is None or len(y) == 0:
            continue

        # Get predictions
        with torch.no_grad():
            logits = probe(X.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()

        y_np = y.numpy()

        # Calculate metrics for this chain
        n_positive = y_np.sum()
        n_total = len(y_np)

        if n_positive == 0:
            continue

        # Find strong predictions (>0.8)
        strong_preds = probs > 0.8

        # True positives with strong predictions
        strong_tp = ((strong_preds) & (y_np == 1)).sum()

        # False positives with strong predictions
        strong_fp = ((strong_preds) & (y_np == 0)).sum()

        # Precision for strong predictions
        if strong_preds.sum() > 0:
            precision = strong_tp / strong_preds.sum()
        else:
            precision = 0.0

        # Recall
        recall = strong_tp / n_positive if n_positive > 0 else 0.0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Score: prioritize chains with good precision AND some strong predictions
        score = f1 * strong_preds.sum()  # Weight by number of strong predictions

        chain_scores.append({
            'idx': idx,
            'task_id': chain.get('task_id', f'chain_{idx}'),
            'n_tokens': n_total,
            'n_backtrack': n_positive,
            'backtrack_rate': n_positive / n_total,
            'n_strong_preds': strong_preds.sum(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'score': score,
            'mean_prob': probs.mean(),
            'max_prob': probs.max(),
        })

    # Sort by score
    chain_scores.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n=== Top {top_k} Chains for Lag +{lag} Predictions ===")
    print(f"{'Rank':<5} {'Task ID':<15} {'Tokens':<8} {'Backtrack':<10} {'Strong Preds':<13} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Score':<8}")
    print("=" * 110)

    for i, chain in enumerate(chain_scores[:top_k]):
        print(f"{i+1:<5} {chain['task_id']:<15} {chain['n_tokens']:<8} "
              f"{chain['n_backtrack']:<10} {chain['n_strong_preds']:<13} "
              f"{chain['precision']:<10.3f} {chain['recall']:<8.3f} "
              f"{chain['f1']:<8.3f} {chain['score']:<8.2f}")

    print(f"\nBest chain: {chain_scores[0]['task_id']}")
    print(f"  - {chain_scores[0]['n_tokens']} tokens total")
    print(f"  - {chain_scores[0]['n_backtrack']} backtracking events")
    print(f"  - {chain_scores[0]['n_strong_preds']} strong predictions (>0.8)")
    print(f"  - Precision: {chain_scores[0]['precision']:.3f}")
    print(f"  - Recall: {chain_scores[0]['recall']:.3f}")

    return [c['task_id'] for c in chain_scores[:top_k]]


if __name__ == "__main__":
    best_chains = find_best_predictions(layer=12, lag=4, top_k=10)
    print(f"\nBest chains: {best_chains}")
