"""Colorize transcript using probe predictions for backtracking."""

import os
import torch
import pickle
import numpy as np
from pathlib import Path
from nnsight import LanguageModel
from transformers import AutoTokenizer
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_probes import LinearProbe
from utils.data_processing import load_annotated_chains, process_chain


def load_probe(probe_path, input_dim=4096):
    """Load a trained probe from disk."""
    probe = LinearProbe(input_dim)
    probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
    probe.eval()
    return probe


def extract_single_layer_activations(text, model, tokenizer, layer_idx):
    """
    Extract activations from a single layer for a given text.

    Returns:
        activations: Tensor of shape [seq_len, hidden_dim]
        tokens: List of token IDs
    """
    with torch.inference_mode():
        with model.trace(text) as tracer:
            # Extract activation from specified layer
            saved_act = model.model.layers[layer_idx].output[0].save()

        # Get the activation tensor [seq_len, hidden_dim]
        activations = saved_act.cpu()

        # Get tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)

    return activations, tokens


def colorize_html(text, tokens, predictions, tokenizer, backtrack_indices=None):
    """
    Create HTML with text colored by prediction values.

    Args:
        text: Original text string
        tokens: List of token IDs
        predictions: Array of prediction values [seq_len]
        tokenizer: HuggingFace tokenizer
        backtrack_indices: Set of token indices that are annotated as backtracking

    Returns:
        HTML string
    """
    # Normalize predictions to 0-1 range for coloring
    pred_min = predictions.min()
    pred_max = predictions.max()
    if pred_max > pred_min:
        normalized_preds = (predictions - pred_min) / (pred_max - pred_min)
    else:
        normalized_preds = np.zeros_like(predictions)

    html_parts = []
    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                font-family: 'Courier New', monospace;
                padding: 40px;
                background-color: #1e1e1e;
                color: #ffffff;
                line-height: 1.8;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background-color: #2d2d2d;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            h1 {
                color: #61dafb;
                border-bottom: 2px solid #61dafb;
                padding-bottom: 10px;
            }
            .legend {
                margin: 20px 0;
                padding: 15px;
                background-color: #3d3d3d;
                border-radius: 5px;
            }
            .token {
                display: inline;
                padding: 2px 0;
                transition: background-color 0.2s;
                color: #000000;
            }
            .token:hover {
                outline: 1px solid #61dafb;
            }
            .backtrack-annotated {
                border: 2px solid #00ff00;
                padding: 1px 2px;
                border-radius: 3px;
                box-shadow: 0 0 5px rgba(0, 255, 0, 0.5);
            }
            .gradient-bar {
                height: 20px;
                background: linear-gradient(to right,
                    rgb(255, 255, 255),
                    rgb(255, 200, 200),
                    rgb(255, 150, 150),
                    rgb(255, 100, 100),
                    rgb(255, 50, 50),
                    rgb(255, 0, 0));
                margin: 10px 0;
                border-radius: 3px;
            }
            .gradient-labels {
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: #aaa;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Transcript Colorized by Backtracking Probe (Lag 0)</h1>
            <div class="legend">
                <strong>Color Scale:</strong> White → Red indicates increasing likelihood of backtracking at current position
                <div class="gradient-bar"></div>
                <div class="gradient-labels">
                    <span>Low prediction (unlikely backtrack)</span>
                    <span>High prediction (likely backtrack)</span>
                </div>
                <div style="margin-top: 15px;">
                    <strong>Ground Truth:</strong> <span class="backtrack-annotated" style="display: inline-block; padding: 3px 8px;">Green box</span> = Annotated backtracking token
                </div>
            </div>
            <div class="content">
    """)

    # Decode each token and color it
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        # Escape HTML special characters
        token_text = token_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Get prediction value for this token
        pred_value = normalized_preds[i]

        # Create color: white (255,255,255) to red (255,0,0)
        r = 255
        g = int(255 * (1 - pred_value))
        b = int(255 * (1 - pred_value))
        color = f"rgb({r}, {g}, {b})"

        # Check if this is an annotated backtracking token
        is_backtrack = backtrack_indices is not None and i in backtrack_indices
        backtrack_class = ' backtrack-annotated' if is_backtrack else ''
        backtrack_label = ' [BACKTRACK]' if is_backtrack else ''

        # Add tooltip with actual prediction value
        html_parts.append(
            f'<span class="token{backtrack_class}" style="background-color: {color};" '
            f'title="Token {i}: {predictions[i]:.4f}{backtrack_label}">{token_text}</span>'
        )

    html_parts.append("""
            </div>
        </div>
    </body>
    </html>
    """)

    return ''.join(html_parts)


def main():
    """Main pipeline to colorize a transcript."""
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    LAYER = 12  # Best layer for lag=0
    LAG = 0
    PROBE_PATH = f"/workspace/probe_training/models/probe_layer{LAYER}_lag+{LAG}.pt"
    DATA_FILE = "/workspace/all_annotated_chains.json"
    OUTPUT_DIR = "/workspace/probe_training/visualizations"
    HIDDEN_DIM = 4096

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load HuggingFace token
    env_file = "/workspace/.env"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading probe from {PROBE_PATH}...")
    probe = load_probe(PROBE_PATH, input_dim=HIDDEN_DIM)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probe = probe.to(device)
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    model = LanguageModel(MODEL_NAME, device_map=device, torch_dtype=torch.bfloat16, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

    print("Loading annotated chains...")
    chains = load_annotated_chains(DATA_FILE)

    # Filter chains: only validation set with backtracking annotations
    print(f"Filtering chains with backtracking annotations...")
    chains_to_process = []
    for chain in chains:
        if 'backtracking' in chain.get('annotated_chain', ''):
            chains_to_process.append(chain)

    print(f"Found {len(chains_to_process)} chains with backtracking annotations")
    print(f"Generating visualizations for first 10 chains...")

    chains_to_process = chains_to_process[:10]

    for chain in chains_to_process:
        task_id = chain.get('task_id', 'unknown')

        print(f"\nProcessing {task_id}...")

        # Process chain to get formatted text and tokens
        tokens, annotation_indices = process_chain(tokenizer, chain)
        text = tokenizer.decode(tokens)

        # Debug: print available annotation categories
        print(f"  Annotation categories: {list(annotation_indices.keys())}")

        # Get backtracking annotations - convert (start, end) ranges to individual indices
        backtrack_indices = set()
        if 'backtracking' in annotation_indices:
            print(f"  Backtracking ranges: {len(annotation_indices['backtracking'])} ranges")
            for start_idx, end_idx in annotation_indices['backtracking']:
                # Include all tokens from start to end (inclusive)
                for idx in range(start_idx, end_idx + 1):
                    backtrack_indices.add(idx)

        print(f"  Found {len(backtrack_indices)} annotated backtracking tokens")

        # Extract activations for this layer
        print(f"  Extracting layer {LAYER} activations...")
        activations, _ = extract_single_layer_activations(text, model, tokenizer, LAYER)

        # Run probe predictions
        print(f"  Running probe predictions...")
        with torch.no_grad():
            activations_device = activations.float().to(device)
            logits = probe(activations_device)
            predictions = torch.sigmoid(logits).cpu().numpy()

        # Generate HTML
        print(f"  Generating HTML visualization...")
        html = colorize_html(text, tokens.tolist(), predictions, tokenizer, backtrack_indices)

        # Save HTML
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}_colorized.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"  Saved to {output_path}")
        print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  Mean prediction: {predictions.mean():.4f}")

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
