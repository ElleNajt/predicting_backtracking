"""Extract activations using TransformerLens, saving per-layer to reduce memory."""

import os
import torch
import pickle
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from utils.data_processing import (
    load_annotated_chains,
    process_chain,
    create_binary_labels,
    train_val_split
)


def extract_activations_by_layer(
    chains,
    model,
    tokenizer,
    layer,
    output_dir="data",
    split_name="train"
):
    """
    Extract activations for a single layer across all chains.
    Saves immediately to reduce memory usage.
    """
    all_activations = []
    all_labels = []
    all_metadata = []

    print(f"Processing {len(chains)} chains for layer {layer}...")

    with torch.inference_mode():
        for chain_idx, chain in enumerate(tqdm(chains)):
            try:
                # Process chain to get tokens and annotation indices
                tokens, annotation_indices = process_chain(tokenizer, chain)

                # Run model and get activations for this layer only
                logits, cache = model.run_with_cache(tokens.unsqueeze(0))

                # Extract activation from this layer: cache has shape [batch, seq, hidden]
                acts = cache[f'blocks.{layer}.hook_resid_post'][0]  # [seq, hidden]

                # Keep as bfloat16, move to CPU
                acts = acts.cpu()

                # Create binary labels
                seq_length = len(tokens)
                labels = create_binary_labels(seq_length, annotation_indices)

                # Store
                all_activations.append(acts)
                all_labels.append(labels)
                all_metadata.append({
                    'chain_id': chain.get('task_id', f'chain_{chain_idx}'),
                    'seq_length': seq_length,
                })

                # Clear GPU cache
                del logits, cache
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing chain {chain_idx}: {e}")
                continue

    # Save this layer's data
    layer_file = os.path.join(output_dir, f"{split_name}_layer{layer}_activations.pkl")
    print(f"Saving layer {layer} activations to {layer_file}...")
    with open(layer_file, 'wb') as f:
        pickle.dump(all_activations, f)

    # Save labels once (same for all layers)
    if layer == 8:  # Only save labels once
        labels_file = os.path.join(output_dir, f"{split_name}_labels.pkl")
        print(f"Saving labels to {labels_file}...")
        with open(labels_file, 'wb') as f:
            pickle.dump({'labels': all_labels, 'metadata': all_metadata}, f)

    print(f"Layer {layer} done!")


def main():
    """Main extraction pipeline."""
    DATA_FILE = "/workspace/all_annotated_chains.json"
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    LAYERS = [8, 10, 12, 14, 16]
    OUTPUT_DIR = "/workspace/probe_training/data"

    # Load HF token
    env_file = "/workspace/.env"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    # Load data
    print("Loading annotated chains...")
    chains = load_annotated_chains(DATA_FILE)
    print(f"Loaded {len(chains)} chains")

    # Split
    print("Splitting into train/val...")
    train_chains, val_chains = train_val_split(chains, val_ratio=0.2, random_seed=42)
    print(f"Train: {len(train_chains)}, Val: {len(val_chains)}")

    # Load model with TransformerLens (it will pick up HF_TOKEN from env)
    print(f"Loading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device='cuda',
        dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Extract activations layer by layer
    for layer in LAYERS:
        print(f"\n=== Extracting layer {layer} (train) ===")
        extract_activations_by_layer(
            train_chains,
            model,
            tokenizer,
            layer,
            output_dir=OUTPUT_DIR,
            split_name="train"
        )

        print(f"\n=== Extracting layer {layer} (val) ===")
        extract_activations_by_layer(
            val_chains,
            model,
            tokenizer,
            layer,
            output_dir=OUTPUT_DIR,
            split_name="val"
        )

    print("\n✅ All activations extracted successfully!")


if __name__ == "__main__":
    main()
