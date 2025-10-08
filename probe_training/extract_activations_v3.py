"""Extract activations efficiently: one forward pass per chain, save per-layer."""

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


def extract_all_layers_per_chain(
    chains,
    model,
    tokenizer,
    layers,
    output_dir="data",
    split_name="train"
):
    """
    Extract activations for ALL layers with a SINGLE forward pass per chain.
    Save per-layer for memory-efficient training later.
    """
    # Initialize storage for each layer
    layer_activations = {layer: [] for layer in layers}
    all_labels = []
    all_metadata = []

    print(f"Processing {len(chains)} chains (1 forward pass each)...")

    with torch.inference_mode():
        for chain_idx, chain in enumerate(tqdm(chains)):
            try:
                # Process chain
                tokens, annotation_indices = process_chain(tokenizer, chain)

                # ONE forward pass - gets activations for ALL layers
                logits, cache = model.run_with_cache(tokens.unsqueeze(0))

                # Extract all layers from this single forward pass
                for layer in layers:
                    # Get activation for this layer: [batch, seq, hidden]
                    acts = cache[f'blocks.{layer}.hook_resid_post'][0]  # [seq, hidden]
                    # Keep as bfloat16, move to CPU
                    layer_activations[layer].append(acts.cpu())

                # Create labels (same for all layers)
                seq_length = len(tokens)
                labels = create_binary_labels(seq_length, annotation_indices)
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

    # Save each layer separately
    print(f"\nSaving per-layer files...")
    for layer in layers:
        layer_file = os.path.join(output_dir, f"{split_name}_layer{layer}_activations.pkl")
        print(f"  Saving layer {layer} to {layer_file}...")
        with open(layer_file, 'wb') as f:
            pickle.dump(layer_activations[layer], f)

    # Save labels once (shared across all layers)
    labels_file = os.path.join(output_dir, f"{split_name}_labels.pkl")
    print(f"  Saving labels to {labels_file}...")
    with open(labels_file, 'wb') as f:
        pickle.dump({'labels': all_labels, 'metadata': all_metadata}, f)

    print(f"✅ {split_name} complete!")


def main():
    """Main extraction pipeline."""
    # Set CUDA memory allocator config to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    DATA_FILE = "/workspace/all_annotated_chains.json"
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    LAYERS = [8, 10, 12, 14, 16]
    OUTPUT_DIR = "/workspace/probe_training/data"

    # Load HF token from .env
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

    # Load model (will pick up HF_TOKEN from env)
    print(f"Loading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device='cuda',
        dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Extract activations (1 forward pass per chain, save all layers)
    print(f"\n=== Extracting train activations ===")
    extract_all_layers_per_chain(
        train_chains,
        model,
        tokenizer,
        LAYERS,
        output_dir=OUTPUT_DIR,
        split_name="train"
    )

    print(f"\n=== Extracting val activations ===")
    extract_all_layers_per_chain(
        val_chains,
        model,
        tokenizer,
        LAYERS,
        output_dir=OUTPUT_DIR,
        split_name="val"
    )

    print("\n✅ All done! Extracted with 1 forward pass per chain.")


if __name__ == "__main__":
    main()
