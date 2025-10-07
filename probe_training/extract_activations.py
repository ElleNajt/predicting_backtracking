"""Extract activations from language model for all annotated chains."""

import os
import torch
import pickle
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoTokenizer
from utils.data_processing import (
    load_annotated_chains,
    process_chain,
    create_binary_labels,
    train_val_split
)


def extract_activations_from_chains(
    chains,
    model,
    tokenizer,
    layers=[8, 10, 12, 14, 16],
    output_file="activations.pkl",
    labels_file="labels.pkl"
):
    """
    Extract activations from model for all chains.

    Args:
        chains: List of chain dictionaries
        model: nnsight LanguageModel
        tokenizer: HuggingFace tokenizer
        layers: List of layer indices to extract from
        output_file: Path to save activations
        labels_file: Path to save labels

    Returns:
        None (saves to disk)
    """
    all_activations = []
    all_labels = []
    all_metadata = []

    print(f"Processing {len(chains)} chains...")

    with torch.inference_mode():
        for chain_idx, chain in enumerate(tqdm(chains)):
            try:
                # Process chain to get tokens and annotation indices
                tokens, annotation_indices = process_chain(tokenizer, chain)

                # Create text for model input
                text = tokenizer.decode(tokens)

                # Extract activations at each layer
                layer_activations = {}

                # Run single forward pass and extract all layer activations
                with model.trace(text) as tracer:
                    saved_acts = {}
                    for layer_idx in layers:
                        # Extract activation from this layer (shape: [batch, seq_len, hidden_dim])
                        saved_acts[layer_idx] = model.model.layers[layer_idx].output[0][0].save()

                # Store activations (move to CPU and convert to float32)
                for layer_idx in layers:
                    layer_activations[layer_idx] = saved_acts[layer_idx].float().cpu()

                torch.cuda.empty_cache()

                # Create binary labels for all categories
                seq_length = len(tokens)
                labels = create_binary_labels(seq_length, annotation_indices)

                # Store results
                all_activations.append(layer_activations)
                all_labels.append(labels)
                all_metadata.append({
                    'chain_id': chain.get('task_id', f'chain_{chain_idx}'),
                    'seq_length': seq_length,
                    'tokens': tokens
                })

            except Exception as e:
                print(f"Error processing chain {chain_idx}: {e}")
                continue

    # Save to disk
    print(f"Saving activations to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(all_activations, f)

    print(f"Saving labels to {labels_file}...")
    with open(labels_file, 'wb') as f:
        pickle.dump({'labels': all_labels, 'metadata': all_metadata}, f)

    print("Done!")


def main():
    """Main extraction pipeline."""
    # Configuration
    DATA_FILE = "/workspace/all_annotated_chains.json"
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    LAYERS = [8, 10, 12, 14, 16]
    OUTPUT_DIR = "/workspace/probe_training/data"

    # Set HuggingFace token if needed
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HuggingFace token from environment")

    # Load data
    print("Loading annotated chains...")
    chains = load_annotated_chains(DATA_FILE)
    print(f"Loaded {len(chains)} chains")

    # Split into train/val
    print("Splitting into train/val...")
    train_chains, val_chains = train_val_split(chains, val_ratio=0.2, random_seed=42)
    print(f"Train: {len(train_chains)}, Val: {len(val_chains)}")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = LanguageModel(MODEL_NAME, device_map="cuda", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Extract activations for training set
    print("\n=== Extracting training activations ===")
    extract_activations_from_chains(
        train_chains,
        model,
        tokenizer,
        layers=LAYERS,
        output_file=os.path.join(OUTPUT_DIR, "train_activations.pkl"),
        labels_file=os.path.join(OUTPUT_DIR, "train_labels.pkl")
    )

    # Extract activations for validation set
    print("\n=== Extracting validation activations ===")
    extract_activations_from_chains(
        val_chains,
        model,
        tokenizer,
        layers=LAYERS,
        output_file=os.path.join(OUTPUT_DIR, "val_activations.pkl"),
        labels_file=os.path.join(OUTPUT_DIR, "val_labels.pkl")
    )

    print("\nAll activations extracted successfully!")


if __name__ == "__main__":
    main()
