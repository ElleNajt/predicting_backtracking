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
    labels_file="labels.pkl",
    batch_size=1
):
    """
    Extract activations from model for all chains, saving in batches to avoid OOM.

    Args:
        chains: List of chain dictionaries
        model: nnsight LanguageModel
        tokenizer: HuggingFace tokenizer
        layers: List of layer indices to extract from
        output_file: Path to save activations
        labels_file: Path to save labels
        batch_size: Save to disk every N chains to avoid OOM

    Returns:
        None (saves to disk)
    """
    all_activations = []
    all_labels = []
    all_metadata = []

    print(f"Processing {len(chains)} chains in batches of {batch_size}...")

    with torch.inference_mode():
        for chain_idx, chain in enumerate(tqdm(chains)):
            try:
                # Process chain to get tokens and annotation indices
                tokens, annotation_indices = process_chain(tokenizer, chain)

                # Create text for model input
                text = tokenizer.decode(tokens)

                # Extract activations at each layer
                saved_acts = {}

                # Run single forward pass and extract all layer activations
                with model.trace(text) as tracer:
                    for layer_idx in layers:
                        # Extract activation from this layer
                        # output[0] is the hidden states tensor [batch, seq_len, hidden_dim]
                        saved_acts[layer_idx] = model.model.layers[layer_idx].output[0].save()

                # After trace, access saved tensors and process them
                # Tensor is already [seq_len, hidden_dim] - nnsight removes batch dim
                layer_activations = {}
                for layer_idx in layers:
                    # Access the saved tensor (already 2D: [seq_len, hidden_dim])
                    tensor = saved_acts[layer_idx]
                    # Keep as bfloat16 to save memory, move to CPU
                    layer_activations[layer_idx] = tensor.cpu()

                # Delete saved_acts to free GPU memory immediately
                del saved_acts
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

                # Save batch to disk and clear memory
                if (chain_idx + 1) % batch_size == 0:
                    batch_num = (chain_idx + 1) // batch_size
                    batch_act_file = output_file.replace('.pkl', f'_batch{batch_num}.pkl')
                    batch_lab_file = labels_file.replace('.pkl', f'_batch{batch_num}.pkl')

                    with open(batch_act_file, 'wb') as f:
                        pickle.dump(all_activations, f)
                    with open(batch_lab_file, 'wb') as f:
                        pickle.dump({'labels': all_labels, 'metadata': all_metadata}, f)

                    # Clear memory
                    all_activations = []
                    all_labels = []
                    all_metadata = []
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing chain {chain_idx}: {e}")
                continue

    # Save final batch if there's remaining data
    if all_activations:
        batch_num = (len(chains) - 1) // batch_size + 1
        batch_act_file = output_file.replace('.pkl', f'_batch{batch_num}.pkl')
        batch_lab_file = labels_file.replace('.pkl', f'_batch{batch_num}.pkl')

        with open(batch_act_file, 'wb') as f:
            pickle.dump(all_activations, f)
        with open(batch_lab_file, 'wb') as f:
            pickle.dump({'labels': all_labels, 'metadata': all_metadata}, f)

    # Combine all batches into final files
    print(f"\nCombining batches into final files...")
    import glob

    all_activations = []
    all_labels = []
    all_metadata = []

    act_pattern = output_file.replace('.pkl', '_batch*.pkl')
    lab_pattern = labels_file.replace('.pkl', '_batch*.pkl')

    for batch_file in sorted(glob.glob(act_pattern)):
        with open(batch_file, 'rb') as f:
            all_activations.extend(pickle.load(f))
        os.remove(batch_file)

    for batch_file in sorted(glob.glob(lab_pattern)):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
            all_labels.extend(batch_data['labels'])
            all_metadata.extend(batch_data['metadata'])
        os.remove(batch_file)

    # Save final combined files
    print(f"Saving final activations to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(all_activations, f)

    print(f"Saving final labels to {labels_file}...")
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
    # Try to load from .env file
    env_file = "/workspace/.env"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HuggingFace token from environment")
    else:
        print("Warning: No HF_TOKEN found")

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
    model = LanguageModel(MODEL_NAME, device_map="cuda", torch_dtype=torch.bfloat16, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

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
