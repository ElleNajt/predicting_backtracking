"""Reorganize existing activations from all-layers to per-layer files."""

import pickle
import os
from tqdm import tqdm

DATA_DIR = "/workspace/probe_training/data"
LAYERS = [8, 10, 12, 14, 16]

def reorganize_split(split_name):
    """Reorganize activations for train or val split."""
    print(f"\n=== Reorganizing {split_name} activations ===")

    # Load the big file
    act_file = os.path.join(DATA_DIR, f"{split_name}_activations.pkl")
    print(f"Loading {act_file}...")
    with open(act_file, 'rb') as f:
        all_activations = pickle.load(f)  # List of dicts {layer: tensor}

    print(f"Loaded {len(all_activations)} chains")

    # Reorganize by layer
    for layer in LAYERS:
        print(f"\nExtracting layer {layer}...")
        layer_activations = []

        for chain_acts in tqdm(all_activations):
            if layer in chain_acts:
                layer_activations.append(chain_acts[layer])

        # Save this layer
        layer_file = os.path.join(DATA_DIR, f"{split_name}_layer{layer}_activations.pkl")
        print(f"Saving to {layer_file}...")
        with open(layer_file, 'wb') as f:
            pickle.dump(layer_activations, f)

        print(f"Layer {layer}: {len(layer_activations)} chains saved")

    print(f"\n✅ {split_name} reorganization complete!")


def main():
    print("Reorganizing activations into per-layer files...")
    print("This avoids loading all 20GB at once during training.")

    reorganize_split("train")
    reorganize_split("val")

    print("\n✅ All done! You can now delete the old files to save space:")
    print("  rm data/train_activations.pkl data/val_activations.pkl")


if __name__ == "__main__":
    main()
