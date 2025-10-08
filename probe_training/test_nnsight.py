"""Test nnsight activation extraction to understand tensor shapes."""

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import os

# Load HF token
env_file = "/workspace/.env"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.strip().startswith('#'):
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

hf_token = os.environ.get("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.1-8B"

print("Loading model...")
model = LanguageModel(MODEL_NAME, device_map="cuda", torch_dtype=torch.bfloat16, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

# Simple test text
text = "The quick brown fox jumps over the lazy dog."

print(f"\nTest text: {text}")
print(f"Tokens: {tokenizer.encode(text)}")

# Test extraction
saved_acts = {}

print("\n=== Testing activation extraction ===")
with model.trace(text) as tracer:
    for layer_idx in [8, 10]:
        saved_acts[layer_idx] = model.model.layers[layer_idx].output[0].save()
        print(f"Inside trace - Layer {layer_idx}: saved reference")

print("\n=== After trace ===")
for layer_idx in [8, 10]:
    tensor = saved_acts[layer_idx]
    print(f"Layer {layer_idx}:")
    print(f"  Type: {type(tensor)}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")

    # Try different indexing approaches
    print(f"\n  Testing indexing:")
    print(f"  tensor.ndim = {tensor.ndim}")

    if tensor.ndim == 3:
        print(f"  tensor[0, :, :].shape = {tensor[0, :, :].shape}")
    elif tensor.ndim == 2:
        print(f"  Already 2D, shape = {tensor.shape}")
    else:
        print(f"  Unexpected ndim: {tensor.ndim}")

print("\n=== Test complete ===")
