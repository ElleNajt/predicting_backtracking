"""Test script to verify setup is correct before running full experiment."""

import os
import sys
import json
import torch


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        from transformers import AutoTokenizer
        from nnsight import LanguageModel
        import plotly
        import tqdm
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_data_file():
    """Test that data file exists and is valid."""
    print("\nTesting data file...")
    data_path = "/workspace/all_annotated_chains.json"

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)

        print(f"✅ Data file loaded: {len(data)} chains")

        # Check first chain structure
        if len(data) > 0:
            required_keys = ['task_id', 'problem', 'reasoning_chain', 'annotated_chain']
            missing = [k for k in required_keys if k not in data[0]]
            if missing:
                print(f"❌ Missing keys in first chain: {missing}")
                return False
            print(f"✅ Data structure valid")

        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")

    if not torch.cuda.is_available():
        print("⚠️  Warning: GPU not available, will use CPU (very slow)")
        return False

    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return True


def test_data_processing():
    """Test data processing pipeline on one sample."""
    print("\nTesting data processing...")

    try:
        from utils.data_processing import (
            load_annotated_chains,
            extract_annotations,
            process_chain,
            create_binary_labels
        )
        from transformers import AutoTokenizer

        # Load one chain
        chains = load_annotated_chains("/workspace/all_annotated_chains.json")
        test_chain = chains[0]

        # Test annotation extraction
        annotations = extract_annotations(test_chain['annotated_chain'])
        print(f"✅ Extracted {len(annotations)} annotations")

        # Test tokenization
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        tokens, annotation_indices = process_chain(tokenizer, test_chain)
        print(f"✅ Tokenized: {len(tokens)} tokens")

        # Test label creation
        labels = create_binary_labels(len(tokens), annotation_indices)
        print(f"✅ Created labels for {len(labels)} categories")

        # Print label statistics
        for category, label_array in labels.items():
            n_positive = label_array.sum()
            if n_positive > 0:
                print(f"   {category}: {n_positive}/{len(label_array)} positive tokens")

        return True

    except Exception as e:
        print(f"❌ Data processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directories...")

    dirs = ['data', 'models', 'results', 'utils']
    all_exist = True

    for d in dirs:
        path = os.path.join('/workspace/probe_training', d)
        if os.path.exists(path):
            print(f"✅ {d}/ exists")
        else:
            print(f"❌ {d}/ missing")
            all_exist = False

    return all_exist


def test_model_loading():
    """Test that model can be loaded (may be slow)."""
    print("\nTesting model loading (this may take a minute)...")

    try:
        from nnsight import LanguageModel

        print("   Loading model...")
        model = LanguageModel(
            "meta-llama/Llama-3.1-8B",
            device_map="cpu",  # Use CPU for test to avoid OOM
            torch_dtype=torch.float32
        )

        print("✅ Model loaded successfully")

        # Clean up
        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Model loading error: {e}")
        print("   (This might be a HuggingFace token issue)")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PROBE TRAINING SETUP TEST")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Data File", test_data_file),
        ("GPU", test_gpu),
        ("Directories", test_directories),
        ("Data Processing", test_data_processing),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    # Optional: test model loading (slow)
    print("\n" + "="*60)
    print("Optional: Test model loading? (slow, ~1 minute)")
    print("Type 'yes' to test model loading, or press Enter to skip")
    response = input("> ").strip().lower()

    if response == 'yes':
        results["Model Loading"] = test_model_loading()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed! Ready to run experiment.")
        print("\nNext steps:")
        print("  1. Upload to RunPod: runpod sync /workspace/probe_training /workspace/probe_training")
        print("  2. Run setup: runpod run 'cd /workspace/probe_training && bash setup_runpod.sh'")
        print("  3. Run experiment: runpod run 'cd /workspace/probe_training && python run_experiment.py'")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before running experiment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
