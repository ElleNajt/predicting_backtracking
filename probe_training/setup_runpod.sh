#!/bin/bash

# Setup script for RunPod environment

echo "🔧 Setting up probe training environment on RunPod"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Verify data file exists
if [ ! -f "/workspace/all_annotated_chains.json" ]; then
    echo "❌ Error: Data file not found at /workspace/all_annotated_chains.json"
    echo "Please upload the data file first"
    exit 1
fi

# Verify GPU availability
echo "🖥️  Checking GPU..."
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create directories if they don't exist
mkdir -p data models results

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the experiment:"
echo "  python run_experiment.py"
echo ""
echo "To run steps individually:"
echo "  1. python extract_activations.py"
echo "  2. python train_probes.py"
echo "  3. python visualize_results.py"
