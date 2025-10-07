# Backtracking Probe Training Experiment

This project trains linear probes on language model activations to predict backtracking-labeled tokens at different temporal lags.

## Overview

The experiment investigates whether models internally represent upcoming backtracking events before they occur in the generated text. We train linear probes on activations from different layers to predict backtracking labels at various time lags.

## Dataset

- **Source**: [latent-backtracking repository](https://github.com/jnward/latent-backtracking)
- **File**: `all_annotated_chains.json`
- **Size**: 1000 annotated reasoning chains
- **Labels**: backtracking, initializing, deduction, uncertainty-estimation, example-testing, adding-knowledge

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

Ensure the annotated chains dataset is available at:
```
/workspace/all_annotated_chains.json
```

### Environment Variables

Set your HuggingFace token if needed:
```bash
export HF_TOKEN="your_token_here"
```

## Running the Experiment

### Full Pipeline

Run the complete experiment:
```bash
cd /workspace/probe_training
python run_experiment.py
```

### Step-by-Step Execution

1. **Extract Activations** (~2-3 hours)
   ```bash
   python extract_activations.py
   ```
   - Processes 1000 chains through Llama-3.1-8B
   - Extracts activations from layers [8, 10, 12, 14, 16]
   - Saves to `data/train_activations.pkl` and `data/val_activations.pkl`

2. **Train Probes** (~1-2 hours)
   ```bash
   python train_probes.py
   ```
   - Trains linear probes for all (layer, lag) combinations
   - Lags: [-16, -12, -8, -4, 0, 4, 8]
   - Saves trained probes to `models/`
   - Saves results to `results/probe_results.csv`

3. **Generate Visualizations** (~1 minute)
   ```bash
   python visualize_results.py
   ```
   - Creates heatmaps and line plots
   - Saves HTML visualizations to `results/`

### Partial Runs

Skip steps if data already exists:
```bash
python run_experiment.py --skip-extraction  # Use existing activations
python run_experiment.py --skip-training    # Use existing probe results
```

## RunPod Deployment

### Upload Files

```bash
# Sync code and data to RunPod
runpod sync /workspace/probe_training /workspace/probe_training
runpod sync /workspace/all_annotated_chains.json /workspace/all_annotated_chains.json
```

### Run Experiment

```bash
runpod run "cd /workspace/probe_training && python run_experiment.py"
```

### Download Results

```bash
runpod sync /workspace/probe_training/results ./results --download
```

## Directory Structure

```
probe_training/
├── data/                       # Extracted activations and labels
│   ├── train_activations.pkl
│   ├── train_labels.pkl
│   ├── val_activations.pkl
│   └── val_labels.pkl
├── models/                     # Trained probe weights
│   └── probe_layer{X}_lag{Y}.pt
├── results/                    # Results and visualizations
│   ├── probe_results.csv
│   ├── heatmap_f1.html
│   ├── lineplot_f1.html
│   └── comparison_metrics.html
├── utils/                      # Utility modules
│   └── data_processing.py
├── extract_activations.py      # Step 1: Extract activations
├── train_probes.py             # Step 2: Train probes
├── visualize_results.py        # Step 3: Generate visualizations
├── run_experiment.py           # Main pipeline script
├── requirements.txt
└── README.md
```

## Key Configuration

### Model
- **Base Model**: `meta-llama/Llama-3.1-8B`
- **Hidden Dimension**: 4096
- **Layers**: [8, 10, 12, 14, 16]

### Probes
- **Architecture**: Linear (4096 → 1)
- **Loss**: Binary cross-entropy with class weights
- **Regularization**: L2 penalty (λ=0.01)
- **Optimizer**: Adam (lr=1e-3)
- **Training**: 50 epochs with early stopping

### Lags
- **Lag = 0**: Predict backtracking at current position
- **Lag = -4**: Predict backtracking 4 tokens in the future
- **Lag = -8**: Predict backtracking 8 tokens in the future
- **Lag = -12**: Predict backtracking 12 tokens in the future
- **Lag = -16**: Predict backtracking 16 tokens in the future
- **Lag = +4/+8**: Predict backtracking 4/8 tokens in the past

## Interpreting Results

### Metrics
- **F1 Score**: Primary metric (handles class imbalance)
- **Accuracy**: Overall correctness
- **AUROC**: Discriminative power

### Key Questions
1. **Does performance degrade with lag?**
   - Compare F1 scores across negative lags
   - Larger negative lags = predicting further into future

2. **Which layers best predict backtracking?**
   - Compare performance across layers
   - Higher layers may have more semantic information

3. **How far ahead can we predict?**
   - Find the largest negative lag with good performance
   - Indicates how far in advance backtracking is represented

### Visualizations
- **Heatmaps**: Show performance across all (layer, lag) combinations
- **Line Plots**: Show how performance changes with lag for each layer
- **Comparison**: Compare F1, accuracy, and AUROC side-by-side

## Expected Runtime

With GPU (A100/H100):
- Activation extraction: 2-3 hours
- Probe training: 1-2 hours
- Visualization: <1 minute
- **Total**: ~3-4 hours

## Troubleshooting

### Out of Memory
- Reduce batch size in `train_probes.py`
- Process fewer chains at a time in `extract_activations.py`

### Missing Data
- Ensure `all_annotated_chains.json` is in `/workspace/`
- Check file permissions

### Import Errors
- Verify all requirements are installed
- Check Python path includes `/workspace/probe_training`

## Citation

If you use this code, please cite the original latent-backtracking work:
- Repository: https://github.com/jnward/latent-backtracking
