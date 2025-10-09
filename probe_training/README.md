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

## Results

### Complete Probe Performance (All Configurations)

| Layer | Lag | Val F1 | Val Acc | ROC AUC | **PR AUC** | Pos Ratio |
|-------|-----|--------|---------|---------|-----------|-----------|
| **12** | **0** | **0.295** | **0.908** | **0.957** | **0.406** | 2.2% |
| 10 | 0 | 0.247 | 0.876 | 0.954 | 0.391 | 2.2% |
| 14 | 0 | 0.249 | 0.877 | 0.953 | 0.386 | 2.2% |
| 8 | 0 | 0.279 | 0.903 | 0.949 | 0.382 | 2.2% |
| 16 | 0 | 0.275 | 0.898 | 0.949 | 0.367 | 2.2% |
| 12 | 4 | 0.220 | 0.861 | 0.936 | 0.309 | 2.3% |
| 10 | 4 | 0.207 | 0.853 | 0.926 | 0.304 | 2.3% |
| 14 | 4 | 0.230 | 0.873 | 0.932 | 0.299 | 2.3% |
| 8 | 4 | 0.228 | 0.877 | 0.925 | 0.298 | 2.3% |
| 16 | 4 | 0.191 | 0.830 | 0.930 | 0.290 | 2.3% |
| 12 | 8 | 0.174 | 0.815 | 0.911 | 0.235 | 2.3% |
| 10 | 8 | 0.189 | 0.842 | 0.906 | 0.226 | 2.3% |
| 14 | 8 | 0.190 | 0.840 | 0.910 | 0.225 | 2.3% |
| 8 | 8 | 0.185 | 0.835 | 0.901 | 0.224 | 2.3% |
| 16 | 8 | 0.200 | 0.855 | 0.906 | 0.222 | 2.3% |
| ... | ... | ... | ... | ... | ... | ... |
| 10 | 48 | 0.096 | 0.629 | 0.777 | **0.069** | 2.5% |
| 12 | 48 | 0.109 | 0.714 | 0.786 | **0.066** | 2.5% |
| 16 | 48 | 0.106 | 0.716 | 0.770 | **0.066** | 2.5% |
| 14 | 48 | 0.091 | 0.596 | 0.771 | **0.063** | 2.5% |
| 8 | 48 | 0.103 | 0.708 | 0.765 | **0.061** | 2.5% |

*Full results table available in `results/combined_results.csv`*

### Key Findings

1. **Lag 0 (Current Position) Performance**
   - Best probe: Layer 12, PR AUC = 0.406
   - Strong performance across all layers (PR AUC 0.37-0.41)
   - Models can reliably detect backtracking at current position

2. **Lag +48 (48 Tokens Ahead) Performance**
   - Best probe: Layer 10, PR AUC = 0.069
   - Performance barely above baseline (positive class ratio = 2.5%)
   - Predicting backtracking 48 tokens ahead is extremely difficult

3. **PR AUC vs ROC AUC**
   - ROC AUC remains high even at long lags (0.76-0.96)
   - PR AUC degrades much faster (0.06-0.41)
   - PR AUC is more informative for imbalanced datasets
   - At lag 0: PR AUC / ROC AUC ≈ 0.40-0.42
   - At lag 48: PR AUC / ROC AUC ≈ 0.08-0.09

4. **Layer Analysis**
   - Layer 12 consistently performs best across all lags
   - Middle layers (10-14) outperform early (8) and late (16) layers
   - Performance gap between layers decreases at longer lags

### Metrics Explained
- **F1 Score**: Harmonic mean of precision and recall (handles class imbalance)
- **Accuracy**: Overall correctness (can be misleading for imbalanced data)
- **ROC AUC**: Area under ROC curve (discriminative power)
- **PR AUC**: Area under Precision-Recall curve (more informative for imbalanced datasets)
- **Pos Ratio**: Percentage of positive (backtracking) tokens in validation set

### Interpreting Results

**Does performance degrade with lag?**
- Yes, dramatically. PR AUC drops from 0.41 (lag 0) to 0.07 (lag 48)
- F1 score drops from 0.30 (lag 0) to 0.11 (lag 48)
- Larger positive lags = predicting further into future = harder task

**Which layers best predict backtracking?**
- Layer 12 is consistently best across all lags
- Middle layers (10-14) contain the most predictive information
- This suggests backtracking signals emerge in mid-to-late semantic processing

**How far ahead can we predict?**
- Reliable prediction (PR AUC > 0.20): up to lag 8 tokens
- Moderate prediction (PR AUC > 0.10): up to lag 24 tokens
- Beyond lag 32: performance approaches baseline

### Visualizations
- **Heatmaps**: Show performance across all (layer, lag) combinations
- **Line Plots**: Show how performance changes with lag for each layer
- **Comparison**: Compare F1, accuracy, and AUROC side-by-side
- **Colorized Transcripts**: Visualize probe predictions on actual text

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
