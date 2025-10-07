# RunPod Deployment Guide

Quick guide for running the experiment on RunPod.

## Prerequisites

1. **RunPod account** with GPU instance
2. **RunPod CLI** installed locally
3. **Data file**: `all_annotated_chains.json` at `/workspace/`

## Step 1: Upload Code and Data

From your local machine:

```bash
# Upload the probe training code
runpod sync /workspace/probe_training /workspace/probe_training

# Upload the data file
runpod sync /workspace/all_annotated_chains.json /workspace/all_annotated_chains.json
```

## Step 2: Setup Environment

```bash
# Run setup script
runpod run "cd /workspace/probe_training && bash setup_runpod.sh"
```

This will:
- Install all Python dependencies
- Verify GPU availability
- Check data file exists
- Create necessary directories

## Step 3: Run the Experiment

### Option A: Full Pipeline (Recommended)

Run everything in one go (~3-4 hours):

```bash
runpod run "cd /workspace/probe_training && python run_experiment.py"
```

### Option B: Step-by-Step

Run each step separately for better control:

```bash
# Step 1: Extract activations (2-3 hours)
runpod run "cd /workspace/probe_training && python extract_activations.py"

# Step 2: Train probes (1-2 hours)
runpod run "cd /workspace/probe_training && python train_probes.py"

# Step 3: Generate visualizations (<1 min)
runpod run "cd /workspace/probe_training && python visualize_results.py"
```

### Option C: Background Execution

For long-running tasks, use `nohup` to prevent disconnection:

```bash
runpod run "cd /workspace/probe_training && nohup python run_experiment.py > experiment.log 2>&1 &"

# Check progress
runpod run "tail -f /workspace/probe_training/experiment.log"
```

## Step 4: Monitor Progress

### Check if process is running
```bash
runpod run "ps aux | grep python"
```

### View logs
```bash
# If using nohup
runpod run "tail -f /workspace/probe_training/experiment.log"

# Or check GPU usage
runpod run "nvidia-smi"
```

## Step 5: Download Results

Once complete, download results to your local machine:

```bash
# Download all results
runpod sync /workspace/probe_training/results ./results --download

# Or download specific files
runpod sync /workspace/probe_training/results/probe_results.csv ./probe_results.csv --download
```

## Recommended RunPod Configuration

- **GPU**: A100 (40GB or 80GB) or H100
- **Template**: PyTorch 2.0+ with CUDA 11.8+
- **Storage**: At least 50GB
- **Network**: Stable connection for syncing

## Resource Usage

| Component | GPU Memory | Time |
|-----------|------------|------|
| Activation extraction | ~20GB | 2-3 hours |
| Probe training | ~5GB | 1-2 hours |
| Visualization | <1GB | <1 minute |

## Troubleshooting

### Out of Memory
If you get OOM errors:
1. Check GPU memory: `runpod run "nvidia-smi"`
2. Reduce batch size in `train_probes.py` (line ~130)
3. Process fewer chains at once

### Connection Lost
If connection drops during execution:
1. The process should continue running if using `nohup`
2. Check progress: `runpod run "ps aux | grep python"`
3. View logs: `runpod run "tail /workspace/probe_training/experiment.log"`

### Missing Dependencies
If imports fail:
```bash
runpod run "cd /workspace/probe_training && pip install -r requirements.txt"
```

### Data File Not Found
Verify data file location:
```bash
runpod run "ls -lh /workspace/all_annotated_chains.json"
```

## Quick Commands Cheat Sheet

```bash
# Check RunPod status
runpod run "hostname && nvidia-smi"

# View results summary
runpod run "cd /workspace/probe_training && python -c 'import pandas as pd; df = pd.read_csv(\"results/probe_results.csv\"); print(df.nlargest(10, \"val_f1\"))'"

# Clean up (if rerunning)
runpod run "cd /workspace/probe_training && rm -rf data/*.pkl models/*.pt results/*.html results/*.csv"

# Download everything
runpod sync /workspace/probe_training ./probe_training_backup --download
```

## Expected Output

After successful completion, you should have:

```
probe_training/
├── data/
│   ├── train_activations.pkl  (~2-5GB)
│   ├── train_labels.pkl       (~10MB)
│   ├── val_activations.pkl    (~500MB-1GB)
│   └── val_labels.pkl         (~2MB)
├── models/
│   └── probe_layer{X}_lag{Y}.pt  (35 files, ~16KB each)
├── results/
│   ├── probe_results.csv
│   ├── heatmap_f1.html
│   ├── heatmap_accuracy.html
│   ├── lineplot_f1.html
│   ├── comparison_metrics.html
│   └── class_balance.html
```

## Cost Estimate

Assuming A100 40GB at ~$1.50/hour:
- Experiment runtime: 3-4 hours
- **Total cost**: ~$5-6

## Next Steps After Completion

1. Download and review visualizations
2. Analyze `probe_results.csv` for trends
3. Identify best performing (layer, lag) combinations
4. Investigate which layers predict backtracking furthest in advance
