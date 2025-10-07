# Backtracking Probe Training Experiment - Setup Complete ✅

## What Was Built

A complete pipeline to train linear probes on Llama-3.1-8B activations to predict backtracking-labeled tokens at different temporal lags.

### Research Question
**Can we predict when the model will backtrack before it happens?**

We test this by training probes on activations at time `t` to predict backtracking labels at time `t + lag`, where negative lags represent predicting the future.

## Project Structure

```
probe_training/
├── extract_activations.py       # Step 1: Extract activations from Llama-3.1-8B
├── train_probes.py              # Step 2: Train 35 linear probes (5 layers × 7 lags)
├── visualize_results.py         # Step 3: Generate visualizations
├── run_experiment.py            # Main pipeline (runs all steps)
├── test_setup.py                # Verify setup before running
├── setup_runpod.sh              # RunPod environment setup
├── requirements.txt             # Python dependencies
├── README.md                    # Full documentation
├── RUNPOD_DEPLOYMENT.md         # RunPod deployment guide
└── utils/
    └── data_processing.py       # Data processing utilities
```

## Quick Start

### Test Locally (Optional)
```bash
cd /workspace/probe_training
python test_setup.py
```

### Deploy to RunPod

1. **Upload code and data**
   ```bash
   runpod sync /workspace/probe_training /workspace/probe_training
   runpod sync /workspace/all_annotated_chains.json /workspace/all_annotated_chains.json
   ```

2. **Setup environment**
   ```bash
   runpod run "cd /workspace/probe_training && bash setup_runpod.sh"
   ```

3. **Run experiment** (~3-4 hours)
   ```bash
   runpod run "cd /workspace/probe_training && python run_experiment.py"
   ```

4. **Download results**
   ```bash
   runpod sync /workspace/probe_training/results ./results --download
   ```

## What The Pipeline Does

### Step 1: Extract Activations (2-3 hours)
- Loads 1000 annotated reasoning chains
- Splits into 800 train / 200 validation
- Processes each through Llama-3.1-8B
- Extracts activations from layers [8, 10, 12, 14, 16]
- Creates binary labels for backtracking tokens
- Saves to `data/train_activations.pkl` and `data/val_activations.pkl`

### Step 2: Train Probes (1-2 hours)
- Trains 35 linear probes (5 layers × 7 lags)
- **Lags tested**: -16, -12, -8, -4, 0, +4, +8
  - Negative lag = predict future (e.g., lag=-4 means activation[t] predicts label[t-4])
  - Lag=0 = predict current position
  - Positive lag = predict past (control)
- Uses binary cross-entropy loss with class balancing
- Early stopping on validation F1 score
- Saves probes to `models/probe_layer{X}_lag{Y}.pt`
- Saves results to `results/probe_results.csv`

### Step 3: Visualize Results (<1 minute)
- Heatmap: F1 score across (layer, lag) combinations
- Line plots: How F1 changes with lag for each layer
- Comparison: F1, accuracy, and AUROC side-by-side
- Class balance analysis
- Saves HTML visualizations to `results/`

## Expected Results

The pipeline produces:

### CSV Results (`results/probe_results.csv`)
Columns: `layer`, `lag`, `train_acc`, `train_f1`, `val_acc`, `val_f1`, `val_auc`, `n_train`, `n_val`, `pos_ratio_train`, `pos_ratio_val`

### Visualizations
- `heatmap_f1.html` - F1 score heatmap
- `heatmap_accuracy.html` - Accuracy heatmap
- `lineplot_f1.html` - F1 vs lag for each layer
- `comparison_metrics.html` - Multi-metric comparison
- `class_balance.html` - Positive class ratios

### Trained Probes
35 probe weights in `models/probe_layer{X}_lag{Y}.pt`

## Key Questions to Answer

After running the experiment, analyze the results to answer:

1. **Does performance degrade with negative lag?**
   - If yes → model represents backtracking in advance
   - If no → backtracking is spontaneous

2. **Which layers best predict future backtracking?**
   - Higher layers may have more semantic planning
   - Lower layers may be more reactive

3. **How far ahead can we predict?**
   - Maximum negative lag with good F1 score
   - Indicates planning horizon

4. **Is there a sweet spot?**
   - Certain (layer, lag) combinations may excel
   - Could indicate when backtracking decision is made

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/Llama-3.1-8B |
| Hidden Dimension | 4096 |
| Layers | [8, 10, 12, 14, 16] |
| Lags | [-16, -12, -8, -4, 0, 4, 8] |
| Primary Label | backtracking |
| Train/Val Split | 800/200 (80/20) |
| Probe Type | Linear (4096 → 1) |
| Loss | Binary cross-entropy + L2 reg |
| Optimizer | Adam (lr=1e-3) |
| Training | 50 epochs, early stopping |

## Resource Requirements

### RunPod Recommended
- **GPU**: A100 40GB or 80GB
- **Storage**: 50GB
- **Cost**: ~$5-6 (3-4 hours at $1.50/hour)

### Memory Usage
- Activation extraction: ~20GB GPU, ~10GB disk
- Probe training: ~5GB GPU, ~1GB disk
- Visualization: <1GB

## Files Generated

```
data/
├── train_activations.pkl    (~2-5 GB)   - Activations for 800 chains
├── train_labels.pkl         (~10 MB)    - Labels for 800 chains
├── val_activations.pkl      (~500 MB)   - Activations for 200 chains
└── val_labels.pkl           (~2 MB)     - Labels for 200 chains

models/
└── probe_layer{X}_lag{Y}.pt (35 × 16KB) - Trained probe weights

results/
├── probe_results.csv        (~5 KB)     - All metrics
├── heatmap_f1.html          (~100 KB)   - F1 heatmap
├── heatmap_accuracy.html    (~100 KB)   - Accuracy heatmap
├── lineplot_f1.html         (~100 KB)   - F1 line plot
├── comparison_metrics.html  (~200 KB)   - Multi-metric plot
└── class_balance.html       (~100 KB)   - Class distribution
```

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size in `train_probes.py` line ~130

### Issue: Connection drops
**Solution**: Use `nohup` for background execution
```bash
runpod run "cd /workspace/probe_training && nohup python run_experiment.py > experiment.log 2>&1 &"
```

### Issue: Missing dependencies
**Solution**: Reinstall requirements
```bash
runpod run "cd /workspace/probe_training && pip install -r requirements.txt"
```

## Next Steps After Results

1. **Download and review** all HTML visualizations
2. **Identify best probes** by F1 score
3. **Compare layers** to see which represents backtracking best
4. **Analyze lag patterns** to understand planning horizon
5. **Write up findings** for paper/presentation

## Additional Analysis (Future Work)

Potential extensions:
- Test other label categories (uncertainty-estimation, deduction, etc.)
- Train on other models (DeepSeek-R1-Distill-Llama-8B)
- Analyze probe weights to find important dimensions
- Test non-linear probes (MLP)
- Investigate failure cases where prediction fails

## Citation

Based on data from:
- Repository: https://github.com/jnward/latent-backtracking
- Data: `new_annotated_chains/all_annotated_chains.json`

## Support

For issues or questions:
1. Check `README.md` and `RUNPOD_DEPLOYMENT.md`
2. Run `python test_setup.py` to diagnose problems
3. Review logs in `experiment.log` (if using nohup)
