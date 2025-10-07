# Quick Start Guide - Backtracking Probe Training

## 🚀 Deploy to RunPod (3 commands)

```bash
# 1. Upload
runpod sync /workspace/probe_training /workspace/probe_training && \
runpod sync /workspace/all_annotated_chains.json /workspace/all_annotated_chains.json

# 2. Setup
runpod run "cd /workspace/probe_training && bash setup_runpod.sh"

# 3. Run (~3-4 hours)
runpod run "cd /workspace/probe_training && python run_experiment.py"

# 4. Download results
runpod sync /workspace/probe_training/results ./results --download
```

## 📊 What You'll Get

After ~3-4 hours, you'll have:
- **35 trained probes** (5 layers × 7 lags)
- **CSV with all metrics** (`probe_results.csv`)
- **5 interactive visualizations** (HTML files)

## 🔍 Key Questions Answered

1. Can we predict backtracking before it happens? (Check lag < 0 performance)
2. Which layers best predict backtracking? (Compare across layers)
3. How far in advance? (Find best negative lag)

## 📁 Main Files

| File | Purpose |
|------|---------|
| `run_experiment.py` | Run complete pipeline |
| `extract_activations.py` | Step 1: Get activations |
| `train_probes.py` | Step 2: Train probes |
| `visualize_results.py` | Step 3: Make plots |
| `test_setup.py` | Verify setup |

## ⚙️ Configuration

- **Model**: Llama-3.1-8B
- **Layers**: [8, 10, 12, 14, 16]
- **Lags**: [-16, -12, -8, -4, 0, +4, +8]
- **Data**: 1000 chains (800 train, 200 val)
- **Label**: backtracking

## 💰 Cost Estimate

- **GPU**: A100 40GB @ ~$1.50/hr
- **Runtime**: 3-4 hours
- **Total**: ~$5-6

## 🐛 Troubleshooting

```bash
# Check if running
runpod run "ps aux | grep python"

# View logs (if using nohup)
runpod run "tail -f /workspace/probe_training/experiment.log"

# Check GPU
runpod run "nvidia-smi"
```

## 📈 After Completion

1. Open `results/probe_results.csv`
2. View HTML visualizations in browser
3. Look for best (layer, lag) combinations
4. Analyze if negative lags work (= predicting future)

## 🔄 Rerun Specific Steps

```bash
# Just extraction
runpod run "cd /workspace/probe_training && python extract_activations.py"

# Just training (if activations exist)
runpod run "cd /workspace/probe_training && python train_probes.py"

# Just visualization (if results exist)
runpod run "cd /workspace/probe_training && python visualize_results.py"
```

---

**Full documentation**: See `README.md` and `RUNPOD_DEPLOYMENT.md`
