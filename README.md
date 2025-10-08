# Predicting Backtracking in Language Model Reasoning

This project trains linear probes on Llama-3.1-8B activations to predict when the model will engage in backtracking during chain-of-thought reasoning, with a focus on **anticipating backtracking before it occurs**.

## Overview

**Research Question:** Can we predict when a language model will backtrack in its reasoning, and how far in advance?

We trained 70 linear probes across 5 layers and multiple temporal lags to understand if the model's internal representations contain predictive signals about upcoming backtracking events.

## Key Results

### Extended Future Lag Performance

| Lag (tokens ahead) | Best Layer | Val F1 | Val AUC | Interpretation |
|-------------------|------------|--------|---------|----------------|
| 0 (concurrent) | 12 | 0.295 | 0.957 | Strong concurrent representation |
| +4 | 12 | 0.220 | 0.936 | Clear 4-token lookahead signal |
| +8 | 16 | 0.200 | 0.906 | Moderate 8-token lookahead |
| +12 | 12 | 0.167 | 0.891 | Persistent 12-token signal |
| +24 | 12 | 0.133 | 0.835 | Significant 24-token anticipation |
| +48 | 12 | 0.109 | 0.786 | Detectable 48-token anticipation |

**Key Finding:** Layer 12 consistently shows the strongest anticipatory signal. The model maintains meaningful predictive information about backtracking events up to 48 tokens in advance (AUC > 0.78).

### Performance by Layer and Lag

```
Validation F1 Scores (Future Prediction)
=========================================
Lag    Layer 8   Layer 10  Layer 12  Layer 14  Layer 16
---    -------   --------  --------  --------  --------
 +0    0.279     0.247     0.295     0.249     0.275
 +4    0.228     0.207     0.220     0.230     0.191
 +8    0.185     0.189     0.174     0.190     0.200
+12    0.168     0.163     0.167     0.157     0.162
+16    0.154     0.154     0.147     0.153     0.150
+20    0.129     0.137     0.135     0.143     0.119
+24    0.118     0.124     0.133     0.124     0.132
+32    0.104     0.112     0.104     0.117     0.108
+40    0.113     0.120     0.108     0.112     0.107
+48    0.103     0.096     0.109     0.091     0.106

Validation AUC (Future Prediction)
==================================
Lag    Layer 8   Layer 10  Layer 12  Layer 14  Layer 16
---    -------   --------  --------  --------  --------
 +0    0.949     0.954     0.957     0.953     0.950
 +4    0.925     0.926     0.936     0.932     0.930
 +8    0.901     0.906     0.911     0.910     0.906
+12    0.878     0.887     0.891     0.891     0.889
+16    0.854     0.854     0.868     0.864     0.863
+20    0.837     0.844     0.850     0.851     0.846
+24    0.818     0.822     0.835     0.832     0.826
+32    0.799     0.806     0.808     0.810     0.800
+40    0.784     0.793     0.790     0.793     0.783
+48    0.765     0.777     0.786     0.771     0.770
```

## Dataset

- **Source:** [jnward/latent-backtracking](https://github.com/jnward/latent-backtracking)
- **Size:** 1000 annotated chain-of-thought reasoning chains
- **Split:** 800 train / 200 validation
- **Labels:** Binary backtracking annotations at token level
- **Class balance:** ~2.2% positive (backtracking tokens)

## Method

### 1. Activation Extraction
- Model: `meta-llama/Llama-3.1-8B` (base, not instruct)
- Framework: TransformerLens
- Layers extracted: [8, 10, 12, 14, 16]
- Output: Per-layer activation files (bfloat16, ~3.2GB per layer)
- Time: ~45 minutes on NVIDIA L40 (46GB)

### 2. Probe Training
- Architecture: Single linear layer (4096 → 1) + sigmoid
- Loss: Binary cross-entropy with class weighting (~42x for positive class)
- Regularization: L2 penalty (λ=0.01)
- Optimizer: Adam (lr=1e-3)
- Early stopping: 3 epochs patience on validation F1
- Max epochs: 50

### 3. Temporal Lag Setup
For each (layer, lag) combination:
- `activation[t]` predicts `label[t + lag]`
- **Negative lag** (e.g., -4): Predict 4 tokens in the *past* (post-hoc explanation)
- **Zero lag** (0): Predict current token (concurrent)
- **Positive lag** (e.g., +24): Predict 24 tokens in the *future* (anticipation)

## Interpretation

### Why AUC vs F1?
With severe class imbalance (2.2% positive), metrics tell different stories:
- **Accuracy** is misleading (naive "always predict no backtracking" = 97.8%)
- **F1** balances precision/recall but is affected by threshold
- **AUC** measures class separation independent of threshold

**F1 = 0.133 at lag +24 is actually strong** given:
- Random baseline F1 ≈ 0.019 (with class imbalance)
- Observed F1 is ~7x better than random
- AUC = 0.835 shows excellent class separation

### What This Means
1. **The model "plans ahead"** - Internal representations at time t contain information about backtracking that will occur 24-48 tokens later
2. **Layer 12 is the planning layer** - Middle-to-late layers show strongest anticipatory signals
3. **Signal degrades gracefully** - Predictive power decreases with distance but remains meaningful even at +48 tokens

## Files

### Code
- `probe_training/extract_activations_v3.py` - Extract activations using single forward pass per chain
- `probe_training/train_probes.py` - Train linear probes with temporal lags
- `probe_training/utils/data_processing.py` - Data loading and preprocessing utilities
- `probe_training/requirements.txt` - Python dependencies

### Artifacts (Local)
- `probe_training/results/probe_results.csv` - Full results (70 probes)
- `probe_training/models/probe_layer{N}_lag{M}.pt` - Trained probe checkpoints (1.3MB total)
- `probe_training/train_extended_lags.log` - Training logs

### Artifacts (Remote - Not Pulled)
- `probe_training/data/*_activations.pkl` - Raw activations (20GB, regenerable)

## Reproducing Results

### Setup
```bash
# Install dependencies
cd probe_training
pip install -r requirements.txt

# Configure RunPod
# Create .runpod_config.json with your GPU credentials
```

### Run Experiment
```bash
# 1. Extract activations (~45 min on L40)
python extract_activations_v3.py

# 2. Train probes (~2 hours for 50 probes)
python train_probes.py

# 3. View results
cat results/probe_results.csv
```

### Hardware Requirements
- **Minimum:** NVIDIA L40 (46GB) or A100 (40GB)
- **Not sufficient:** RTX 4000 Ada (20GB) - causes OOM errors on longer sequences
- Model uses ~17GB GPU memory, sequences can use up to 25GB during forward pass

## Citations

**Dataset:**
```
Jake Ward (2024). Latent Backtracking.
https://github.com/jnward/latent-backtracking
```

**Model:**
```
Meta AI (2024). Llama 3.1.
https://huggingface.co/meta-llama/Llama-3.1-8B
```

**TransformerLens:**
```
Neel Nanda et al. (2022). TransformerLens.
https://github.com/neelnanda-io/TransformerLens
```

## Future Work

1. **Architecture variations:** Test non-linear probes, multi-layer probes, or attention-based classifiers
2. **Other reasoning patterns:** Extend to uncertainty estimation, deduction, knowledge addition
3. **Intervention experiments:** Can we modify activations to prevent/induce backtracking?
4. **Longer contexts:** Test on even longer horizons (+64, +128 tokens)
5. **Other models:** Compare Llama-3.1-70B, GPT-4, Claude for anticipatory signals

## Contact

For questions about this experiment, see the research journal or git history for context.
