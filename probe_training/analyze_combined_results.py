"""Analyze combined ROC AUC and PR AUC results."""

import pandas as pd

# Load both result files
roc_results = pd.read_csv('/workspace/probe_training/probe_results.csv')
pr_results = pd.read_csv('/workspace/probe_training/results/pr_auc_results.csv/pr_auc_results.csv')

# Merge on layer and lag
combined = pd.merge(roc_results, pr_results, on=['layer', 'lag'], how='inner')

# Select key columns
analysis = combined[['layer', 'lag', 'val_f1', 'val_auc', 'pr_auc', 'val_acc']].copy()

# Sort by PR AUC
analysis_sorted = analysis.sort_values('pr_auc', ascending=False)

print("=== Top 15 Probes by PR AUC ===")
print(analysis_sorted.head(15).to_string(index=False))

print("\n=== Lag 0 Performance (Current Position Backtracking) ===")
lag0 = analysis[analysis['lag'] == 0].sort_values('pr_auc', ascending=False)
print(lag0.to_string(index=False))

print("\n=== Lag +48 Performance (48 tokens ahead) ===")
lag48 = analysis[analysis['lag'] == 48].sort_values('pr_auc', ascending=False)
print(lag48.to_string(index=False))

print("\n=== PR AUC vs ROC AUC Comparison (Lag 0) ===")
print("Layer  ROC_AUC  PR_AUC   Ratio(PR/ROC)")
for _, row in lag0.iterrows():
    ratio = row['pr_auc'] / row['val_auc'] if row['val_auc'] > 0 else 0
    print(f"{int(row['layer']):5d}  {row['val_auc']:.4f}   {row['pr_auc']:.4f}   {ratio:.3f}")

# Save combined results
combined.to_csv('/workspace/probe_training/results/combined_results.csv', index=False)
print("\nCombined results saved to /workspace/probe_training/results/combined_results.csv")
