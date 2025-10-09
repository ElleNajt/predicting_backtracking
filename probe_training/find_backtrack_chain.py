"""Find a chain with actual backtracking annotations and good probe performance."""

import json

# Load chains
with open('/workspace/all_annotated_chains.json') as f:
    chains = json.load(f)

# Validation set
val_chains = chains[800:]

# List of top chains from previous analysis
top_chain_ids = ['systems_63', 'systems_52', 'systems_2', 'verbal_94', 'verbal_65',
                 'verbal_97', 'systems_25', 'systems_73', 'verbal_43', 'verbal_22']

print("Checking which top chains have backtracking annotations:\n")

for task_id in top_chain_ids:
    for chain in val_chains:
        if chain.get('task_id') == task_id:
            has_backtrack = 'backtracking' in chain.get('annotated_chain', '')
            if has_backtrack:
                # Count approximate backtrack occurrences
                count = chain['annotated_chain'].count('["backtracking"]')
                print(f"✓ {task_id}: HAS backtracking ({count} annotations)")
            else:
                print(f"✗ {task_id}: no backtracking")
            break
