"""Main script to run the complete probe training experiment."""

import os
import sys
import argparse
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with code {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Run backtracking probe training experiment")
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip activation extraction (use existing data)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip probe training (use existing results)'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization generation'
    )

    args = parser.parse_args()

    print("🚀 Starting Backtracking Probe Training Experiment")
    print(f"Working directory: {os.getcwd()}")

    # Check for data file
    data_file = "/workspace/all_annotated_chains.json"
    if not os.path.exists(data_file):
        print(f"\n❌ Error: Data file not found: {data_file}")
        print("Please ensure the annotated chains dataset is available")
        sys.exit(1)

    # Step 1: Extract activations
    if not args.skip_extraction:
        run_command(
            "python extract_activations.py",
            "Step 1: Extracting activations from model"
        )
    else:
        print("\n⏭️  Skipping activation extraction")

    # Step 2: Train probes
    if not args.skip_training:
        run_command(
            "python train_probes.py",
            "Step 2: Training linear probes"
        )
    else:
        print("\n⏭️  Skipping probe training")

    # Step 3: Generate visualizations
    if not args.skip_visualization:
        run_command(
            "python visualize_results.py",
            "Step 3: Generating visualizations"
        )
    else:
        print("\n⏭️  Skipping visualization generation")

    print("\n" + "="*60)
    print("🎉 Experiment completed successfully!")
    print("="*60)
    print("\nResults are available in:")
    print("  - probe_training/models/     (trained probe weights)")
    print("  - probe_training/results/    (CSV results and visualizations)")
    print("\nNext steps:")
    print("  1. Review probe_training/results/probe_results.csv")
    print("  2. Open HTML visualizations in probe_training/results/")
    print("  3. Analyze which layers and lags best predict backtracking")


if __name__ == "__main__":
    main()
