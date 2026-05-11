"""
Main Experiment Runner (V4 — reordered pipeline)

Orchestrates the entire experimental pipeline.

V4 changes:
  - Reordered steps: Probing moved from 9→4 (after Error Analysis)
  - New pipeline order:
      1. Data Preparation
      2. Model Inference
      3. Error Analysis
      4. Probability Probing (was Step 9)
      5. CKA Analysis (was Step 4)
      6. Projection Matrix Training (was Step 5)
      7. Injection Experiments (was Step 6)
      8. Visualization (was Step 7)
      9. Summary Report (was Step 8)
  - Updated file mappings and CLI help
"""

import os
import sys
import argparse
import subprocess
import time
from typing import List, Dict


def print_banner(message: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(message.center(80))
    print("=" * 80 + "\n")


def run_command(command: List[str], step_name: str) -> bool:
    """Run a subprocess command and handle errors."""
    print_banner(f"Starting: {step_name}")
    print(f"Command: {' '.join(command)}\n")

    start_time = time.time()
    try:
        result = subprocess.run(
            command, check=True, text=True, capture_output=False,
        )
        elapsed = time.time() - start_time
        print(f"\n  Step completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n  Step FAILED after {elapsed:.1f}s: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n  Step interrupted by user")
        return False


# Step names for display
STEP_NAMES = {
    1: "Data Preparation (HuggingFace QA)",
    2: "Model Inference",
    3: "Error Analysis",
    4: "Probability Probing (3-Model)",
    5: "CKA Analysis",
    6: "Projection Matrix Training",
    7: "Injection Experiments",
    8: "Visualization",
    9: "Summary Report",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run heterogeneous model alignment experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --all
  python run_experiment.py --steps 1 2 3 4
  python run_experiment.py --all --max_samples 10  (for testing)
  python run_experiment.py --steps 6 7 --small_model qwen1.5B
  python run_experiment.py --steps 4 --max_probing_samples 20
  python run_experiment.py --steps 4 --probing_models qwen1.5B qwen7B
        """,
    )

    # Step selection
    parser.add_argument("--all", action="store_true", help="Run all steps (1-9)")
    parser.add_argument("--steps", nargs="+", type=int, choices=range(1, 10),
                        help="Specific steps to run (1-9)")

    # Data preparation
    parser.add_argument("--total_samples", type=int, default=300)

    # Inference
    parser.add_argument("--model", type=str,
                        choices=["qwen1.5B", "qwen3B", "qwen7B", "all"], default="all")
    parser.add_argument("--max_samples", type=int, default=None)

    # Projection & injection
    parser.add_argument("--no_multiple_layers", action="store_true")
    parser.add_argument("--small_model", type=str, default="qwen1.5B",
                        choices=["qwen1.5B", "qwen3B"])
    parser.add_argument("--large_model", type=str, default="qwen7B",
                        choices=["qwen7B"])

    # Injection
    parser.add_argument("--max_injection_samples", type=int, default=50)

    # Visualization
    parser.add_argument("--num_vis_cases", type=int, default=10)
    parser.add_argument("--skip_individual_plots", action="store_true")

    # Probing (Step 4)
    parser.add_argument("--probing_models", nargs="+", type=str,
                        choices=["qwen1.5B", "qwen3B", "qwen7B"],
                        default=["qwen1.5B", "qwen3B", "qwen7B"],
                        help="Models to probe in Step 4 (default: all 3)")
    parser.add_argument("--max_probing_samples", type=int, default=None,
                        help="Max samples for Step 4 probing (default: all error samples)")

    # Control
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    # Determine steps
    if args.all:
        steps_to_run = list(range(1, 10))
    elif args.steps:
        steps_to_run = sorted(args.steps)
    else:
        parser.print_help()
        print("\nError: Must specify either --all or --steps")
        sys.exit(1)

    print_banner("Heterogeneous Model Alignment Experiment")
    print(f"  Steps: {steps_to_run}")
    print(f"  Continue on error: {args.continue_on_error}")
    print(f"  Small model: {args.small_model}, Large model: {args.large_model}")

    results: Dict[int, bool] = {}
    start_time = time.time()

    for step in steps_to_run:
        cmd = [sys.executable]
        success = False

        if step == 1:
            cmd = [sys.executable, "step1_prepare_data.py",
                   "--total_samples", str(args.total_samples)]
            success = run_command(cmd, f"Step 1: {STEP_NAMES[1]}")

        elif step == 2:
            cmd = [sys.executable, "step2_run_inference.py", "--model", args.model]
            if args.max_samples:
                cmd.extend(["--max_samples", str(args.max_samples)])
            success = run_command(cmd, f"Step 2: {STEP_NAMES[2]}")

        elif step == 3:
            cmd = [sys.executable, "step3_error_analysis.py",
                   "--num_samples", str(args.total_samples)]
            success = run_command(cmd, f"Step 3: {STEP_NAMES[3]}")

        elif step == 4:
            cmd = [sys.executable, "step4_probability_probing.py",
                   "--small_model", args.small_model,
                   "--large_model", args.large_model]
            # Add probing models (pass all in a single --models flag)
            cmd.extend(["--models"] + args.probing_models)
            if args.max_probing_samples:
                cmd.extend(["--max_samples", str(args.max_probing_samples)])
            success = run_command(cmd, f"Step 4: {STEP_NAMES[4]}")

        elif step == 5:
            cmd = [sys.executable, "step5_cka_analysis.py",
                   "--num_samples", str(args.total_samples)]
            success = run_command(cmd, f"Step 5: {STEP_NAMES[5]}")

        elif step == 6:
            cmd = [sys.executable, "step6_train_projection.py",
                   "--small_model", args.small_model,
                   "--large_model", args.large_model]
            if not args.no_multiple_layers:
                cmd.append("--try_multiple_layers")
            success = run_command(cmd, f"Step 6: {STEP_NAMES[6]}")

        elif step == 7:
            cmd = [sys.executable, "step7_injection_experiment.py",
                   "--max_samples", str(args.max_injection_samples),
                   "--small_model", args.small_model,
                   "--large_model", args.large_model]
            success = run_command(cmd, f"Step 7: {STEP_NAMES[7]}")

        elif step == 8:
            cmd = [sys.executable, "step8_visualization.py",
                   "--num_cases", str(args.num_vis_cases)]
            if args.skip_individual_plots:
                cmd.append("--skip_individual")
            success = run_command(cmd, f"Step 8: {STEP_NAMES[8]}")

        elif step == 9:
            cmd = [sys.executable, "step9_summary.py"]
            success = run_command(cmd, f"Step 9: {STEP_NAMES[9]}")

        results[step] = success

        if not success and not args.continue_on_error:
            print_banner("Experiment Terminated Due to Error")
            break

    # Final summary
    total_time = time.time() - start_time
    print_banner("Experiment Complete")

    print(f"{'Step':<6} {'Name':<45} {'Status'}")
    print("-" * 80)
    for step, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {step:<4} {STEP_NAMES[step]:<45} {status}")

    print("-" * 80)
    print(f"Total time: {total_time / 60:.1f} minutes")

    all_success = all(results.values())
    if all_success:
        print("\nAll steps completed successfully!")
        print("  Results: experiment_results/summary_report.json")
        print("  Analysis: experiment_results/analysis/")
        print("  Probing: experiment_results/probing/")
    else:
        print("\nSome steps failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()