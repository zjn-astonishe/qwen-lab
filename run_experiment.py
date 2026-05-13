"""
Main Experiment Runner (V7 — with causal proof pipeline)

Orchestrates the entire experimental pipeline.

V7 changes:
  - Added 3 causal-proof steps: 4b (Logit Cluster), 6b (Feature Distance), 7b (Cross-Model Decode)
  - Pipeline order:
      1.  Data Preparation
      2.  Model Inference
      3.  Error Analysis
      4.  Probability Probing (3-Model)
      4b. Logit Semantic Cluster Analysis    ← Phase I causal proof
      5.  CKA Analysis
      6.  Decisive Token Analysis
      6b. Feature Cosine Distance           ← Phase II causal proof
      7.  Projection Matrix Training
      7b. Cross-Model LM Head Decoding      ← Phase III causal proof
      8.  Injection Experiments
      9.  Visualization
      10. Summary Report
  - Steps 3-6b: analysis group; Steps 7-7b: intervention group; Steps 8-10: output group
  - New steps consume step2 .pt outputs, zero additional inference cost
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
    1:   "Data Preparation (HuggingFace QA)",
    2:   "Model Inference",
    3:   "Error Analysis",
    4:   "Probability Probing (3-Model)",
    "4b": "Logit Semantic Cluster Analysis (Phase I)",
    5:   "CKA Analysis",
    6:   "Decisive Token Analysis",
    "6b": "Feature Cosine Distance (Phase II)",
    7:   "Projection Matrix Training",
    "7b": "Cross-Model LM Head Decoding (Phase III)",
    8:   "Injection Experiments",
    9:   "Visualization",
    10:  "Summary Report",
}

# Execution order: defines the canonical run sequence
PIPELINE_ORDER = [
    1, 2, 3, 4, "4b", 5, 6, "6b", 7, "7b", 8, 9, 10,
]


def main():
    parser = argparse.ArgumentParser(
        description="Run heterogeneous model alignment experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --all
  python run_experiment.py --steps 1 2 3 4
  python run_experiment.py --all --max_samples 10  (for testing)
  python run_experiment.py --steps 7 8 --small_model qwen1.5B
  python run_experiment.py --steps 4 --max_probing_samples 20
  python run_experiment.py --steps 4 --probing_models qwen1.5B qwen7B
  python run_experiment.py --steps 6 --decisive_model qwen7B
  python run_experiment.py --steps 6 --decisive_model qwen3B --entropy_threshold 3.0
        """,
    )

    # Step selection
    parser.add_argument("--all", action="store_true", help="Run all steps (1-10 + 4b 6b 7b)")
    parser.add_argument("--steps", nargs="+", type=str,
                        help="Specific steps to run (e.g., 1 2 3 4 4b 6b 7b)")

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
                        choices=["qwen3B", "qwen7B"])

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

    # Decisive Token Analysis (Step 6)
    parser.add_argument("--decisive_model", type=str, default="qwen7B",
                        choices=["qwen1.5B", "qwen3B", "qwen7B"],
                        help="Model to analyze in Step 6 (default: qwen7B)")
    parser.add_argument("--entropy_percentile", type=float, default=90,
                        help="Percentile for entropy filtering in Step 6 (default: 90)")
    parser.add_argument("--decisive_max_display", type=int, default=12,
                        help="Max samples to display in Step 6 plots/report (default: 12)")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device for new steps 4b/6b/7b (default: cuda if available)")

    # Control
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        args.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    def parse_step_key(s: str):
        """Parse step key: integers as int, others as str."""
        try:
            return int(s)
        except ValueError:
            return s.lower()

    def validate_steps(keys):
        """Validate step keys against PIPELINE_ORDER."""
        valid = set(PIPELINE_ORDER)
        for k in keys:
            if k not in valid:
                parser.error(f"Invalid step '{k}'. Valid: {sorted(str(s) for s in valid)}")

    def sort_steps(keys):
        """Sort steps by pipeline order."""
        order_map = {s: i for i, s in enumerate(PIPELINE_ORDER)}
        return sorted(keys, key=lambda k: order_map[k])

    # Determine steps
    if args.all:
        steps_to_run = list(PIPELINE_ORDER)
    elif args.steps:
        steps_to_run = sort_steps([parse_step_key(s) for s in args.steps])
        validate_steps(steps_to_run)
    else:
        parser.print_help()
        print("\nError: Must specify either --all or --steps")
        sys.exit(1)

    print_banner("Heterogeneous Model Alignment Experiment")
    print(f"  Steps: {steps_to_run}")
    print(f"  Device: {args.device}")
    print(f"  Continue on error: {args.continue_on_error}")
    print(f"  Small model: {args.small_model}, Large model: {args.large_model}")

    results: Dict[int, bool] = {}
    start_time = time.time()

    for step in steps_to_run:
        cmd = [sys.executable]
        success = False

        if step == "4b":
            cmd = [sys.executable, "step4b_logit_cluster_analysis.py",
                   "--num_samples", str(args.total_samples),
                   "--device", args.device]
            success = run_command(cmd, f"Step 4b: {STEP_NAMES['4b']}")

        elif step == "6b":
            cmd = [sys.executable, "step6b_feature_cosine_distance.py",
                   "--num_samples", str(args.total_samples),
                   "--device", args.device]
            success = run_command(cmd, f"Step 6b: {STEP_NAMES['6b']}")

        elif step == "7b":
            cmd = [sys.executable, "step7b_cross_model_decode.py",
                   "--num_samples", str(args.total_samples),
                   "--small_model", args.small_model,
                   "--large_model", args.large_model,
                   "--device", args.device]
            success = run_command(cmd, f"Step 7b: {STEP_NAMES['7b']}")

        elif step == 1:
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
            cmd.extend(["--models"] + args.probing_models)
            if args.max_probing_samples:
                cmd.extend(["--max_samples", str(args.max_probing_samples)])
            success = run_command(cmd, f"Step 4: {STEP_NAMES[4]}")

        elif step == 5:
            cmd = [sys.executable, "step5_cka_analysis.py",
                   "--num_samples", str(args.total_samples)]
            success = run_command(cmd, f"Step 5: {STEP_NAMES[5]}")

        elif step == 6:
            cmd = [sys.executable, "step6_decisive_token.py",
                   "--model", args.decisive_model,
                   "--entropy_percentile", str(args.entropy_percentile),
                   "--max_display", str(args.decisive_max_display)]
            if args.max_samples:
                cmd.extend(["--num_samples", str(args.max_samples)])
            success = run_command(cmd, f"Step 6: {STEP_NAMES[6]}")

        elif step == 7:
            # Train projection matrices for ALL model pairs needed by step6b:
            #   qwen1.5B -> qwen7B, qwen3B -> qwen7B, qwen1.5B -> qwen3B
            all_pairs = [
                ("qwen1.5B", "qwen7B"),
                ("qwen3B", "qwen7B"),
                ("qwen1.5B", "qwen3B"),
            ]
            step7_success = True
            for s_model, l_model in all_pairs:
                cmd = [sys.executable, "step7_train_projection.py",
                       "--small_model", s_model,
                       "--large_model", l_model]
                if not args.no_multiple_layers:
                    cmd.append("--try_multiple_layers")
                pair_ok = run_command(cmd, f"Step 7: Projection {s_model} -> {l_model}")
                if not pair_ok:
                    step7_success = False
                    if not args.continue_on_error:
                        break
            success = step7_success

        elif step == 8:
            cmd = [sys.executable, "step8_injection_experiment.py",
                   "--max_samples", str(args.max_injection_samples),
                   "--small_model", args.small_model,
                   "--large_model", args.large_model]
            success = run_command(cmd, f"Step 8: {STEP_NAMES[8]}")

        elif step == 9:
            cmd = [sys.executable, "step9_visualization.py",
                   "--num_cases", str(args.num_vis_cases)]
            if args.skip_individual_plots:
                cmd.append("--skip_individual")
            success = run_command(cmd, f"Step 9: {STEP_NAMES[9]}")

        elif step == 10:
            cmd = [sys.executable, "step10_summary.py"]
            success = run_command(cmd, f"Step 10: {STEP_NAMES[10]}")

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
        step_str = str(step).rjust(3) if isinstance(step, int) else step.rjust(3)
        print(f"  {step_str} {STEP_NAMES[step]:<45} {status}")

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