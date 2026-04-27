"""
Main Experiment Runner
Orchestrates the entire experimental pipeline
"""

import os
import sys
import argparse
import subprocess
import time
from typing import List


def print_banner(message: str):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(message.center(80))
    print("="*80 + "\n")


def run_command(command: List[str], step_name: str) -> bool:
    """
    Run a command and handle errors
    
    Returns:
        True if successful, False otherwise
    """
    print_banner(f"Starting: {step_name}")
    print(f"Command: {' '.join(command)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=False
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed successfully in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_name} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ {step_name} interrupted by user")
        return False


def run_step1_data_prep(use_synthetic: bool = False, num_samples: int = 200):
    """Step 1: Prepare data"""
    cmd = [sys.executable, "step1_prepare_data.py"]
    
    if use_synthetic:
        cmd.append("--use_synthetic")
    
    cmd.extend(["--num_samples", str(num_samples)])
    
    return run_command(cmd, "Step 1: Data Preparation")


def run_step2_inference(model: str = "all", max_samples: int = None):
    """Step 2: Run model inference"""
    cmd = [sys.executable, "step2_run_inference.py", "--model", model]
    
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    
    return run_command(cmd, f"Step 2: Model Inference ({model})")


def run_step3_error_analysis():
    """Step 3: Analyze errors"""
    cmd = [sys.executable, "step3_error_analysis.py"]
    return run_command(cmd, "Step 3: Error Analysis")


def run_step4_cka_analysis(num_samples: int = 200):
    """Step 4: CKA analysis"""
    cmd = [sys.executable, "step4_cka_analysis.py", "--num_samples", str(num_samples)]
    return run_command(cmd, "Step 4: CKA Analysis")


def run_step5_projection(try_multiple_layers: bool = True):
    """Step 5: Train projection matrix"""
    cmd = [sys.executable, "step5_train_projection.py"]
    
    if try_multiple_layers:
        cmd.append("--try_multiple_layers")
    
    return run_command(cmd, "Step 5: Projection Matrix Training")


def run_step6_injection(max_samples: int = 50):
    """Step 6: Injection experiments"""
    cmd = [sys.executable, "step6_injection_experiment.py", "--max_samples", str(max_samples)]
    return run_command(cmd, "Step 6: Injection Experiments")


def run_step7_visualization(num_cases: int = 10, skip_individual: bool = False):
    """Step 7: Create visualizations"""
    cmd = [sys.executable, "step7_visualization.py", "--num_cases", str(num_cases)]
    
    if skip_individual:
        cmd.append("--skip_individual")
    
    return run_command(cmd, "Step 7: Visualization")


def run_step8_summary():
    """Step 8: Generate summary report"""
    cmd = [sys.executable, "step8_summary.py"]
    return run_command(cmd, "Step 8: Summary Report")


def main():
    parser = argparse.ArgumentParser(
        description="Run heterogeneous model alignment experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps
  python run_experiment.py --all
  
  # Run specific steps
  python run_experiment.py --steps 1 2 3
  
  # Run with synthetic data (for testing)
  python run_experiment.py --all --use_synthetic --max_samples 10
  
  # Run inference only for small model
  python run_experiment.py --steps 2 --model qwen1.5B
        """
    )
    
    # Step selection
    parser.add_argument("--all", action="store_true",
                      help="Run all steps (1-8)")
    parser.add_argument("--steps", nargs="+", type=int,
                      choices=range(1, 9),
                      help="Specific steps to run (1-8)")
    
    # Data preparation options
    parser.add_argument("--use_synthetic", action="store_true",
                      help="Use synthetic data (for testing)")
    parser.add_argument("--num_samples", type=int, default=200,
                      help="Number of test samples")
    
    # Inference options
    parser.add_argument("--model", type=str, 
                      choices=["qwen1.5B", "qwen7B", "qwen14B", "all"],
                      default="all",
                      help="Which model(s) to run inference for")
    parser.add_argument("--max_samples", type=int,
                      help="Maximum samples to process (for testing)")
    
    # Projection options
    parser.add_argument("--no_multiple_layers", action="store_true",
                      help="Don't try multiple layers for projection")
    
    # Injection options
    parser.add_argument("--max_injection_samples", type=int, default=50,
                      help="Maximum B-ball samples for injection")
    
    # Visualization options
    parser.add_argument("--num_vis_cases", type=int, default=10,
                      help="Number of cases to visualize")
    parser.add_argument("--skip_individual_plots", action="store_true",
                      help="Skip individual case plots")
    
    # Continue on error
    parser.add_argument("--continue_on_error", action="store_true",
                      help="Continue even if a step fails")
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if args.all:
        steps_to_run = list(range(1, 9))
    elif args.steps:
        steps_to_run = sorted(args.steps)
    else:
        parser.print_help()
        print("\nError: Must specify either --all or --steps")
        sys.exit(1)
    
    print_banner("Heterogeneous Model Alignment Experiment")
    print(f"Steps to run: {steps_to_run}")
    print(f"Continue on error: {args.continue_on_error}")
    
    # Track results
    results = {}
    start_time = time.time()
    
    # Run selected steps
    for step in steps_to_run:
        success = False
        
        if step == 1:
            success = run_step1_data_prep(
                use_synthetic=args.use_synthetic,
                num_samples=args.num_samples
            )
        elif step == 2:
            success = run_step2_inference(
                model=args.model,
                max_samples=args.max_samples
            )
        elif step == 3:
            success = run_step3_error_analysis()
        elif step == 4:
            success = run_step4_cka_analysis(
                num_samples=args.num_samples
            )
        elif step == 5:
            success = run_step5_projection(
                try_multiple_layers=not args.no_multiple_layers
            )
        elif step == 6:
            success = run_step6_injection(
                max_samples=args.max_injection_samples
            )
        elif step == 7:
            success = run_step7_visualization(
                num_cases=args.num_vis_cases,
                skip_individual=args.skip_individual_plots
            )
        elif step == 8:
            success = run_step8_summary()
        
        results[step] = success
        
        if not success and not args.continue_on_error:
            print_banner("Experiment Terminated Due to Error")
            break
    
    # Print final summary
    total_time = time.time() - start_time
    
    print_banner("Experiment Complete")
    print("Results Summary:")
    print("-" * 80)
    
    for step, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        step_names = {
            1: "Data Preparation",
            2: "Model Inference",
            3: "Error Analysis",
            4: "CKA Analysis",
            5: "Projection Training",
            6: "Injection Experiments",
            7: "Visualization",
            8: "Summary Report"
        }
        print(f"Step {step} ({step_names[step]}): {status}")
    
    print("-" * 80)
    print(f"Total time: {total_time/60:.1f} minutes")
    
    # Overall status
    all_success = all(results.values())
    if all_success:
        print("\n✓ All steps completed successfully!")
        print("\nView results at:")
        print("  - experiment_results/summary_report.json")
        print("  - experiment_results/analysis/")
    else:
        print("\n✗ Some steps failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
