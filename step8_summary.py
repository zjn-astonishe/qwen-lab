"""
Step 8: Summary Report
Generate a comprehensive summary report of all experimental results
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from typing import Dict, Any

from config import (
    MODELS, DATA_CONFIG, ANALYSIS_CONFIG, INJECTION_CONFIG, 
    OUTPUT_PATHS, GENERATION_CONFIG
)


def load_error_analysis() -> Dict[str, Any]:
    """Load and summarize error analysis results"""
    error_path = ANALYSIS_CONFIG["error_analysis_output"]
    
    if not os.path.exists(error_path):
        return {"status": "not_found"}
    
    df = pd.read_csv(error_path)
    
    summary = {
        "total_samples": len(df),
        "samples_with_errors": int(df['has_error'].sum()),
        "error_rate": float(df['has_error'].mean()),
        "b_ball_dilemma_count": int(df['is_b_ball_dilemma'].sum()),
        "b_ball_dilemma_rate": 0.0
    }
    
    if summary["samples_with_errors"] > 0:
        summary["b_ball_dilemma_rate"] = float(
            summary["b_ball_dilemma_count"] / summary["samples_with_errors"]
        )
    
    # Error type breakdown
    error_types = df[df['has_error']]['error_type'].value_counts().to_dict()
    summary["error_types"] = {str(k): int(v) for k, v in error_types.items()}
    
    return summary


def load_cka_analysis() -> Dict[str, Any]:
    """Load and summarize CKA analysis results"""
    summary = {}
    
    pairs = [
        ("1.5B_vs_7B", OUTPUT_PATHS["cka_matrix_1.5B_vs_7B"]),
        ("7B_vs_14B", OUTPUT_PATHS["cka_matrix_7B_vs_14B"]),
        ("1.5B_vs_14B", OUTPUT_PATHS["cka_matrix_1.5B_vs_14B"])
    ]
    
    for pair_name, path in pairs:
        if os.path.exists(path):
            cka_matrix = np.load(path)
            
            # Calculate statistics
            n_rows, n_cols = cka_matrix.shape
            min_dim = min(n_rows, n_cols)
            diagonal = np.array([cka_matrix[i, i] for i in range(min_dim)])
            
            summary[pair_name] = {
                "matrix_shape": cka_matrix.shape,
                "mean_cka": float(np.mean(cka_matrix)),
                "max_cka": float(np.max(cka_matrix)),
                "min_cka": float(np.min(cka_matrix)),
                "mean_diagonal_cka": float(np.mean(diagonal)),
                "std_diagonal_cka": float(np.std(diagonal))
            }
            
            # Find best aligned layer pair
            max_idx = np.unravel_index(np.argmax(cka_matrix), cka_matrix.shape)
            summary[pair_name]["best_layer_pair"] = {
                "layer1": int(max_idx[0]),
                "layer2": int(max_idx[1]),
                "cka_score": float(cka_matrix[max_idx])
            }
        else:
            summary[pair_name] = {"status": "not_found"}
    
    return summary


def load_projection_results() -> Dict[str, Any]:
    """Load projection matrix training results"""
    layer_comparison_path = os.path.join(INJECTION_CONFIG["projection_dir"], "layer_comparison.json")
    
    if os.path.exists(layer_comparison_path):
        with open(layer_comparison_path, 'r') as f:
            layer_results = json.load(f)
        
        # Find best layer
        best_layer = None
        best_cosine = -1
        
        for layer, metrics in layer_results.items():
            if metrics["test_cosine"] > best_cosine:
                best_cosine = metrics["test_cosine"]
                best_layer = layer
        
        return {
            "status": "success",
            "best_layer": best_layer,
            "best_test_cosine": best_cosine,
            "all_layers": layer_results
        }
    else:
        # Check if projection matrix exists
        if os.path.exists(OUTPUT_PATHS["projection_matrix"]):
            return {
                "status": "matrix_exists",
                "message": "Projection matrix found but layer comparison not available"
            }
        else:
            return {"status": "not_found"}


def load_injection_results() -> Dict[str, Any]:
    """Load and summarize injection experiment results"""
    injection_path = INJECTION_CONFIG["results_output"]
    
    if not os.path.exists(injection_path):
        return {"status": "not_found"}
    
    df = pd.read_csv(injection_path)
    
    # Filter valid results
    valid_df = df[df['gt_rank_after_injection'] > 0]
    
    if len(valid_df) == 0:
        return {
            "status": "no_valid_results",
            "total_experiments": len(df)
        }
    
    summary = {
        "total_experiments": len(df),
        "valid_experiments": len(valid_df),
        "mean_gt_rank": float(valid_df['gt_rank_after_injection'].mean()),
        "median_gt_rank": float(valid_df['gt_rank_after_injection'].median()),
        "min_gt_rank": int(valid_df['gt_rank_after_injection'].min()),
        "max_gt_rank": int(valid_df['gt_rank_after_injection'].max())
    }
    
    # Best configuration
    best_config = valid_df.groupby(['injection_layer', 'alpha'])['gt_rank_after_injection'].mean()
    best_config = best_config.sort_values()
    
    if len(best_config) > 0:
        best_params = best_config.index[0]
        summary["best_configuration"] = {
            "injection_layer": int(best_params[0]),
            "alpha": float(best_params[1]),
            "avg_gt_rank": float(best_config.iloc[0])
        }
    
    # Performance by alpha
    alpha_performance = valid_df.groupby('alpha')['gt_rank_after_injection'].agg(['mean', 'std']).to_dict()
    summary["performance_by_alpha"] = {
        str(alpha): {
            "mean_rank": float(alpha_performance['mean'][alpha]),
            "std_rank": float(alpha_performance['std'][alpha])
        }
        for alpha in alpha_performance['mean'].keys()
    }
    
    # Performance by layer
    layer_performance = valid_df.groupby('injection_layer')['gt_rank_after_injection'].agg(['mean', 'std']).to_dict()
    summary["performance_by_layer"] = {
        str(layer): {
            "mean_rank": float(layer_performance['mean'][layer]),
            "std_rank": float(layer_performance['std'][layer])
        }
        for layer in layer_performance['mean'].keys()
    }
    
    return summary


def create_summary_report() -> Dict[str, Any]:
    """Create comprehensive summary report"""
    print("Generating summary report...")
    
    report = {
        "experiment_info": {
            "experiment_name": "Heterogeneous Model Hidden State Alignment Experiment",
            "models": {
                "small": MODELS["qwen1.5B"]["model_name"],
                "medium": MODELS["qwen7B"]["model_name"],
                "large": MODELS["qwen14B"]["model_name"]
            },
            "dataset": DATA_CONFIG["dataset_name"],
            "num_test_samples": DATA_CONFIG["num_samples"],
            "num_alignment_samples": DATA_CONFIG["num_alignment_samples"]
        },
        "error_analysis": load_error_analysis(),
        "cka_analysis": load_cka_analysis(),
        "projection_training": load_projection_results(),
        "injection_experiments": load_injection_results()
    }
    
    return report


def print_summary_report(report: Dict[str, Any]):
    """Print formatted summary report to console"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY REPORT")
    print("="*80)
    
    # Experiment Info
    print("\n## EXPERIMENT CONFIGURATION ##")
    print(f"Small Model: {report['experiment_info']['models']['small']}")
    print(f"Medium Model: {report['experiment_info']['models']['medium']}")
    print(f"Large Model: {report['experiment_info']['models']['large']}")
    print(f"Dataset: {report['experiment_info']['dataset']}")
    print(f"Test Samples: {report['experiment_info']['num_test_samples']}")
    print(f"Alignment Samples: {report['experiment_info']['num_alignment_samples']}")
    
    # Error Analysis
    print("\n## ERROR ANALYSIS (Small Model) ##")
    error_data = report['error_analysis']
    if error_data.get('status') != 'not_found':
        print(f"Total Samples Analyzed: {error_data['total_samples']}")
        print(f"Samples with Errors: {error_data['samples_with_errors']} ({error_data['error_rate']*100:.1f}%)")
        print(f"B-ball Dilemma Cases: {error_data['b_ball_dilemma_count']} ({error_data['b_ball_dilemma_rate']*100:.1f}% of errors)")
        
        if error_data.get('error_types'):
            print("\nError Type Distribution:")
            for error_type, count in error_data['error_types'].items():
                print(f"  - {error_type}: {count}")
    else:
        print("Error analysis results not found")
    
    # CKA Analysis
    print("\n## CKA SIMILARITY ANALYSIS ##")
    cka_data = report['cka_analysis']
    for pair_name, pair_data in cka_data.items():
        if pair_data.get('status') != 'not_found':
            print(f"\n{pair_name}:")
            print(f"  Mean CKA: {pair_data['mean_cka']:.4f}")
            print(f"  Max CKA: {pair_data['max_cka']:.4f}")
            print(f"  Mean Diagonal CKA: {pair_data['mean_diagonal_cka']:.4f}")
            
            best_pair = pair_data.get('best_layer_pair', {})
            if best_pair:
                print(f"  Best Aligned Layers: {best_pair['layer1']} <-> {best_pair['layer2']} (CKA={best_pair['cka_score']:.4f})")
    
    # Projection Training
    print("\n## PROJECTION MATRIX TRAINING ##")
    proj_data = report['projection_training']
    if proj_data.get('status') == 'success':
        print(f"Best Layer: {proj_data['best_layer']}")
        print(f"Best Test Cosine Similarity: {proj_data['best_test_cosine']:.4f}")
    else:
        print(f"Status: {proj_data.get('status', 'unknown')}")
    
    # Injection Experiments
    print("\n## INJECTION EXPERIMENTS ##")
    inj_data = report['injection_experiments']
    if inj_data.get('status') != 'not_found' and inj_data.get('valid_experiments', 0) > 0:
        print(f"Total Experiments: {inj_data['total_experiments']}")
        print(f"Valid Experiments: {inj_data['valid_experiments']}")
        print(f"Mean GT Token Rank: {inj_data['mean_gt_rank']:.2f}")
        print(f"Median GT Token Rank: {inj_data['median_gt_rank']:.1f}")
        print(f"Best GT Token Rank: {inj_data['min_gt_rank']}")
        
        if 'best_configuration' in inj_data:
            best_config = inj_data['best_configuration']
            print(f"\nBest Configuration:")
            print(f"  Injection Layer: {best_config['injection_layer']}")
            print(f"  Alpha: {best_config['alpha']}")
            print(f"  Average GT Rank: {best_config['avg_gt_rank']:.2f}")
    else:
        print(f"Status: {inj_data.get('status', 'not_found')}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Generate summary report")
    parser.add_argument("--output", type=str, default=OUTPUT_PATHS["summary_report"],
                      help="Path to save summary report JSON")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 8: Summary Report Generation")
    print("="*80)
    
    # Create summary report
    report = create_summary_report()
    
    # Save to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary report saved to: {args.output}")
    
    # Print to console
    print_summary_report(report)
    
    print("\n" + "="*80)
    print("Summary Report Generation Complete")
    print("="*80)


if __name__ == "__main__":
    main()
