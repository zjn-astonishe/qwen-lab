"""
Step 9: Summary Report (V4 — renumbered)

Generate a comprehensive summary report of all experimental results.

V4 changes:
  - Renumbered from Step 8 to Step 9 (pipeline reordering)
  - Added load_probing_results() for probability probing data
  - Integrated probing summary into create_summary_report()
  - Added probing section to print_summary_report()

Previous optimizations:
  - Adapts to dynamic model-pair projection paths via config.get_projection_paths()
  - Adapts to dynamic injection result paths
  - Uses utils.safe_parse for robust dict parsing
  - Enhanced report with three-model comparison data
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import argparse
from typing import Dict, Any, Optional

from config import (
    MODELS, DATA_CONFIG, ANALYSIS_CONFIG, INJECTION_CONFIG, OUTPUT_PATHS,
    PROBING_CONFIG, get_projection_paths,
)


def load_error_analysis() -> Dict[str, Any]:
    """Load and summarize error analysis results."""
    error_path = ANALYSIS_CONFIG["error_analysis_output"]

    if not os.path.exists(error_path):
        return {"status": "not_found"}

    df = pd.read_csv(error_path)

    summary = {
        "total_samples": int(len(df)),
        "samples_with_errors": int(df['has_error'].sum()),
        "error_rate": float(df['has_error'].mean()),
        "b_ball_dilemma_count": int(df['is_b_ball_dilemma'].sum()),
        "b_ball_dilemma_rate": 0.0,
    }

    if summary["samples_with_errors"] > 0:
        summary["b_ball_dilemma_rate"] = float(
            summary["b_ball_dilemma_count"] / summary["samples_with_errors"]
        )

    # Error type breakdown
    evaluable = df[df['has_error']]
    if len(evaluable) > 0:
        error_types = evaluable['error_type'].value_counts().to_dict()
        summary["error_types"] = {str(k): int(v) for k, v in error_types.items()}

    return summary


def load_three_model_comparison() -> Dict[str, Any]:
    """Load three-model comparison results if available."""
    path = ANALYSIS_CONFIG.get("three_model_comparison", "")
    if not path or not os.path.exists(path):
        return {"status": "not_found"}

    df = pd.read_csv(path)
    n = len(df)
    if n == 0:
        return {"status": "empty"}

    result = {
        "total_evaluable": n,
        "accuracy": {},
    }

    for m in ["1.5B", "3B", "7B"]:
        col = f"correct_{m}"
        if col in df.columns:
            result["accuracy"][m] = {
                "correct": int(df[col].sum()),
                "total": n,
                "rate": float(df[col].mean()),
            }

    # Injection targets per small model
    result["injection_targets"] = {}
    for small_m in ["1.5B", "3B"]:
        small_col = f"correct_{small_m}"
        if small_col in df.columns and "correct_7B" in df.columns:
            targets = df[(df["correct_7B"]) & (~df[small_col])]
            result["injection_targets"][f"{small_m}_vs_7B"] = int(len(targets))

    return result


def load_cka_analysis() -> Dict[str, Any]:
    """Load and summarize CKA analysis results."""
    summary = {}

    pairs = [
        ("1.5B_vs_7B", OUTPUT_PATHS["cka_matrix_1.5B_vs_7B"]),
        ("7B_vs_3B", OUTPUT_PATHS["cka_matrix_7B_vs_3B"]),
        ("1.5B_vs_3B", OUTPUT_PATHS["cka_matrix_1.5B_vs_3B"]),
    ]

    for pair_name, path in pairs:
        if not os.path.exists(path):
            summary[pair_name] = {"status": "not_found"}
            continue

        cka_matrix = np.load(path)
        n_rows, n_cols = cka_matrix.shape
        min_dim = min(n_rows, n_cols)
        diagonal = np.array([cka_matrix[i, i] for i in range(min_dim)])

        max_idx = np.unravel_index(np.argmax(cka_matrix), cka_matrix.shape)

        summary[pair_name] = {
            "matrix_shape": list(cka_matrix.shape),
            "mean_cka": float(np.mean(cka_matrix)),
            "max_cka": float(np.max(cka_matrix)),
            "min_cka": float(np.min(cka_matrix)),
            "mean_diagonal_cka": float(np.mean(diagonal)),
            "std_diagonal_cka": float(np.std(diagonal)),
            "best_layer_pair": {
                "layer1": int(max_idx[0]),
                "layer2": int(max_idx[1]),
                "cka_score": float(cka_matrix[max_idx]),
            },
        }

    return summary


def load_projection_results(small_model: str = "qwen1.5B",
                           large_model: str = "qwen7B") -> Dict[str, Any]:
    """Load projection matrix training results."""
    paths = get_projection_paths(small_model, large_model)
    layer_comparison_path = paths["layer_comparison"]

    if os.path.exists(layer_comparison_path):
        with open(layer_comparison_path, 'r') as f:
            layer_results = json.load(f)

        best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]["test_cosine"])

        return {
            "status": "success",
            "model_pair": f"{small_model} -> {large_model}",
            "best_layer": best_layer,
            "best_test_cosine": layer_results[best_layer]["test_cosine"],
            "all_layers": layer_results,
        }
    else:
        if os.path.exists(paths["projection_matrix"]):
            return {"status": "matrix_exists", "model_pair": f"{small_model} -> {large_model}"}
        return {"status": "not_found"}


def load_injection_results(small_model: str = "qwen3B",
                           large_model: str = "qwen7B") -> Dict[str, Any]:
    """Load and summarize injection experiment results."""
    # Try dynamic path first, fall back to default
    from config import get_injection_results_path
    injection_path = get_injection_results_path(small_model, large_model)

    if not os.path.exists(injection_path):
        injection_path = INJECTION_CONFIG["results_output"]

    if not os.path.exists(injection_path):
        return {"status": "not_found"}

    df = pd.read_csv(injection_path)
    valid_df = df[df['gt_rank_after_injection'] > 0]

    if len(valid_df) == 0:
        return {"status": "no_valid_results", "total_experiments": len(df)}

    summary: Dict[str, Any] = {
        "total_experiments": int(len(df)),
        "valid_experiments": int(len(valid_df)),
        "errors_corrected": int(df["error_corrected"].sum()),
        "correction_rate": float(df["error_corrected"].mean()),
        "mean_gt_rank": float(valid_df['gt_rank_after_injection'].mean()),
        "median_gt_rank": float(valid_df['gt_rank_after_injection'].median()),
        "min_gt_rank": int(valid_df['gt_rank_after_injection'].min()),
        "max_gt_rank": int(valid_df['gt_rank_after_injection'].max()),
    }

    # Best configuration
    best_config = valid_df.groupby(['injection_layer', 'alpha'])['gt_rank_after_injection'].mean()
    best_config = best_config.sort_values()
    if len(best_config) > 0:
        params = best_config.index[0]
        summary["best_configuration"] = {
            "injection_layer": int(params[0]),
            "alpha": float(params[1]),
            "avg_gt_rank": float(best_config.iloc[0]),
        }

    # Performance by alpha
    alpha_perf = valid_df.groupby('alpha')['gt_rank_after_injection'].agg(['mean', 'std'])
    summary["performance_by_alpha"] = {
        str(a): {"mean_rank": float(row['mean']), "std_rank": float(row['std'])}
        for a, row in alpha_perf.iterrows()
    }

    # Performance by layer
    layer_perf = valid_df.groupby('injection_layer')['gt_rank_after_injection'].agg(['mean', 'std'])
    summary["performance_by_layer"] = {
        str(l): {"mean_rank": float(row['mean']), "std_rank": float(row['std'])}
        for l, row in layer_perf.iterrows()
    }

    return summary


def load_probing_results() -> Dict[str, Any]:
    """Load and summarize probability probing results from Step 9."""
    probing_dir = PROBING_CONFIG.get("output_dir", "experiment_results/probing")
    json_path = os.path.join(probing_dir, "probing_results.json")

    if not os.path.exists(json_path):
        return {"status": "not_found"}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if not results:
        return {"status": "empty"}

    n = len(results)
    summary: Dict[str, Any] = {
        "status": "success",
        "total_samples_probed": n,
        "final_layer_stats": {},
        "divergence": {},
        "early_exit": {},
    }

    # Final layer stats per model
    for tag in ["1.5B", "3B", "7B"]:
        key_gt_prob = f"{tag}_final_gt_prob"
        key_gt_rank = f"{tag}_final_gt_rank"

        gt_probs = [r.get(key_gt_prob, 0) for r in results if r.get(key_gt_prob) is not None]
        gt_ranks = [r.get(key_gt_rank, 999) for r in results if r.get(key_gt_rank) is not None]

        if gt_probs:
            summary["final_layer_stats"][tag] = {
                "mean_gt_prob": float(np.mean(gt_probs)),
                "median_gt_prob": float(np.median(gt_probs)),
                "mean_gt_rank": float(np.mean(gt_ranks)),
                "median_gt_rank": float(np.median(gt_ranks)),
                "samples": len(gt_probs),
            }

    # Early exit switch point analysis (1.5B -> 7B)
    switch_progress = []
    switch_gaps = []
    for r in results:
        # Approximate: find the progress where 7B P(GT) - 1.5B P(GT) is maximized
        layers_15 = r.get("1.5B_layers", [])
        layers_7b = r.get("7B_layers", [])
        if not layers_15 or not layers_7b:
            continue

        best_gap = -1
        best_pct = 0
        for pct in range(0, 101, 5):
            s = min(layers_15, key=lambda l: abs(l["progress"] - pct))
            l_ = min(layers_7b, key=lambda l: abs(l["progress"] - pct))
            if abs(s["progress"] - pct) > 10 or abs(l_["progress"] - pct) > 10:
                continue
            gap = l_["gt_prob"] - s["gt_prob"]
            if gap > best_gap:
                best_gap = gap
                best_pct = pct

        if best_gap > 0:
            switch_progress.append(best_pct)
            switch_gaps.append(best_gap)

    if switch_progress:
        summary["early_exit"] = {
            "optimal_switch_progress": {
                "mean": float(np.mean(switch_progress)),
                "median": float(np.median(switch_progress)),
            },
            "mean_gt_prob_gap_at_switch": float(np.mean(switch_gaps)),
            "samples_with_divergence": len(switch_progress),
        }

    return summary


def create_summary_report() -> Dict[str, Any]:
    """Create comprehensive summary report."""
    print("Generating summary report...")

    report = {
        "experiment_info": {
            "experiment_name": "Heterogeneous Model Hidden State Alignment Experiment",
            "models": {
                "small": MODELS["qwen1.5B"]["model_name"],
                "medium": MODELS["qwen3B"]["model_name"],
                "large": MODELS["qwen7B"]["model_name"],
            },
            "dataset": DATA_CONFIG["dataset_name"],
            "total_test_samples": DATA_CONFIG["total_samples"],
            "num_alignment_samples": DATA_CONFIG["num_alignment_samples"],
        },
        "error_analysis": load_error_analysis(),
        "three_model_comparison": load_three_model_comparison(),
        "cka_analysis": load_cka_analysis(),
        "projection_training": load_projection_results(),
        "injection_experiments": load_injection_results(),
        "probing": load_probing_results(),
    }

    return report


def print_summary_report(report: Dict[str, Any]):
    """Print formatted summary report to console."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY REPORT")
    print("=" * 80)

    info = report['experiment_info']
    print(f"\n## EXPERIMENT CONFIGURATION ##")
    print(f"  Small:  {info['models']['small']}")
    print(f"  Medium: {info['models']['medium']}")
    print(f"  Large:  {info['models']['large']}")
    print(f"  Dataset: {info['dataset']}")
    print(f"  Test samples: {info['total_test_samples']}, Alignment: {info['num_alignment_samples']}")

    # Error analysis
    err = report.get('error_analysis', {})
    print(f"\n## ERROR ANALYSIS ##")
    if err.get('status') != 'not_found':
        print(f"  Total: {err['total_samples']}")
        print(f"  Errors: {err['samples_with_errors']} ({err['error_rate'] * 100:.1f}%)")
        print(f"  B-ball: {err['b_ball_dilemma_count']} ({err['b_ball_dilemma_rate'] * 100:.1f}% of errors)")
        for et, cnt in err.get('error_types', {}).items():
            print(f"    {et}: {cnt}")

    # Three-model comparison
    tmc = report.get('three_model_comparison', {})
    if tmc.get('status') not in ('not_found', 'empty'):
        print(f"\n## THREE-MODEL COMPARISON ##")
        print(f"  Evaluable samples: {tmc['total_evaluable']}")
        for m, acc in tmc.get('accuracy', {}).items():
            print(f"  {m}: {acc['correct']}/{acc['total']} ({acc['rate'] * 100:.1f}%)")
        for pair, count in tmc.get('injection_targets', {}).items():
            print(f"  Injection targets ({pair}): {count}")

    # CKA
    print(f"\n## CKA SIMILARITY ##")
    for pair, data in report.get('cka_analysis', {}).items():
        if data.get('status') != 'not_found':
            print(f"  {pair}: mean={data['mean_cka']:.4f}, "
                  f"diag={data['mean_diagonal_cka']:.4f}, "
                  f"best={data['best_layer_pair']['layer1']}<->{data['best_layer_pair']['layer2']} "
                  f"({data['best_layer_pair']['cka_score']:.4f})")

    # Projection
    proj = report.get('projection_training', {})
    print(f"\n## PROJECTION MATRIX ##")
    if proj.get('status') == 'success':
        print(f"  Model pair: {proj.get('model_pair', '?')}")
        print(f"  Best layer: {proj['best_layer']}, cosine sim: {proj['best_test_cosine']:.4f}")
    else:
        print(f"  Status: {proj.get('status', '?')}")

    # Injection
    inj = report.get('injection_experiments', {})
    print(f"\n## INJECTION EXPERIMENTS ##")
    if inj.get('status') not in ('not_found', 'no_valid_results'):
        print(f"  Total: {inj['total_experiments']}")
        print(f"  Corrected: {inj.get('errors_corrected', 0)} ({inj.get('correction_rate', 0) * 100:.1f}%)")
        print(f"  Mean GT rank: {inj['mean_gt_rank']:.2f}, median: {inj['median_gt_rank']:.1f}")
        if 'best_configuration' in inj:
            bc = inj['best_configuration']
            print(f"  Best: layer={bc['injection_layer']}, alpha={bc['alpha']}, "
                  f"avg_rank={bc['avg_gt_rank']:.2f}")
    else:
        print(f"  Status: {inj.get('status', '?')}")

    # Probing
    prob = report.get('probing', {})
    print(f"\n## PROBABILITY PROBING ##")
    if prob.get('status') == 'success':
        print(f"  Samples probed: {prob['total_samples_probed']}")
        for tag, stats in prob.get('final_layer_stats', {}).items():
            print(f"  {tag} final P(GT): mean={stats['mean_gt_prob']:.4f}, "
                  f"median={stats['median_gt_prob']:.4f}")
            print(f"  {tag} final GT rank: mean={stats['mean_gt_rank']:.1f}, "
                  f"median={stats['median_gt_rank']:.1f}")
        ee = prob.get('early_exit', {})
        if ee:
            sp = ee.get('optimal_switch_progress', {})
            print(f"  Early exit (1.5B->7B): switch at {sp.get('median', 0):.0f}% progress "
                  f"(mean {sp.get('mean', 0):.0f}%)")
            print(f"  Mean GT prob gap at switch: {ee.get('mean_gt_prob_gap_at_switch', 0):.4f}")
    else:
        print(f"  Status: {prob.get('status', '?')}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Step 9: Summary Report")
    parser.add_argument("--output", type=str, default=OUTPUT_PATHS["summary_report"])
    args = parser.parse_args()

    print("=" * 80)
    print("Step 9: Summary Report Generation")
    print("=" * 80)

    report = create_summary_report()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {args.output}")
    print_summary_report(report)
    print("=" * 80)


if __name__ == "__main__":
    main()