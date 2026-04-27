"""
Step 7: Visualization
Generate visualizations for probability distributions and injection effects
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

from config import MODELS, ANALYSIS_CONFIG, INJECTION_CONFIG


def load_tokenizer(model_key: str):
    """Load tokenizer for decoding tokens"""
    model_name = MODELS[model_key]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


def plot_probability_distribution_comparison(
    sample_idx: int,
    small_probs: torch.Tensor,
    large_probs: torch.Tensor,
    tokenizer,
    top_k: int = 30,
    save_path: str = None
):
    """
    Plot side-by-side comparison of probability distributions from different models
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get top-k tokens and their probabilities
    small_topk = torch.topk(small_probs, k=top_k)
    large_topk = torch.topk(large_probs, k=top_k)
    
    # Decode tokens
    small_tokens = [tokenizer.decode([idx.item()]) for idx in small_topk.indices]
    large_tokens = [tokenizer.decode([idx.item()]) for idx in large_topk.indices]
    
    small_probs_values = small_topk.values.numpy()
    large_probs_values = large_topk.values.numpy()
    
    # Plot small model
    axes[0].barh(range(top_k), small_probs_values[::-1], color='coral')
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(small_tokens[::-1], fontsize=8)
    axes[0].set_xlabel('Probability', fontsize=12)
    axes[0].set_title('Small Model (1.5B) - Top 30 Tokens', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot large model
    axes[1].barh(range(top_k), large_probs_values[::-1], color='skyblue')
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels(large_tokens[::-1], fontsize=8)
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title('Large Model (7B) - Top 30 Tokens', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Probability Distribution Comparison - Sample {sample_idx}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_injection_effect(
    sample_idx: int,
    original_probs: torch.Tensor,
    injected_probs: torch.Tensor,
    gt_token: int,
    tokenizer,
    alpha: float,
    injection_layer: int,
    top_k: int = 30,
    save_path: str = None
):
    """
    Plot probability distribution before and after injection
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get top-k tokens
    original_topk = torch.topk(original_probs, k=top_k)
    injected_topk = torch.topk(injected_probs, k=top_k)
    
    # Decode tokens
    original_tokens = [tokenizer.decode([idx.item()]) for idx in original_topk.indices]
    injected_tokens = [tokenizer.decode([idx.item()]) for idx in injected_topk.indices]
    
    original_probs_values = original_topk.values.numpy()
    injected_probs_values = injected_topk.values.numpy()
    
    # Highlight ground truth token if in top-k
    gt_token_str = tokenizer.decode([gt_token]) if gt_token >= 0 else None
    
    # Plot original
    colors_orig = ['red' if tokenizer.decode([idx.item()]) == gt_token_str else 'skyblue' 
                   for idx in original_topk.indices]
    axes[0].barh(range(top_k), original_probs_values[::-1], color=colors_orig[::-1])
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(original_tokens[::-1], fontsize=8)
    axes[0].set_xlabel('Probability', fontsize=12)
    axes[0].set_title('Before Injection (7B Original)', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot injected
    colors_inj = ['red' if tokenizer.decode([idx.item()]) == gt_token_str else 'coral' 
                  for idx in injected_topk.indices]
    axes[1].barh(range(top_k), injected_probs_values[::-1], color=colors_inj[::-1])
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels(injected_tokens[::-1], fontsize=8)
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title(f'After Injection (α={alpha}, layer={injection_layer})', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    
    if gt_token_str:
        plt.suptitle(f'Injection Effect - Sample {sample_idx} (GT: {gt_token_str})', 
                    fontsize=16, y=1.02)
    else:
        plt.suptitle(f'Injection Effect - Sample {sample_idx}', fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_b_ball_cases(
    error_analysis_path: str,
    num_cases: int = 10
):
    """
    Visualize typical B-ball dilemma cases
    """
    print("Visualizing B-ball dilemma cases...")
    
    # Load error analysis
    df = pd.read_csv(error_analysis_path)
    b_ball_samples = df[df['is_b_ball_dilemma'] == True].head(num_cases)
    
    tokenizer = load_tokenizer("qwen1.5B")
    
    small_output_dir = MODELS["qwen1.5B"]["output_dir"]
    large_output_dir = MODELS["qwen7B"]["output_dir"]
    plot_dir = ANALYSIS_CONFIG["prob_plots_dir"]
    
    os.makedirs(plot_dir, exist_ok=True)
    
    for idx, row in b_ball_samples.iterrows():
        sample_idx = int(row['sample_idx'])
        
        # Load outputs
        small_path = os.path.join(small_output_dir, f"sample_{sample_idx:03d}.pt")
        large_path = os.path.join(large_output_dir, f"sample_{sample_idx:03d}.pt")
        
        if not os.path.exists(small_path) or not os.path.exists(large_path):
            continue
        
        small_output = torch.load(small_path, map_location="cpu")
        large_output = torch.load(large_path, map_location="cpu")
        
        small_probs = small_output.get("probs_per_step", [])
        large_probs = large_output.get("probs_per_step", [])
        
        if not small_probs or not large_probs:
            continue
        
        # Plot comparison
        save_path = os.path.join(plot_dir, f"case_{sample_idx:03d}_comparison.png")
        plot_probability_distribution_comparison(
            sample_idx,
            small_probs[0],
            large_probs[0],
            tokenizer,
            top_k=30,
            save_path=save_path
        )
        
        print(f"Saved plot for sample {sample_idx}")


def visualize_injection_results(
    injection_results_path: str,
    num_cases: int = 10
):
    """
    Visualize injection experiment results
    """
    print("Visualizing injection results...")
    
    if not os.path.exists(injection_results_path):
        print(f"Injection results not found at {injection_results_path}")
        return
    
    df = pd.read_csv(injection_results_path)
    
    # Select best performing injections (lowest GT rank)
    valid_results = df[df['gt_rank_after_injection'] > 0]
    if len(valid_results) == 0:
        print("No valid injection results to visualize")
        return
    
    best_results = valid_results.nsmallest(num_cases, 'gt_rank_after_injection')
    
    tokenizer = load_tokenizer("qwen7B")
    large_output_dir = MODELS["qwen7B"]["output_dir"]
    small_output_dir = MODELS["qwen1.5B"]["output_dir"]
    plot_dir = ANALYSIS_CONFIG["prob_plots_dir"]
    
    for idx, row in best_results.iterrows():
        sample_idx = int(row['sample_idx'])
        alpha = float(row['alpha'])
        injection_layer = int(row['injection_layer'])
        gt_token = int(row['gt_token'])
        
        # Load original large model output
        large_path = os.path.join(large_output_dir, f"sample_{sample_idx:03d}.pt")
        if not os.path.exists(large_path):
            continue
        
        large_output = torch.load(large_path, map_location="cpu")
        original_probs = large_output.get("probs_per_step", [])
        
        if not original_probs:
            continue
        
        # For injected probs, we would need to re-run or load from saved results
        # For now, skip this visualization or implement saving injected probs in step6
        print(f"Sample {sample_idx}: GT rank improved to {row['gt_rank_after_injection']}")


def create_summary_plots(
    error_analysis_path: str,
    injection_results_path: str,
    cka_output_dir: str
):
    """
    Create summary visualizations
    """
    print("Creating summary plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Error type distribution
    if os.path.exists(error_analysis_path):
        df_errors = pd.read_csv(error_analysis_path)
        error_counts = df_errors['error_type'].value_counts()
        
        axes[0, 0].bar(error_counts.index, error_counts.values, color='coral')
        axes[0, 0].set_xlabel('Error Type', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Error Type Distribution (1.5B Model)', fontsize=14)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add B-ball ratio
        b_ball_count = df_errors['is_b_ball_dilemma'].sum()
        total_errors = df_errors['has_error'].sum()
        if total_errors > 0:
            b_ball_ratio = b_ball_count / total_errors * 100
            axes[0, 0].text(0.5, 0.95, f'B-ball Dilemma: {b_ball_count}/{total_errors} ({b_ball_ratio:.1f}%)',
                          transform=axes[0, 0].transAxes, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Injection effectiveness by alpha
    if os.path.exists(injection_results_path):
        df_injection = pd.read_csv(injection_results_path)
        valid_inj = df_injection[df_injection['gt_rank_after_injection'] > 0]
        
        if len(valid_inj) > 0:
            alpha_performance = valid_inj.groupby('alpha')['gt_rank_after_injection'].mean()
            
            axes[0, 1].plot(alpha_performance.index, alpha_performance.values, 
                          marker='o', linewidth=2, markersize=8, color='steelblue')
            axes[0, 1].set_xlabel('Alpha (Mixing Coefficient)', fontsize=12)
            axes[0, 1].set_ylabel('Average GT Token Rank', fontsize=12)
            axes[0, 1].set_title('Injection Effectiveness vs Alpha', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].invert_yaxis()  # Lower rank is better
    
    # Plot 3: Injection effectiveness by layer
    if os.path.exists(injection_results_path):
        df_injection = pd.read_csv(injection_results_path)
        valid_inj = df_injection[df_injection['gt_rank_after_injection'] > 0]
        
        if len(valid_inj) > 0:
            layer_performance = valid_inj.groupby('injection_layer')['gt_rank_after_injection'].mean()
            
            axes[1, 0].bar(layer_performance.index.astype(str), layer_performance.values, 
                         color='mediumseagreen')
            axes[1, 0].set_xlabel('Injection Layer', fontsize=12)
            axes[1, 0].set_ylabel('Average GT Token Rank', fontsize=12)
            axes[1, 0].set_title('Injection Effectiveness by Layer', fontsize=14)
            axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Entropy distribution comparison
    if os.path.exists(error_analysis_path):
        df_errors = pd.read_csv(error_analysis_path)
        
        # Extract entropy values from details column (if stored as dict string)
        b_ball_samples = df_errors[df_errors['is_b_ball_dilemma'] == True]
        non_b_ball_errors = df_errors[(df_errors['has_error'] == True) & (df_errors['is_b_ball_dilemma'] == False)]
        
        if len(b_ball_samples) > 0 and len(non_b_ball_errors) > 0:
            axes[1, 1].hist([len(b_ball_samples), len(non_b_ball_errors)], 
                          bins=2, color=['orange', 'gray'], label=['B-ball', 'Other Errors'])
            axes[1, 1].set_xlabel('Error Category', fontsize=12)
            axes[1, 1].set_ylabel('Count', fontsize=12)
            axes[1, 1].set_title('B-ball vs Other Errors', fontsize=14)
            axes[1, 1].legend()
    
    plt.tight_layout()
    summary_path = os.path.join(cka_output_dir, "experiment_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Create visualizations")
    parser.add_argument("--num_cases", type=int, default=10,
                      help="Number of cases to visualize")
    parser.add_argument("--skip_individual", action="store_true",
                      help="Skip individual case visualizations")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 7: Visualization")
    print("="*80)
    
    error_analysis_path = ANALYSIS_CONFIG["error_analysis_output"]
    injection_results_path = INJECTION_CONFIG["results_output"]
    cka_output_dir = ANALYSIS_CONFIG["cka_output_dir"]
    
    # Visualize B-ball cases
    if not args.skip_individual and os.path.exists(error_analysis_path):
        visualize_b_ball_cases(error_analysis_path, num_cases=args.num_cases)
    
    # Create summary plots
    create_summary_plots(error_analysis_path, injection_results_path, cka_output_dir)
    
    print("\n" + "="*80)
    print("Visualization Complete")
    print(f"Plots saved to: {ANALYSIS_CONFIG['prob_plots_dir']}")
    print("="*80)


if __name__ == "__main__":
    main()
