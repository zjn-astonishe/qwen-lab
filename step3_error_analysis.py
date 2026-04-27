"""
Step 3: Error Analysis
Analyze errors from the smallest model (1.5B) and identify "B-ball dilemma" cases
"""

import json
import os
import re
import torch
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from scipy.stats import entropy

from config import MODELS, ANALYSIS_CONFIG


def parse_tool_call_from_text(text: str) -> Dict[str, Any]:
    """
    Parse function call from generated text
    Supports formats like:
    - function_name(param1=value1, param2=value2)
    - {"function": "function_name", "arguments": {...}}
    """
    parsed = {
        "function_name": None,
        "arguments": {}
    }
    
    # Try JSON format first
    json_match = re.search(r'\{[^{}]*"function"[^{}]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            parsed["function_name"] = data.get("function")
            parsed["arguments"] = data.get("arguments", {})
            return parsed
        except:
            pass
    
    # Try function call format: function_name(...)
    func_match = re.search(r'(\w+)\s*\(([^)]*)\)', text)
    if func_match:
        parsed["function_name"] = func_match.group(1)
        args_str = func_match.group(2)
        
        # Parse arguments
        if args_str:
            arg_pairs = re.findall(r'(\w+)\s*=\s*([^,]+)', args_str)
            for key, value in arg_pairs:
                parsed["arguments"][key.strip()] = value.strip()
    
    return parsed


def compare_tool_calls(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, bool]:
    """
    Compare predicted and ground truth tool calls
    """
    result = {
        "function_match": False,
        "arguments_match": False,
        "has_error": True
    }
    
    pred_func = predicted.get("function_name", "").lower()
    gt_func = str(ground_truth.get("function", "")).lower()
    
    # Check function name match
    if pred_func and gt_func and pred_func == gt_func:
        result["function_match"] = True
    
    # Check arguments match (simplified)
    pred_args = predicted.get("arguments", {})
    gt_args = ground_truth.get("arguments", {})
    
    if pred_args and gt_args:
        # Simple check: do keys match?
        if set(pred_args.keys()) == set(gt_args.keys()):
            result["arguments_match"] = True
    
    # Overall error check
    if result["function_match"] and result["arguments_match"]:
        result["has_error"] = False
    
    return result


def calculate_top_k_overlap(probs1: torch.Tensor, probs2: torch.Tensor, k: int = 20) -> float:
    """
    Calculate Jaccard similarity between top-k tokens of two probability distributions
    """
    # Get top-k indices
    topk1 = torch.topk(probs1, k=k).indices.tolist()
    topk2 = torch.topk(probs2, k=k).indices.tolist()
    
    # Calculate Jaccard similarity
    set1 = set(topk1)
    set2 = set(topk2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_entropy(probs: torch.Tensor) -> float:
    """
    Calculate entropy of probability distribution
    """
    probs_np = probs.cpu().numpy()
    # Filter out zero probabilities to avoid log(0)
    probs_np = probs_np[probs_np > 0]
    return float(entropy(probs_np))


def identify_b_ball_dilemma(
    small_model_probs: torch.Tensor,
    large_model_probs: torch.Tensor,
    ground_truth_token: int,
    predicted_token: int,
    top_k: int = 20,
    entropy_threshold: float = 2.5
) -> Dict[str, Any]:
    """
    Identify if this error is a "B-ball dilemma"
    
    Criteria:
    1. Small model's predicted token != ground truth
    2. Ground truth token is in small model's top-k
    3. Small model's distribution has high entropy (平权分布)
    4. Small and large model have similar top-k overlap
    """
    result = {
        "is_b_ball_dilemma": False,
        "gt_in_topk": False,
        "gt_rank": -1,
        "entropy": 0.0,
        "topk_overlap": 0.0,
        "predicted_prob": 0.0,
        "gt_prob": 0.0
    }
    
    # Calculate entropy
    result["entropy"] = calculate_entropy(small_model_probs)
    
    # Check if ground truth is in top-k
    topk_indices = torch.topk(small_model_probs, k=top_k).indices.tolist()
    if ground_truth_token in topk_indices:
        result["gt_in_topk"] = True
        result["gt_rank"] = topk_indices.index(ground_truth_token) + 1
    else:
        # Find actual rank
        sorted_indices = torch.argsort(small_model_probs, descending=True).tolist()
        if ground_truth_token in sorted_indices:
            result["gt_rank"] = sorted_indices.index(ground_truth_token) + 1
    
    # Calculate top-k overlap with large model
    result["topk_overlap"] = calculate_top_k_overlap(small_model_probs, large_model_probs, k=top_k)
    
    # Get probabilities
    result["predicted_prob"] = float(small_model_probs[predicted_token])
    if ground_truth_token >= 0 and ground_truth_token < len(small_model_probs):
        result["gt_prob"] = float(small_model_probs[ground_truth_token])
    
    # Determine if it's a B-ball dilemma
    if (result["gt_in_topk"] and 
        result["entropy"] > entropy_threshold and
        result["topk_overlap"] > 0.3):  # Some overlap with larger model
        result["is_b_ball_dilemma"] = True
    
    return result


def analyze_sample_errors(
    sample_idx: int,
    small_model_output: Dict[str, Any],
    large_model_output: Dict[str, Any],
    tokenizer
) -> Dict[str, Any]:
    """
    Analyze errors for a single sample
    """
    analysis = {
        "sample_idx": sample_idx,
        "sample_id": small_model_output.get("sample_id", f"sample_{sample_idx}"),
        "has_error": False,
        "error_type": None,
        "is_b_ball_dilemma": False,
        "details": {}
    }
    
    # Parse tool calls
    small_generated = small_model_output.get("generated_text", "")
    ground_truth = small_model_output.get("ground_truth", {})
    
    predicted_call = parse_tool_call_from_text(small_generated)
    
    # Compare with ground truth
    comparison = compare_tool_calls(predicted_call, ground_truth)
    
    analysis["has_error"] = comparison["has_error"]
    
    if comparison["has_error"]:
        if not comparison["function_match"]:
            analysis["error_type"] = "function_name_error"
        elif not comparison["arguments_match"]:
            analysis["error_type"] = "argument_error"
        else:
            analysis["error_type"] = "other_error"
        
        # Analyze first generation step for B-ball dilemma
        if len(small_model_output.get("probs_per_step", [])) > 0:
            small_probs = small_model_output["probs_per_step"][0]
            
            # Get corresponding large model probs
            if len(large_model_output.get("probs_per_step", [])) > 0:
                large_probs = large_model_output["probs_per_step"][0]
                
                # Get predicted and ground truth tokens
                predicted_token = torch.argmax(small_probs).item()
                
                # Try to find ground truth token
                gt_func = str(ground_truth.get("function", ""))
                if gt_func:
                    gt_tokens = tokenizer.encode(gt_func, add_special_tokens=False)
                    if gt_tokens:
                        gt_token = gt_tokens[0]
                        
                        # Check for B-ball dilemma
                        b_ball_info = identify_b_ball_dilemma(
                            small_probs,
                            large_probs,
                            gt_token,
                            predicted_token,
                            top_k=ANALYSIS_CONFIG["top_k_overlap"],
                            entropy_threshold=ANALYSIS_CONFIG["entropy_threshold"]
                        )
                        
                        analysis["is_b_ball_dilemma"] = b_ball_info["is_b_ball_dilemma"]
                        analysis["details"] = b_ball_info
    
    return analysis


def load_model_outputs(model_key: str, num_samples: int) -> List[Dict[str, Any]]:
    """
    Load model outputs from saved files
    """
    output_dir = MODELS[model_key]["output_dir"]
    outputs = []
    
    for i in range(num_samples):
        output_path = os.path.join(output_dir, f"sample_{i:03d}.pt")
        if os.path.exists(output_path):
            try:
                output = torch.load(output_path, map_location="cpu")
                outputs.append(output)
            except Exception as e:
                print(f"Error loading {output_path}: {e}")
                outputs.append(None)
        else:
            outputs.append(None)
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Analyze model errors")
    parser.add_argument("--small_model", type=str, default="qwen1.5B",
                      help="Small model to analyze")
    parser.add_argument("--large_model", type=str, default="qwen7B",
                      help="Large model for comparison")
    parser.add_argument("--num_samples", type=int, default=200,
                      help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 3: Error Analysis")
    print("="*80)
    
    # Load tokenizer for the small model
    from transformers import AutoTokenizer
    model_name = MODELS[args.small_model]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model outputs
    print(f"Loading outputs for {args.small_model}...")
    small_outputs = load_model_outputs(args.small_model, args.num_samples)
    
    print(f"Loading outputs for {args.large_model}...")
    large_outputs = load_model_outputs(args.large_model, args.num_samples)
    
    # Analyze each sample
    print("Analyzing errors...")
    results = []
    
    for i in tqdm(range(args.num_samples)):
        if small_outputs[i] is None or large_outputs[i] is None:
            print(f"Skipping sample {i} (missing output)")
            continue
        
        analysis = analyze_sample_errors(
            i,
            small_outputs[i],
            large_outputs[i],
            tokenizer
        )
        results.append(analysis)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_path = ANALYSIS_CONFIG["error_analysis_output"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print("\n" + "="*80)
    print("Error Analysis Results")
    print("="*80)
    print(f"Total samples analyzed: {len(results)}")
    print(f"Samples with errors: {df['has_error'].sum()} ({df['has_error'].mean()*100:.1f}%)")
    
    if df['has_error'].sum() > 0:
        print("\nError type distribution:")
        print(df[df['has_error']]['error_type'].value_counts())
        
        print(f"\nB-ball dilemma cases: {df['is_b_ball_dilemma'].sum()} ({df['is_b_ball_dilemma'].sum()/df['has_error'].sum()*100:.1f}% of errors)")
    
    print(f"\nResults saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
