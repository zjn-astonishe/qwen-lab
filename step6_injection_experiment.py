"""
Step 6: Injection Experiment
Inject small model's hidden states into large model and evaluate the effect
"""

import os
import json
import torch
import pandas as pd
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODELS, INJECTION_CONFIG, ANALYSIS_CONFIG, HARDWARE_CONFIG, OUTPUT_PATHS


def load_projection_matrix() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load trained projection matrix and bias
    """
    W_path = OUTPUT_PATHS["projection_matrix"]
    b_path = OUTPUT_PATHS["projection_bias"]
    
    if not os.path.exists(W_path) or not os.path.exists(b_path):
        raise FileNotFoundError(f"Projection matrix not found. Please run step5 first.")
    
    W = torch.load(W_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")
    
    print(f"Loaded projection matrix: {W.shape}")
    print(f"Loaded bias vector: {b.shape}")
    
    return W, b


def load_b_ball_samples(error_analysis_path: str, max_samples: int = 50) -> List[int]:
    """
    Load sample indices that are marked as B-ball dilemma
    """
    df = pd.read_csv(error_analysis_path)
    
    # Filter B-ball dilemma samples
    b_ball_samples = df[df['is_b_ball_dilemma'] == True]['sample_idx'].tolist()
    
    print(f"Found {len(b_ball_samples)} B-ball dilemma samples")
    
    if len(b_ball_samples) > max_samples:
        b_ball_samples = b_ball_samples[:max_samples]
        print(f"Limited to {max_samples} samples for injection experiment")
    
    return b_ball_samples


def inject_hidden_state(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    injection_hidden: torch.Tensor,
    injection_layer: int,
    alpha: float = 0.5,
    max_new_tokens: int = 512
) -> Dict[str, any]:
    """
    Run model inference with hidden state injection at specified layer
    
    Args:
        model: Large model to inject into
        tokenizer: Tokenizer
        input_ids: Input token IDs
        injection_hidden: Hidden state to inject, shape [hidden_dim]
        injection_layer: Which layer to inject into (negative index from end)
        alpha: Mixing coefficient (0 = no injection, 1 = full injection)
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Dictionary with generated output and probability distributions
    """
    device = model.device
    input_ids = input_ids.to(device)
    injection_hidden = injection_hidden.to(device)
    
    # Create hook to inject hidden state
    injected = {"done": False}
    
    def injection_hook(module, input, output):
        if injected["done"]:
            return output
        
        # output is typically a tuple, first element is hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Inject at the last token position
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Mix current hidden state with injected hidden state
        current_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]
        mixed_hidden = (1 - alpha) * current_hidden + alpha * injection_hidden.unsqueeze(0)
        
        # Replace last token's hidden state
        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] = mixed_hidden
        
        injected["done"] = True
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
    
    # Register hook at target layer
    # Get the layer module (architecture-specific)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Unknown model architecture")
    
    target_layer = layers[injection_layer]
    handle = target_layer.register_forward_hook(injection_hook)
    
    try:
        # Generate with injection
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        # Extract results
        generated_ids = outputs.sequences[0]
        scores = outputs.scores
        
        # Process scores
        probs_per_step = []
        for score in scores:
            probs = torch.softmax(score[0], dim=-1)
            probs_per_step.append(probs.cpu())
        
        result = {
            "generated_ids": generated_ids.cpu(),
            "probs_per_step": probs_per_step,
            "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True)
        }
        
    finally:
        # Remove hook
        handle.remove()
    
    return result


def run_injection_experiment(
    sample_idx: int,
    small_model_output: Dict,
    W: torch.Tensor,
    b: torch.Tensor,
    model,
    tokenizer,
    alpha_values: List[float],
    injection_layers: List[int],
    step_idx: int = 0
) -> List[Dict]:
    """
    Run injection experiment for a single sample with multiple configurations
    """
    results = []
    
    # Get small model's hidden state to inject
    hidden_states = small_model_output.get("hidden_states_per_step", [])
    if not hidden_states or len(hidden_states) <= step_idx:
        print(f"Warning: No hidden states for sample {sample_idx}")
        return results
    
    # Get input
    input_ids = small_model_output["input_ids"]
    ground_truth = small_model_output.get("ground_truth", {})
    
    # Load original large model output for comparison
    large_output_path = os.path.join(MODELS["qwen7B"]["output_dir"], f"sample_{sample_idx:03d}.pt")
    if os.path.exists(large_output_path):
        original_large_output = torch.load(large_output_path, map_location="cpu")
        original_probs = original_large_output.get("probs_per_step", [])
    else:
        original_probs = None
    
    # Get small model's original probs
    small_probs = small_model_output.get("probs_per_step", [])
    
    # Try different injection configurations
    for injection_layer in injection_layers:
        # Convert to negative index
        layer_idx = -injection_layer
        
        # Get hidden state from that layer
        step_hidden = hidden_states[step_idx]
        if abs(layer_idx) >= len(step_hidden):
            continue
        
        small_hidden = step_hidden[layer_idx]  # [hidden_dim_small]
        
        # Project to large model space
        projected_hidden = small_hidden @ W.T + b  # [hidden_dim_large]
        
        for alpha in alpha_values:
            try:
                # Run injection
                injection_result = inject_hidden_state(
                    model,
                    tokenizer,
                    input_ids,
                    projected_hidden,
                    layer_idx,
                    alpha=alpha,
                    max_new_tokens=20  # Just need first few tokens
                )
                
                # Compare probability distributions at first step
                if len(injection_result["probs_per_step"]) > 0:
                    injected_probs = injection_result["probs_per_step"][0]
                    
                    # Get predicted tokens
                    small_pred = torch.argmax(small_probs[0]).item() if small_probs else -1
                    original_pred = torch.argmax(original_probs[0]).item() if original_probs else -1
                    injected_pred = torch.argmax(injected_probs).item()
                    
                    # Get ground truth token
                    gt_func = str(ground_truth.get("function", ""))
                    gt_token = -1
                    if gt_func:
                        gt_tokens = tokenizer.encode(gt_func, add_special_tokens=False)
                        if gt_tokens:
                            gt_token = gt_tokens[0]
                    
                    # Calculate ranks
                    sorted_indices = torch.argsort(injected_probs, descending=True)
                    gt_rank = -1
                    if gt_token >= 0:
                        gt_rank = (sorted_indices == gt_token).nonzero(as_tuple=True)[0].item() + 1
                    
                    # Record result
                    results.append({
                        "sample_idx": sample_idx,
                        "injection_layer": injection_layer,
                        "alpha": alpha,
                        "small_pred_token": small_pred,
                        "original_large_pred_token": original_pred,
                        "injected_pred_token": injected_pred,
                        "gt_token": gt_token,
                        "gt_rank_after_injection": gt_rank,
                        "injected_text": injection_result["generated_text"][:100]
                    })
                
            except Exception as e:
                print(f"Error in injection (sample={sample_idx}, layer={injection_layer}, alpha={alpha}): {e}")
                continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run injection experiment")
    parser.add_argument("--small_model", type=str, default="qwen1.5B",
                      help="Small model key")
    parser.add_argument("--large_model", type=str, default="qwen7B",
                      help="Large model key")
    parser.add_argument("--max_samples", type=int, default=50,
                      help="Maximum number of B-ball samples to test")
    parser.add_argument("--error_analysis", type=str, default=ANALYSIS_CONFIG["error_analysis_output"],
                      help="Path to error analysis CSV")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 6: Injection Experiment")
    print("="*80)
    
    # Load projection matrix
    W, b = load_projection_matrix()
    
    # Load B-ball dilemma samples
    b_ball_samples = load_b_ball_samples(args.error_analysis, args.max_samples)
    
    if not b_ball_samples:
        print("No B-ball dilemma samples found. Exiting.")
        return
    
    # Load large model
    print(f"\nLoading large model: {args.large_model}")
    model_name = MODELS[args.large_model]["model_name"]
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./models"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if HARDWARE_CONFIG["dtype"] == "float16" else torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="./models"
    )
    model.eval()
    
    print(f"Model loaded on device: {model.device}")
    
    # Run injection experiments
    all_results = []
    
    small_output_dir = MODELS[args.small_model]["output_dir"]
    
    for sample_idx in tqdm(b_ball_samples, desc="Running injection experiments"):
        # Load small model output
        small_path = os.path.join(small_output_dir, f"sample_{sample_idx:03d}.pt")
        if not os.path.exists(small_path):
            print(f"Warning: Small model output not found for sample {sample_idx}")
            continue
        
        small_output = torch.load(small_path, map_location="cpu")
        
        # Run injection with different configurations
        sample_results = run_injection_experiment(
            sample_idx,
            small_output,
            W,
            b,
            model,
            tokenizer,
            INJECTION_CONFIG["alpha_values"],
            INJECTION_CONFIG["injection_layers"]
        )
        
        all_results.extend(sample_results)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = INJECTION_CONFIG["results_output"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*80}")
        print("Injection Experiment Results")
        print('='*80)
        print(f"Total experiments run: {len(all_results)}")
        print(f"Results saved to: {output_path}")
        
        # Analyze results
        if 'gt_rank_after_injection' in df.columns:
            valid_ranks = df[df['gt_rank_after_injection'] > 0]
            if len(valid_ranks) > 0:
                print(f"\nAverage GT token rank after injection: {valid_ranks['gt_rank_after_injection'].mean():.2f}")
                
                # Best configuration
                best_config = valid_ranks.groupby(['injection_layer', 'alpha'])['gt_rank_after_injection'].mean()
                best_config = best_config.sort_values()
                print("\nBest configurations (by average GT rank):")
                print(best_config.head(5))
        
        print('='*80)
    else:
        print("No results collected")


if __name__ == "__main__":
    main()
