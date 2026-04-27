"""
Step 2: Run Inference
Load models and run inference on sampled data, recording all intermediate states
"""

import json
import os
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODELS, DATA_CONFIG, GENERATION_CONFIG, HARDWARE_CONFIG


def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype: str = "float16"):
    """
    Load model and tokenizer
    """
    print(f"Loading model: {model_name}")
    print(f"Device: {device}, dtype: {dtype}")
    
    # Determine torch dtype
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./models"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="./models",
        max_memory=HARDWARE_CONFIG.get("max_memory", None)
    )
    
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    return model, tokenizer


def prepare_input(sample: Dict[str, Any], tokenizer) -> str:
    """
    Prepare input text from sample messages
    """
    messages = sample["messages"]
    
    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return text
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
    
    # Fallback: manually construct prompt
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            text += f"System: {content}\n\n"
        elif role == "user":
            text += f"User: {content}\n\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n\n"
        elif role == "tool":
            text += f"Tool: {content}\n\n"
    
    text += "Assistant: "
    return text


def run_inference_with_states(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 512
) -> Dict[str, Any]:
    """
    Run inference and capture all intermediate states
    """
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # Generate with state recording
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=GENERATION_CONFIG.get("temperature", 0.0),
            do_sample=GENERATION_CONFIG.get("do_sample", False),
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            output_attentions=GENERATION_CONFIG.get("output_attentions", False),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # Extract information
    generated_ids = outputs.sequences[0]  # [seq_len]
    scores = outputs.scores  # List of [batch=1, vocab_size]
    hidden_states = outputs.hidden_states  # Tuple of tuples
    
    # Process scores to get logits and probs
    logits_per_step = []
    probs_per_step = []
    
    for score in scores:
        logits = score[0]  # [vocab_size]
        probs = torch.softmax(logits, dim=-1)
        
        logits_per_step.append(logits.cpu())
        probs_per_step.append(probs.cpu())
    
    # Process hidden states
    # hidden_states is a tuple of length num_generated_tokens
    # Each element is a tuple of (num_layers + 1) tensors [batch=1, seq_len, hidden_dim]
    hidden_states_per_step = []
    
    if hidden_states is not None:
        for step_hidden in hidden_states:
            # step_hidden is tuple of layer hidden states
            layer_hiddens = []
            for layer_hidden in step_hidden:
                # Take the last token's hidden state
                last_hidden = layer_hidden[0, -1, :].cpu()  # [hidden_dim]
                layer_hiddens.append(last_hidden)
            hidden_states_per_step.append(layer_hiddens)
    
    return {
        "input_ids": input_ids[0].cpu(),
        "generated_ids": generated_ids.cpu(),
        "logits_per_step": logits_per_step,
        "probs_per_step": probs_per_step,
        "hidden_states_per_step": hidden_states_per_step,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True)
    }


def process_sample(
    sample: Dict[str, Any],
    model,
    tokenizer,
    sample_idx: int,
    output_dir: str
) -> bool:
    """
    Process a single sample
    """
    try:
        # Prepare input
        input_text = prepare_input(sample, tokenizer)
        
        # Run inference
        result = run_inference_with_states(
            model,
            tokenizer,
            input_text,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"]
        )
        
        # Add ground truth
        result["ground_truth"] = sample.get("ground_truth", {})
        result["sample_id"] = sample.get("id", f"sample_{sample_idx}")
        result["input_text"] = input_text
        
        # Save result
        output_path = os.path.join(output_dir, f"sample_{sample_idx:03d}.pt")
        torch.save(result, output_path)
        
        return True
    
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference_for_model(
    model_key: str,
    samples: List[Dict[str, Any]],
    skip_existing: bool = True
):
    """
    Run inference for a specific model on all samples
    """
    print("="*80)
    print(f"Running inference for model: {model_key}")
    print("="*80)
    
    model_config = MODELS[model_key]
    model_name = model_config["model_name"]
    output_dir = model_config["output_dir"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=HARDWARE_CONFIG["device"],
        dtype=HARDWARE_CONFIG["dtype"]
    )
    
    # Process samples
    print(f"\nProcessing {len(samples)} samples...")
    successful = 0
    failed = 0
    
    for idx, sample in enumerate(tqdm(samples)):
        output_path = os.path.join(output_dir, f"sample_{idx:03d}.pt")
        
        # Skip if already exists
        if skip_existing and os.path.exists(output_path):
            print(f"Skipping sample {idx} (already exists)")
            successful += 1
            continue
        
        success = process_sample(sample, model, tokenizer, idx, output_dir)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print(f"\nInference complete for {model_key}")
    print(f"Successful: {successful}/{len(samples)}")
    print(f"Failed: {failed}/{len(samples)}")
    print("="*80)
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--model", type=str, choices=["qwen1.5B", "qwen7B", "qwen14B", "all"],
                      default="all", help="Which model to run inference for")
    parser.add_argument("--data_path", type=str, default=DATA_CONFIG["sampled_data_path"],
                      help="Path to sampled data")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                      help="Skip samples that already have output files")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 2: Model Inference")
    print("="*80)
    
    # Load samples
    print(f"Loading samples from: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} samples")
    
    # Limit samples if specified
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples for testing")
    
    # Determine which models to run
    if args.model == "all":
        models_to_run = ["qwen1.5B", "qwen7B", "qwen14B"]
    else:
        models_to_run = [args.model]
    
    # Run inference for each model
    results = {}
    for model_key in models_to_run:
        successful, failed = run_inference_for_model(
            model_key,
            samples,
            skip_existing=args.skip_existing
        )
        results[model_key] = {
            "successful": successful,
            "failed": failed,
            "total": len(samples)
        }
    
    # Print summary
    print("\n" + "="*80)
    print("Inference Complete - Summary")
    print("="*80)
    for model_key, result in results.items():
        print(f"{model_key}:")
        print(f"  Successful: {result['successful']}/{result['total']}")
        print(f"  Failed: {result['failed']}/{result['total']}")
        print(f"  Success Rate: {result['successful']/result['total']*100:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
