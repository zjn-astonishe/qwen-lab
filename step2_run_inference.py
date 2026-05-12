"""
Step 2: Run Inference
Load models and run inference on sampled data, recording all intermediate states
(logits, hidden states).

V5 fix (transformers 5.x compat):
  - model.generate() no longer passes output_hidden_states (causes IndexError in
    transformers 5.x due to internal format change).
  - Instead, generate text first, then do a separate forward pass on the full
    sequence (input + generated) to extract all hidden states at every position.
  - torch_dtype renamed to dtype (transformers 5.x deprecation).
  - Output format is unchanged — downstream steps 3-9 require no modification.
"""

import json
import os
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODELS, DATA_CONFIG, GENERATION_CONFIG, HARDWARE_CONFIG, MEMORY_CONFIG
from utils import cleanup_gpu


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype: str = "float16"):
    """Load a HuggingFace causal LM and its tokenizer."""
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir="./models"
    )

    # transformers 5.x uses 'dtype' instead of 'torch_dtype'
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            max_memory=HARDWARE_CONFIG.get("max_memory", None),
        )
    except TypeError:
        # Fallback for older transformers that use torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            max_memory=HARDWARE_CONFIG.get("max_memory", None),
        )

    model.eval()

    print(f"  Model: {model_name}")
    print(f"  Device: {model.device}, dtype: {model.dtype}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_input(sample: Dict[str, Any], tokenizer) -> str:
    """Build the prompt string from a sample's messages."""
    messages = sample["messages"]
    tools = sample.get("tools", None)

    if hasattr(tokenizer, "apply_chat_template"):
        # Try with tools first, then without
        for kwargs in [
            {"tools": tools, "tokenize": False, "add_generation_prompt": True},
            {"tokenize": False, "add_generation_prompt": True},
        ]:
            try:
                if "tools" in kwargs and kwargs["tools"] is None:
                    del kwargs["tools"]
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:
                continue

    # Fallback: manual formatting
    role_map = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
    parts = []
    for msg in messages:
        role = role_map.get(msg.get("role", "user"), "User")
        parts.append(f"{role}: {msg.get('content', '')}")
    parts.append("Assistant: ")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Hidden states extraction via forward pass
# ---------------------------------------------------------------------------

def extract_hidden_states_from_forward(
    model,
    full_sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_length: int,
    skip_embedding: bool = True,
) -> Dict[str, Any]:
    """Run a single forward pass on the full sequence and extract hidden states.

    The forward pass uses causal attention, so each position's hidden state
    only depends on previous positions — identical to what model.generate()
    would produce internally.

    Args:
        model: HuggingFace CausalLM
        full_sequence_ids: tensor of shape (1, total_seq_len) = input + generated
        attention_mask: corresponding attention mask
        input_length: number of input tokens (to separate prefill from gen steps)
        skip_embedding: whether to skip the embedding layer (index 0)

    Returns:
        Dict with:
            - "hidden_states_per_step": list of per-generation-step layer hidden states
            - "prefill_hidden_states": list of per-layer hidden states at last input token
    """
    result = {}

    # Use inference_mode for better performance
    with torch.inference_mode():
        outputs = model(
            input_ids=full_sequence_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,  # Don't need cache for single forward pass
        )

    # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim) per layer
    # Index 0 = embedding, 1..N = transformer layers
    all_layers_hs = outputs.hidden_states
    num_layers_total = len(all_layers_hs)  # including embedding

    # Determine layer range (skip embedding if requested)
    layer_start = 1 if skip_embedding and num_layers_total > 1 else 0

    # --- Prefill hidden states: last INPUT token ---
    # Optimize: extract all layers at once for the prefill position
    prefill_idx = input_length - 1
    prefill_layers = [
        all_layers_hs[layer_idx][0, prefill_idx, :].cpu().clone()
        for layer_idx in range(layer_start, num_layers_total)
    ]
    result["prefill_hidden_states"] = prefill_layers

    # --- Per-step generation hidden states ---
    total_len = full_sequence_ids.shape[1]
    num_generated = total_len - input_length

    # Optimize: batch extract hidden states for all generation steps
    hidden_states_per_step = []
    if num_generated > 0:
        for gen_step in range(num_generated):
            pos = input_length + gen_step
            step_layers = [
                all_layers_hs[layer_idx][0, pos, :].cpu().clone()
                for layer_idx in range(layer_start, num_layers_total)
            ]
            hidden_states_per_step.append(step_layers)

    result["hidden_states_per_step"] = hidden_states_per_step

    # Cleanup
    del outputs, all_layers_hs
    cleanup_gpu()
    return result


# ---------------------------------------------------------------------------
# Generation with full state recording
# ---------------------------------------------------------------------------

def run_inference_with_states(
    model, tokenizer, input_text: str,
    max_new_tokens: int = 512,
    save_hidden_states: bool = True,
    save_full_logits: bool = False,
    top_k_logits: int = 100,
    skip_embedding: bool = True,
) -> Dict[str, Any]:
    """Generate tokens and record logits / hidden states at every step.

    V5: Decoupled generation from hidden-states extraction.
      1. model.generate() produces the text + per-step scores (NO output_hidden_states).
      2. A separate forward pass on the full sequence extracts all hidden states.
    
    Optimizations:
      - Use torch.inference_mode() for better performance
      - Enable use_cache for KV cache optimization
      - Optimize tensor operations to reduce memory transfers
    """
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_length = input_ids.shape[1]

    # --- Step 1: Generate tokens (without output_hidden_states) ---
    # Use inference_mode for better performance than no_grad
    with torch.inference_mode():
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            # NOTE: output_hidden_states removed — causes IndexError in transformers 5.x
            "output_attentions": False,
            "pad_token_id": pad_token_id,
            "do_sample": GENERATION_CONFIG.get("do_sample", False),
            "use_cache": True,  # Enable KV cache for faster generation
        }
        gen_outputs = model.generate(**gen_kwargs)

    # --- Decode ---
    generated_ids = gen_outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # --- Logits / probs from generation scores ---
    top_k_info: List[Dict[str, torch.Tensor]] = []

    if gen_outputs.scores is not None:
        # Process all scores at once for better efficiency
        for score in gen_outputs.scores:
            # score shape: (batch, vocab_size) in standard transformers
            if score.dim() == 3:
                score = score.squeeze(0)  # handle (1, 1, vocab) if needed
            if score.dim() == 2:
                score = score[0]  # (batch, vocab) -> (vocab,)

            # Keep on GPU for softmax computation, then move to CPU
            logits_gpu = score
            probs_gpu = torch.softmax(logits_gpu, dim=-1)
            
            if save_full_logits:
                top_k_info.append({
                    "full_logits": logits_gpu.cpu(),
                    "full_probs": probs_gpu.cpu()
                })
            else:
                k_val = min(max(1, top_k_logits), probs_gpu.size(0))
                top_k_probs, top_k_indices = torch.topk(probs_gpu, k=k_val)
                top_k_logits_vals = logits_gpu[top_k_indices]
                top_k_info.append({
                    "indices": top_k_indices.cpu(),
                    "logits": top_k_logits_vals.cpu(),
                    "probs": top_k_probs.cpu(),
                })
            
            # Explicit cleanup
            del logits_gpu, probs_gpu
    del gen_outputs.scores

    # --- Step 2: Extract hidden states via forward pass ---
    if save_hidden_states:
        # Build full sequence: input + generated tokens
        # generated_ids already includes input_ids prefix
        full_ids = generated_ids.unsqueeze(0).to(model.device)
        
        # Optimize attention mask construction
        if attention_mask is not None:
            # Reconstruct full attention mask efficiently
            num_new_tokens = full_ids.shape[1] - input_length
            if num_new_tokens > 0:
                new_mask = torch.ones(
                    1, num_new_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                full_mask = torch.cat([attention_mask, new_mask], dim=1)
            else:
                full_mask = attention_mask
        else:
            full_mask = torch.ones_like(full_ids)

        hs_result = extract_hidden_states_from_forward(
            model, full_ids, full_mask,
            input_length=input_length,
            skip_embedding=skip_embedding,
        )
    else:
        hs_result = {}

    del gen_outputs
    cleanup_gpu()

    # --- Assemble result (same format as before) ---
    result: Dict[str, Any] = {
        "input_ids": input_ids.cpu(),
        "generated_ids": generated_ids.cpu(),
        "generated_text": generated_text,
        "num_generated_tokens": len(generated_ids) - input_length,
    }

    if save_full_logits:
        result["logits_per_step"] = [e["full_logits"] for e in top_k_info]
        result["probs_per_step"] = [e["full_probs"] for e in top_k_info]
    else:
        result["top_k_info"] = top_k_info

    if "hidden_states_per_step" in hs_result:
        result["hidden_states_per_step"] = hs_result["hidden_states_per_step"]

    return result


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(sample, model, tokenizer, sample_idx, output_dir,
                   skip_existing=True) -> bool:
    """Process a single sample: run inference and save results."""
    output_path = os.path.join(output_dir, f"sample_{sample_idx:03d}.pt")

    if skip_existing and os.path.exists(output_path):
        return True

    try:
        input_text = prepare_input(sample, tokenizer)

        result = run_inference_with_states(
            model, tokenizer, input_text,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            save_hidden_states=MEMORY_CONFIG.get("save_hidden_states", True),
            save_full_logits=MEMORY_CONFIG.get("save_full_logits", False),
            top_k_logits=MEMORY_CONFIG.get("save_top_k_logits", 100),
            skip_embedding=MEMORY_CONFIG.get("skip_embedding_layer", True),
        )

        # Prefill hidden states are already included in the result
        # from extract_hidden_states_from_forward
        # No need for a separate get_prefill_hidden_states call

        result["ground_truth"] = sample.get("ground_truth", {})
        result["ground_truth_normalized"] = sample.get("ground_truth_normalized", {})
        result["sample_id"] = sample.get("id", f"sample_{sample_idx}")
        result["input_text"] = input_text

        os.makedirs(output_dir, exist_ok=True)
        torch.save(result, output_path)

        del result
        cleanup_gpu()
        return True

    except Exception as e:
        import traceback
        print(f"  Error processing sample {sample_idx} ({sample.get('id', '?')}): {e}")
        # Print full traceback for debugging
        traceback.print_exc()
        cleanup_gpu()
        return False


# ---------------------------------------------------------------------------
# Full model loop
# ---------------------------------------------------------------------------

def run_inference_for_model(model_key: str, samples: List[Dict],
                            skip_existing: bool = True) -> Dict[str, int]:
    """Run inference for a single model on all samples."""
    model_config = MODELS[model_key]
    model_name = model_config["model_name"]
    output_dir = model_config["output_dir"]

    print(f"\n{'=' * 80}")
    print(f"Running inference: {model_key} ({model_name})")
    print(f"  Samples: {len(samples)}, skip_existing: {skip_existing}")
    print(f"{'=' * 80}")

    model, tokenizer = load_model_and_tokenizer(
        model_name, device=HARDWARE_CONFIG["device"], dtype=HARDWARE_CONFIG["dtype"]
    )

    successful = 0
    failed = 0
    cleanup_interval = MEMORY_CONFIG.get("cleanup_frequency", 10)

    for idx, sample in enumerate(tqdm(samples, desc=model_key)):
        if process_sample(sample, model, tokenizer, idx, output_dir, skip_existing):
            successful += 1
        else:
            failed += 1

        if (idx + 1) % cleanup_interval == 0:
            cleanup_gpu()

    del model, tokenizer
    cleanup_gpu()

    print(f"  Done: {successful} ok, {failed} failed / {len(samples)} total")
    return {"successful": successful, "failed": failed, "total": len(samples)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2: Model Inference")
    parser.add_argument("--model", type=str,
                        choices=["qwen1.5B", "qwen7B", "qwen3B", "all"],
                        default="all", help="Which model to run inference for")
    parser.add_argument("--data_type", type=str, choices=["test", "alignment"],
                        default="test", help="Which dataset to use")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to data (overrides data_type)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip samples that already have output files")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    args = parser.parse_args()

    print("=" * 80)
    print("Step 2: Model Inference")
    print("=" * 80)

    # Determine data path
    if args.data_path:
        data_path = args.data_path
    elif args.data_type == "alignment":
        data_path = DATA_CONFIG["alignment_data_path"]
    else:
        data_path = DATA_CONFIG["sampled_data_path"]

    print(f"Loading samples from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"  Loaded {len(samples)} samples")

    if args.max_samples is not None:
        samples = samples[:args.max_samples]
        print(f"  Limited to {len(samples)} samples")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = {}
    for model_key in models_to_run:
        results[model_key] = run_inference_for_model(model_key, samples, args.skip_existing)

    # Summary
    print(f"\n{'=' * 80}")
    print("Inference Complete")
    print(f"{'=' * 80}")
    for model_key, r in results.items():
        rate = r["successful"] / r["total"] * 100 if r["total"] else 0
        print(f"  {model_key}: {r['successful']}/{r['total']} ok ({rate:.1f}%), "
              f"{r['failed']} failed")


if __name__ == "__main__":
    main()
