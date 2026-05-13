"""
Step 2: Run Inference (V6 — performance optimized)

Load models and run inference on sampled data, recording all intermediate states
(logits, hidden states).

V6 optimizations:
  - Added torch.compile() for 20-40% inference speedup
  - Replaced torch.no_grad() with torch.inference_mode() (faster)
  - Reduced cleanup_gpu() from 3×/sample to 1×/20 samples (eliminates GPU stalls)
  - Batched GPU→CPU transfers: torch.stack all layers → single .cpu() call
    (reduces 56 CUDA synchronizations to 2 per sample)
  - Removed cleanup_gpu from extract_hidden_states_from_forward (was called
    inside the hot path, stalling GPU between every sample)
  - Added --no_compile flag to disable torch.compile() if memory is tight

V5 fix (transformers 5.x compat):
  - model.generate() no longer passes output_hidden_states
  - Separate forward pass on full sequence for hidden states extraction
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

def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype: str = "float16",
                             use_compile: bool = True):
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

    # torch.compile() for 20-40% speedup (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        try:
            print("  Compiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile() OK (reduce-overhead mode)")
        except Exception as e:
            print(f"  torch.compile() failed ({e}), using eager mode")

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

    V6 optimized: batched GPU→CPU transfer using torch.stack.
    Old code: 56 individual .cpu() calls → New code: 2 batched .cpu() calls.
    """
    result = {}

    with torch.inference_mode():
        outputs = model(
            input_ids=full_sequence_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
        )

    all_layers_hs = outputs.hidden_states
    num_layers_total = len(all_layers_hs)

    layer_start = 1 if skip_embedding and num_layers_total > 1 else 0
    num_transformer_layers = num_layers_total - layer_start
    total_len = full_sequence_ids.shape[1]

    # --- Prefill: last input token — stack all layers at once, single .cpu() ---
    prefill_stack = torch.stack(
        [all_layers_hs[layer_start + i][0, input_length - 1, :]
         for i in range(num_transformer_layers)],
        dim=0,  # (num_layers, hidden_dim)
    ).cpu()
    result["prefill_hidden_states"] = [prefill_stack[i] for i in range(num_transformer_layers)]

    # --- Per-generation-step hidden states: batch all steps, single .cpu() ---
    num_generated = total_len - input_length
    if num_generated > 0:
        gen_stacks = []
        for gen_step in range(num_generated):
            pos = input_length + gen_step
            step_stack = torch.stack(
                [all_layers_hs[layer_start + i][0, pos, :]
                 for i in range(num_transformer_layers)],
                dim=0,
            )
            gen_stacks.append(step_stack)

        # (num_generated, num_layers, hidden_dim) → single GPU→CPU transfer
        all_gen = torch.stack(gen_stacks, dim=0).cpu()

        hidden_states_per_step = []
        for gen_step in range(num_generated):
            step_layers = [all_gen[gen_step, i] for i in range(num_transformer_layers)]
            hidden_states_per_step.append(step_layers)

        result["hidden_states_per_step"] = hidden_states_per_step

    del outputs
    # NO cleanup_gpu() here — removed from hot path
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
    """Generate tokens and record logits / hidden states at every step."""
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_length = input_ids.shape[1]

    # --- Step 1: Generate tokens ---
    with torch.inference_mode():
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_attentions": False,
            "pad_token_id": pad_token_id,
            "do_sample": GENERATION_CONFIG.get("do_sample", False),
        }
        gen_outputs = model.generate(**gen_kwargs)

    # --- Decode ---
    generated_ids = gen_outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # --- Logits / probs from generation scores ---
    top_k_info: List[Dict[str, torch.Tensor]] = []

    if gen_outputs.scores is not None:
        for score in gen_outputs.scores:
            if score.dim() == 3:
                score = score.squeeze(0)
            if score.dim() == 2:
                score = score[0]

            logits = score.cpu()
            probs = torch.softmax(logits, dim=-1)

            if save_full_logits:
                top_k_info.append({"full_logits": logits, "full_probs": probs})
            else:
                k_val = min(max(1, top_k_logits), probs.size(0))
                top_k_probs, top_k_indices = torch.topk(probs, k=k_val)
                top_k_logits_vals = logits[top_k_indices]
                top_k_info.append({
                    "indices": top_k_indices,
                    "logits": top_k_logits_vals,
                    "probs": top_k_probs,
                })

            del logits, probs
    del gen_outputs.scores

    # --- Step 2: Extract hidden states via forward pass ---
    if save_hidden_states:
        full_ids = generated_ids.unsqueeze(0).to(model.device)
        full_mask = torch.ones_like(full_ids)
        if attention_mask is not None:
            full_mask = torch.cat([
                attention_mask,
                torch.ones(1, full_ids.shape[1] - input_length,
                          dtype=attention_mask.dtype, device=attention_mask.device),
            ], dim=1)

        hs_result = extract_hidden_states_from_forward(
            model, full_ids, full_mask,
            input_length=input_length,
            skip_embedding=skip_embedding,
        )
    else:
        hs_result = {}

    del gen_outputs

    # --- Assemble result ---
    result: Dict[str, Any] = {
        "input_ids": input_ids.cpu(),
        "generated_ids": generated_ids.cpu(),
        "generated_text": generated_text,
        "generated_answer_only": tokenizer.decode(
            generated_ids[input_length:], skip_special_tokens=True
        ),
        "num_generated_tokens": len(generated_ids) - input_length,
    }

    if save_full_logits:
        result["logits_per_step"] = [e["full_logits"] for e in top_k_info]
        result["probs_per_step"] = [e["full_probs"] for e in top_k_info]
    else:
        result["top_k_info"] = top_k_info

    if "hidden_states_per_step" in hs_result:
        result["hidden_states_per_step"] = hs_result["hidden_states_per_step"]
    if "prefill_hidden_states" in hs_result:
        result["prefill_hidden_states"] = hs_result["prefill_hidden_states"]

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

        result["ground_truth"] = sample.get("ground_truth", {})
        result["ground_truth_normalized"] = sample.get("ground_truth_normalized", {})
        result["sample_id"] = sample.get("id", f"sample_{sample_idx}")
        result["input_text"] = input_text

        os.makedirs(output_dir, exist_ok=True)
        torch.save(result, output_path)

        del result
        return True

    except Exception as e:
        import traceback
        print(f"  Error processing sample {sample_idx} ({sample.get('id', '?')}): {e}")
        traceback.print_exc()
        cleanup_gpu()
        return False


# ---------------------------------------------------------------------------
# Full model loop
# ---------------------------------------------------------------------------

def run_inference_for_model(model_key: str, samples: List[Dict],
                            skip_existing: bool = True,
                            use_compile: bool = True) -> Dict[str, int]:
    """Run inference for a single model on all samples."""
    model_config = MODELS[model_key]
    model_name = model_config["model_name"]
    output_dir = model_config["output_dir"]

    print(f"\n{'=' * 80}")
    print(f"Running inference: {model_key} ({model_name})")
    print(f"  Samples: {len(samples)}, skip_existing: {skip_existing}")
    print(f"{'=' * 80}")

    model, tokenizer = load_model_and_tokenizer(
        model_name, device=HARDWARE_CONFIG["device"],
        dtype=HARDWARE_CONFIG["dtype"], use_compile=use_compile,
    )

    successful = 0
    failed = 0
    cleanup_interval = 20  # Reduced from 10 — less GPU stalling

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
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile() (slower but uses less memory)")
    args = parser.parse_args()

    print("=" * 80)
    print("Step 2: Model Inference (V6 — optimized)")
    print("=" * 80)

    # Determine data path
    if args.data_path:
        data_path = args.data_path
    elif args.data_type == "alignment":
        data_path = DATA_CONFIG["alignment_data_path"]
    else:
        data_path = DATA_CONFIG["sampled_data_path"]

    print(f"Loading samples from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f"  Loaded {len(samples)} samples")

    if args.max_samples is not None:
        samples = samples[:args.max_samples]
        print(f"  Limited to {len(samples)} samples")

    use_compile = not args.no_compile
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = {}
    for model_key in models_to_run:
        results[model_key] = run_inference_for_model(
            model_key, samples, args.skip_existing, use_compile=use_compile,
        )

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