"""
Step 1: Prepare Data (V2 — QA datasets from HuggingFace)

Loads multiple datasets from HuggingFace Hub and converts them to a unified
QA format compatible with the experiment pipeline.

Supported datasets:
  - GSM8K (openai/gsm8k): Grade-school math word problems, numerical answers
  - ARC-Challenge (allenai/ai2_arc): Science reasoning, multiple-choice
  - MMLU (cais/mmlu): Knowledge QA across 57 subjects, multiple-choice

Unified sample format:
  {
      "id": str,
      "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
      "ground_truth": {
          "answer": str,           # The correct answer (e.g., "A", "42", "True")
          "answer_type": str,      # "multiple_choice" or "numerical"
          "dataset": str,          # Source dataset name
          "raw_question": str,     # Original question text
          "choices": list or null,  # For MC: [{"label": "A", "text": "..."}, ...]
      },
      "tools": [],                # Empty — no tools for QA tasks
      "task_type": "qa",
  }

Design principle:
  - Simple, single-turn QA → maximum model disagreement between sizes
  - 7B should be correct where 1.5B/3B fail (injection targets)
  - Clear ground truth with unambiguous correct answers
"""

import json
import random
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from tqdm import tqdm

from config import DATA_CONFIG


# ---------------------------------------------------------------------------
# Utility: JSON serialization for numpy types
# ---------------------------------------------------------------------------

def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# System prompts per dataset
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "gsm8k": (
        "You are a helpful math assistant. Solve the following problem step by step. "
        "After your reasoning, provide the final numerical answer."
    ),
    "arc_challenge": (
        "You are a helpful science assistant. Read the question carefully and choose "
        "the best answer from the given choices. Respond with just the letter of your "
        "answer (A, B, C, D, or E)."
    ),
    "mmlu": (
        "You are a knowledgeable assistant. Read the question and choose the best answer "
        "from the four choices. Respond with just the letter of your answer (A, B, C, or D)."
    ),
}


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------

def load_gsm8k(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from HuggingFace.

    Format: question (str), answer (str with step-by-step reasoning + #### <number>)
    We extract the final numerical answer from after ####.
    """
    from datasets import load_dataset

    print(f"\nLoading GSM8K dataset (max {num_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:num_samples]

    system_prompt = SYSTEM_PROMPTS["gsm8k"]
    samples = []

    for idx in tqdm(indices, desc="Processing GSM8K"):
        example = ds[idx]
        question = example["question"]
        answer_text = example["answer"]

        # Extract the final numerical answer (after ####)
        match = re.search(r'#{2,}\s*([\d,]+(?:\.\d+)?)', answer_text)
        if match:
            answer = match.group(1).replace(",", "")  # remove commas
        else:
            # Fallback: try to find the last number in the answer
            numbers = re.findall(r'[\d,]+(?:\.\d+)?', answer_text)
            answer = numbers[-1].replace(",", "") if numbers else "0"

        # Build the user message with the question
        user_message = question

        sample_id = f"gsm8k_{idx:05d}"

        samples.append({
            "id": sample_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "ground_truth": {
                "answer": answer,
                "answer_type": "numerical",
                "dataset": "gsm8k",
                "raw_question": question,
                "choices": None,
            },
            "tools": [],
            "task_type": "qa",
        })

    print(f"  Loaded {len(samples)} GSM8K samples")
    return samples


# ---------------------------------------------------------------------------
# ARC-Challenge loader
# ---------------------------------------------------------------------------

def load_arc_challenge(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load ARC-Challenge dataset from HuggingFace.

    Format: question (str), choices (dict with 'text' and 'label' lists), answerKey (str)
    """
    from datasets import load_dataset

    print(f"\nLoading ARC-Challenge dataset (max {num_samples} samples)...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    random.seed(seed + 1)  # different seed from GSM8K
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:num_samples]

    system_prompt = SYSTEM_PROMPTS["arc_challenge"]
    samples = []

    for idx in tqdm(indices, desc="Processing ARC-Challenge"):
        example = ds[idx]
        question = example["question"]
        choices = example["choices"]
        answer_key = example["answerKey"]

        choices_list = [
            {"label": label, "text": text}
            for label, text in zip(choices["label"], choices["text"])
        ]

        # Format the choices into the question
        choices_text = "\n".join(
            f"{c['label']}. {c['text']}" for c in choices_list
        )
        user_message = f"{question}\n\nChoices:\n{choices_text}"

        sample_id = f"arc_{example['id']}"

        samples.append({
            "id": sample_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "ground_truth": {
                "answer": answer_key,
                "answer_type": "multiple_choice",
                "dataset": "arc_challenge",
                "raw_question": question,
                "choices": choices_list,
            },
            "tools": [],
            "task_type": "qa",
        })

    print(f"  Loaded {len(samples)} ARC-Challenge samples")
    return samples


# ---------------------------------------------------------------------------
# MMLU loader
# ---------------------------------------------------------------------------

def load_mmlu(subjects: List[str], num_per_subject: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load MMLU dataset from HuggingFace (multiple subjects).

    Format per subject: question (str), choices (list of 4 strings), answer (int 0-3)
    """
    from datasets import load_dataset

    total = len(subjects) * num_per_subject
    print(f"\nLoading MMLU dataset ({len(subjects)} subjects, ~{num_per_subject} each, max {total})...")

    system_prompt = SYSTEM_PROMPTS["mmlu"]
    samples = []
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for subject_idx, subject in enumerate(tqdm(subjects, desc="MMLU subjects")):
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"  Warning: Could not load MMLU subject '{subject}': {e}")
            continue

        random.seed(seed + 2 + subject_idx)
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:num_per_subject]

        for idx in indices:
            example = ds[idx]
            question = example["question"]
            choices = example["choices"]
            answer_idx = example["answer"]

            answer_letter = label_map.get(answer_idx, "A")
            choices_list = [
                {"label": label, "text": text}
                for label, text in zip(["A", "B", "C", "D"], choices)
            ]

            choices_text = "\n".join(
                f"{c['label']}. {c['text']}" for c in choices_list
            )
            user_message = f"{question}\n\nChoices:\n{choices_text}"

            sample_id = f"mmlu_{subject}_{idx:04d}"

            samples.append({
                "id": sample_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "ground_truth": {
                    "answer": answer_letter,
                    "answer_type": "multiple_choice",
                    "dataset": "mmlu",
                    "subject": subject,
                    "raw_question": question,
                    "choices": choices_list,
                },
                "tools": [],
                "task_type": "qa",
            })

    print(f"  Loaded {len(samples)} MMLU samples across {len(subjects)} subjects")
    return samples


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_samples(samples: List[Dict[str, Any]], output_path: str):
    """Save samples as a JSON file."""
    print(f"Saving {len(samples)} samples to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializable = _make_json_serializable(samples)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1: Prepare QA Dataset from HuggingFace")
    parser.add_argument(
        "--num_gsm8k", type=int, default=100,
        help="Number of GSM8K samples",
    )
    parser.add_argument(
        "--num_arc", type=int, default=100,
        help="Number of ARC-Challenge samples",
    )
    parser.add_argument(
        "--num_mmlu_per_subject", type=int, default=10,
        help="Number of MMLU samples per subject",
    )
    parser.add_argument(
        "--seed", type=int, default=DATA_CONFIG["random_seed"],
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--total_samples", type=int, default=DATA_CONFIG["total_samples"],
        help="Total samples to include in test set (randomly sampled from all loaded)",
    )
    parser.add_argument(
        "--alignment_samples", type=int, default=DATA_CONFIG["num_alignment_samples"],
        help="Number of alignment samples (from test set)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Step 1: Prepare QA Dataset from HuggingFace")
    print("=" * 80)

    # --- Load all datasets ---
    all_samples = []

    # GSM8K
    gsm8k_samples = load_gsm8k(args.num_gsm8k, seed=args.seed)
    all_samples.extend(gsm8k_samples)

    # ARC-Challenge
    arc_samples = load_arc_challenge(args.num_arc, seed=args.seed)
    all_samples.extend(arc_samples)

    # MMLU
    mmlu_config = DATA_CONFIG["datasets"]["mmlu"]
    mmlu_samples = load_mmlu(
        subjects=mmlu_config["hf_subjects"],
        num_per_subject=args.num_mmlu_per_subject,
        seed=args.seed,
    )
    all_samples.extend(mmlu_samples)

    print(f"\n{'=' * 80}")
    print(f"Total samples loaded: {len(all_samples)}")
    print(f"  GSM8K:        {len(gsm8k_samples)}")
    print(f"  ARC-Challenge: {len(arc_samples)}")
    print(f"  MMLU:          {len(mmlu_samples)}")

    # --- Shuffle and split ---
    random.seed(args.seed)
    random.shuffle(all_samples)

    total = min(args.total_samples, len(all_samples))
    test_samples = all_samples[:total]

    # Alignment samples: take from the end (different from test)
    remaining = all_samples[total:]
    n_alignment = min(args.alignment_samples, len(remaining))
    if n_alignment < args.alignment_samples and len(all_samples) > total:
        # Not enough remaining; take from the beginning of remaining or supplement
        alignment_start = total
        alignment_end = min(total + n_alignment, len(all_samples))
        alignment_samples = all_samples[alignment_start:alignment_end]
    else:
        alignment_samples = remaining[:n_alignment]

    # If still not enough alignment samples, duplicate some from test
    if len(alignment_samples) < args.alignment_samples:
        deficit = args.alignment_samples - len(alignment_samples)
        alignment_samples.extend(test_samples[:deficit])
        print(f"  Warning: supplemented alignment set with {deficit} test samples")

    print(f"\nFinal split:")
    print(f"  Test set:       {len(test_samples)} samples")
    print(f"  Alignment set:  {len(alignment_samples)} samples")

    # --- Dataset composition summary ---
    from collections import Counter
    ds_counts = Counter(s["ground_truth"]["dataset"] for s in test_samples)
    print(f"\nTest set composition:")
    for ds_name, count in ds_counts.most_common():
        print(f"  {ds_name}: {count}")

    # --- Save ---
    output_base = os.path.dirname(DATA_CONFIG["sampled_data_path"])
    os.makedirs(output_base, exist_ok=True)

    save_samples(test_samples, DATA_CONFIG["sampled_data_path"])
    save_samples(alignment_samples, DATA_CONFIG["alignment_data_path"])

    # Also save a small preview
    preview_path = os.path.join(output_base, "preview.json")
    with open(preview_path, "w", encoding="utf-8") as f:
        # Save first 3 samples as preview
        preview = _make_json_serializable(test_samples[:3])
        json.dump(preview, f, indent=2, ensure_ascii=False)
    print(f"\nPreview saved to: {preview_path}")

    print("\n" + "=" * 80)
    print("Step 1 Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()