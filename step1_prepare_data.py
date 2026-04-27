"""
Step 1: Prepare Data
Download BFCL V3 dataset, filter multi-turn samples, and create sampled datasets
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm

from config import DATA_CONFIG


def download_bfcl_v3_data(data_dir: str):
    """
    Download BFCL V3 dataset
    Note: This is a placeholder. You may need to implement actual download logic
    based on where BFCL V3 is hosted (GitHub, HuggingFace, etc.)
    """
    print(f"Downloading BFCL V3 dataset to {data_dir}...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # TODO: Implement actual download logic
    # For now, assume the user has downloaded the data manually
    # or provide instructions
    
    print(f"Please ensure BFCL V3 data is available in {data_dir}")
    print("Expected format: JSON files with function calling examples")
    
    # Check if data exists
    data_files = list(Path(data_dir).glob("*.json"))
    if not data_files:
        print(f"\nWARNING: No JSON files found in {data_dir}")
        print("Please download BFCL V3 dataset from:")
        print("https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard")
        print("\nOr use HuggingFace datasets:")
        print("from datasets import load_dataset")
        print("dataset = load_dataset('gorilla-llm/Berkeley-Function-Calling-Leaderboard')")
        return False
    
    print(f"Found {len(data_files)} data files")
    return True


def load_bfcl_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load BFCL V3 data from JSON files
    """
    print("Loading BFCL V3 data...")
    all_data = []
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        print("Attempting to load from HuggingFace datasets...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train")
            
            for item in tqdm(dataset):
                all_data.append(dict(item))
            
            print(f"Loaded {len(all_data)} samples from HuggingFace")
            return all_data
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            print("Creating synthetic samples for testing...")
            return create_synthetic_samples()
    
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    print(f"Loaded {len(all_data)} samples from local files")
    return all_data


def create_synthetic_samples(num_samples: int = 500) -> List[Dict[str, Any]]:
    """
    Create synthetic samples for testing when real data is not available
    """
    print(f"Creating {num_samples} synthetic samples...")
    
    synthetic_data = []
    functions = [
        "get_weather", "search_web", "send_email", "create_calendar_event",
        "get_stock_price", "translate_text", "calculate_math", "get_news"
    ]
    
    for i in range(num_samples):
        num_turns = random.randint(2, 5)
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant with access to various tools."
        })
        
        # Multi-turn conversation
        for turn in range(num_turns):
            # User message
            user_content = f"User request {turn + 1} for sample {i}"
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Assistant tool call
            func_name = random.choice(functions)
            tool_call = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{i}_{turn}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps({"param": f"value_{turn}"})
                    }
                }]
            }
            messages.append(tool_call)
            
            # Tool response
            messages.append({
                "role": "tool",
                "content": f"Result from {func_name}",
                "tool_call_id": f"call_{i}_{turn}"
            })
        
        synthetic_data.append({
            "id": f"synthetic_{i}",
            "messages": messages,
            "ground_truth": {
                "function": random.choice(functions),
                "arguments": {"param": "expected_value"}
            }
        })
    
    return synthetic_data


def convert_to_unified_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert sample to unified format with messages list
    """
    # If already in correct format, return as is
    if "messages" in sample:
        return sample
    
    # Otherwise, try to construct messages from other fields
    messages = []
    
    # Add system message if exists
    if "system" in sample:
        messages.append({
            "role": "system",
            "content": sample["system"]
        })
    
    # Add conversation turns
    if "conversation" in sample:
        messages.extend(sample["conversation"])
    elif "prompt" in sample:
        messages.append({
            "role": "user",
            "content": sample["prompt"]
        })
    
    return {
        "id": sample.get("id", "unknown"),
        "messages": messages,
        "ground_truth": sample.get("ground_truth", sample.get("expected_output", {}))
    }


def filter_multi_turn_samples(data: List[Dict[str, Any]], min_turns: int = 2) -> List[Dict[str, Any]]:
    """
    Filter samples with at least min_turns conversation turns
    """
    print(f"Filtering samples with at least {min_turns} turns...")
    
    filtered = []
    for sample in tqdm(data):
        messages = sample.get("messages", [])
        
        # Count user turns (exclude system messages)
        user_turns = sum(1 for msg in messages if msg.get("role") == "user")
        
        if user_turns >= min_turns:
            filtered.append(sample)
    
    print(f"Filtered to {len(filtered)} multi-turn samples (from {len(data)} total)")
    return filtered


def sample_data(data: List[Dict[str, Any]], num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly sample data with fixed seed for reproducibility
    """
    print(f"Sampling {num_samples} samples with seed {seed}...")
    
    random.seed(seed)
    
    if len(data) < num_samples:
        print(f"WARNING: Requested {num_samples} samples but only {len(data)} available")
        return data
    
    sampled = random.sample(data, num_samples)
    print(f"Sampled {len(sampled)} samples")
    
    return sampled


def save_samples(samples: List[Dict[str, Any]], output_path: str):
    """
    Save sampled data to JSON file
    """
    print(f"Saving samples to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare BFCL V3 dataset")
    parser.add_argument("--data_dir", type=str, default=DATA_CONFIG["data_dir"],
                      help="Directory containing BFCL V3 data")
    parser.add_argument("--num_samples", type=int, default=DATA_CONFIG["num_samples"],
                      help="Number of samples to select")
    parser.add_argument("--num_alignment", type=int, default=DATA_CONFIG["num_alignment_samples"],
                      help="Number of alignment samples to select")
    parser.add_argument("--min_turns", type=int, default=DATA_CONFIG["min_turns"],
                      help="Minimum number of conversation turns")
    parser.add_argument("--seed", type=int, default=DATA_CONFIG["random_seed"],
                      help="Random seed for sampling")
    parser.add_argument("--use_synthetic", action="store_true",
                      help="Use synthetic data for testing")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 1: Data Preparation")
    print("="*80)
    
    # Download/check data
    if not args.use_synthetic:
        data_available = download_bfcl_v3_data(args.data_dir)
        if not data_available:
            print("\nUsing synthetic data for testing...")
            args.use_synthetic = True
    
    # Load data
    if args.use_synthetic:
        all_data = create_synthetic_samples(num_samples=1000)
    else:
        all_data = load_bfcl_data(args.data_dir)
    
    if not all_data:
        print("ERROR: No data loaded. Exiting.")
        return
    
    # Convert to unified format
    print("\nConverting to unified format...")
    unified_data = [convert_to_unified_format(sample) for sample in tqdm(all_data)]
    
    # Filter multi-turn samples
    multi_turn_data = filter_multi_turn_samples(unified_data, args.min_turns)
    
    if len(multi_turn_data) < args.num_samples + args.num_alignment:
        print(f"\nWARNING: Not enough multi-turn samples!")
        print(f"Required: {args.num_samples + args.num_alignment}")
        print(f"Available: {len(multi_turn_data)}")
        print("Proceeding with available samples...")
    
    # Sample data
    total_needed = args.num_samples + args.num_alignment
    sampled_data = sample_data(multi_turn_data, min(total_needed, len(multi_turn_data)), args.seed)
    
    # Split into test and alignment sets
    test_samples = sampled_data[:args.num_samples]
    alignment_samples = sampled_data[args.num_samples:args.num_samples + args.num_alignment]
    
    # Save samples
    save_samples(test_samples, DATA_CONFIG["sampled_data_path"])
    save_samples(alignment_samples, DATA_CONFIG["alignment_data_path"])
    
    # Print statistics
    print("\n" + "="*80)
    print("Data Preparation Complete!")
    print("="*80)
    print(f"Test samples: {len(test_samples)}")
    print(f"Alignment samples: {len(alignment_samples)}")
    print(f"Total samples: {len(sampled_data)}")
    print(f"Output directory: {os.path.dirname(DATA_CONFIG['sampled_data_path'])}")
    print("="*80)


if __name__ == "__main__":
    main()
