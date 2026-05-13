"""
Utility Functions
Common helper functions used across the experiment.

V2: Added centralized model output loading, safe dict parsing, and
    memory estimation that uses actual config values.
"""

import os
import json
import ast
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Time / size formatting
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_file_size(path: str) -> int:
    """Get file or directory size in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    return 0


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Dict[str, Any], path: str, indent: int = 2):
    """Save data as JSON with proper formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON data."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}

    usage = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        usage[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "utilization": round(allocated / total, 3) if total > 0 else 0,
        }
    return usage


def cleanup_gpu():
    """Aggressively release GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

def create_experiment_metadata() -> Dict[str, Any]:
    """Create metadata for the current experiment run."""
    import sys
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch_version": getattr(torch, "__version__", "N/A"),
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
            }
            metadata["gpu_info"].append(gpu_info)
    return metadata


# ---------------------------------------------------------------------------
# Model output loading (centralized)
# ---------------------------------------------------------------------------

def load_model_output(model_output_dir: str, sample_idx: int,
                      map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    """Load a single model output file (.pt).

    Returns None if the file does not exist or fails to load.
    """
    path = os.path.join(model_output_dir, f"sample_{sample_idx:03d}.pt")
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


def load_model_outputs(model_key: str, num_samples: int,
                       sub_dir: str = "",
                       map_location: str = "cpu") -> List[Optional[Dict[str, Any]]]:
    """Load all model outputs for a given model.

    Args:
        model_key: Key in MODELS config (e.g., "qwen1.5B").
        num_samples: Number of samples to load.
        sub_dir: Optional sub-directory within output_dir.
        map_location: Device to map tensors to (default: "cpu").
                    Pass "cuda" to load directly to GPU.

    Returns:
        List of output dicts (None for missing/failed samples).
    """
    # Lazy import to avoid circular dependency at module level
    from config import MODELS
    base = MODELS[model_key]["output_dir"]
    if sub_dir:
        base = os.path.join(base, sub_dir)

    outputs = []
    for i in range(num_samples):
        outputs.append(load_model_output(base, i, map_location=map_location))
    return outputs


# ---------------------------------------------------------------------------
# Safe dict / JSON parsing (replaces eval())
# ---------------------------------------------------------------------------

def safe_parse(raw) -> Dict[str, Any]:
    """Safely parse a value that may be a dict or a dict-like string.

    Uses ast.literal_eval instead of eval() for security.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip() not in ("{}", "", "nan", "None"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError, SyntaxError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def print_progress_bar(iteration: int, total: int, prefix: str = '',
                       suffix: str = '', length: int = 50, fill: str = '\u2588'):
    """Print a text progress bar to stderr."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model) -> int:
    """Count total parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Experiment logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Simple experiment logger that writes to both file and console."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment Log - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(log_entry.strip())

    def info(self, message: str):
        self.log(message, "INFO")

    def warning(self, message: str):
        self.log(message, "WARNING")

    def error(self, message: str):
        self.log(message, "ERROR")

    def success(self, message: str):
        self.log(message, "SUCCESS")


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_memory_requirements(
    num_samples: int,
    num_models: int = 3,
    max_new_tokens: int = 512,
    models: Optional[Dict] = None,
) -> Dict[str, float]:
    """Estimate memory requirements for the experiment.

    Args:
        num_samples: Number of test samples.
        num_models: Number of models to run.
        max_new_tokens: Maximum tokens generated per sample.
        models: MODELS config dict for accurate hidden dim estimation.
    """
    # Hidden dim defaults (bytes per token per layer, fp16)
    default_dims = {"1.5B": 1536, "3B": 2048, "7B": 3584}
    dims = []
    if models:
        for key, cfg in models.items():
            dims.append(cfg.get("hidden_dim", 1536))
    else:
        dims = [1536, 2048, 3584]

    avg_dim = sum(dims) / len(dims)
    # Per sample: max_new_tokens * num_layers * hidden_dim * 2 bytes (fp16)
    avg_layers = 30  # rough average
    per_sample_bytes = max_new_tokens * avg_layers * avg_dim * 2
    total_data_gb = (num_samples * num_models * per_sample_bytes) / (1024 ** 3)

    # Model weights (fp16)
    model_weight_gb = sum(d * 28 * 2 / (1024 ** 3) for d in dims)

    return {
        "per_sample_mb": round(per_sample_bytes / (1024 ** 2), 1),
        "model_weights_gb": round(model_weight_gb, 1),
        "data_storage_gb": round(total_data_gb, 1),
        "total_estimated_gb": round(model_weight_gb + total_data_gb, 1),
        "recommended_gpu_memory_gb": round(max(40, model_weight_gb * 1.5), 0),
        "recommended_disk_space_gb": round(max(100, total_data_gb * 2.5), 0),
    }


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate experiment configuration. Returns a list of error messages."""
    errors = []
    required_keys = ["MODELS", "DATA_CONFIG", "GENERATION_CONFIG",
                     "ANALYSIS_CONFIG", "INJECTION_CONFIG"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config key: {key}")

    if "MODELS" in config:
        for model_key, model_config in config["MODELS"].items():
            if "model_name" not in model_config:
                errors.append(f"Model '{model_key}' missing 'model_name'")
            if "hidden_dim" not in model_config:
                errors.append(f"Model '{model_key}' missing 'hidden_dim'")
            if "output_dir" not in model_config:
                errors.append(f"Model '{model_key}' missing 'output_dir'")

    return errors