"""
Utility Functions
Common helper functions used across the experiment
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, List
from datetime import datetime


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_file_size(path: str) -> int:
    """Get file size in bytes"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    return 0


def save_json(data: Dict[str, Any], path: str, indent: int = 2):
    """Save data as JSON with proper formatting"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON data"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {}
    
    usage = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        usage[f"gpu_{i}"] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization": allocated / total if total > 0 else 0
        }
    
    return usage


def create_experiment_metadata() -> Dict[str, Any]:
    """Create metadata for the current experiment"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "pytorch_version": torch.__version__ if torch else "N/A",
        "cuda_available": torch.cuda.is_available() if torch else False,
        "gpu_info": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9
            }
            metadata["gpu_info"].append(gpu_info)
    
    return metadata


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                      length: int = 50, fill: str = '█'):
    """Print a progress bar"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def count_parameters(model) -> int:
    """Count total parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """Count trainable parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExperimentLogger:
    """Simple experiment logger"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize log file
        with open(log_file, 'w') as f:
            f.write(f"Experiment Log - {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def info(self, message: str):
        """Log info message"""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log warning message"""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log error message"""
        self.log(message, "ERROR")
    
    def success(self, message: str):
        """Log success message"""
        self.log(message, "SUCCESS")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate experiment configuration"""
    errors = []
    
    # Check required keys
    required_keys = ["MODELS", "DATA_CONFIG", "GENERATION_CONFIG", 
                    "ANALYSIS_CONFIG", "INJECTION_CONFIG"]
    
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config key: {key}")
    
    # Validate model configs
    if "MODELS" in config:
        for model_key, model_config in config["MODELS"].items():
            if "model_name" not in model_config:
                errors.append(f"Model {model_key} missing 'model_name'")
            if "hidden_dim" not in model_config:
                errors.append(f"Model {model_key} missing 'hidden_dim'")
    
    return errors


def estimate_memory_requirements(num_samples: int, num_models: int = 3) -> Dict[str, float]:
    """Estimate memory requirements for the experiment"""
    
    # Rough estimates based on model sizes
    model_sizes_gb = {
        "1.5B": 3.5,
        "7B": 14.0,
        "14B": 28.0
    }
    
    # Per sample storage (hidden states, outputs)
    per_sample_mb = 50  # Approximate
    
    total_model_memory = sum(model_sizes_gb.values())
    total_data_memory = (num_samples * per_sample_mb * num_models) / 1024  # GB
    
    return {
        "model_memory_gb": total_model_memory,
        "data_memory_gb": total_data_memory,
        "total_estimated_gb": total_model_memory + total_data_memory,
        "recommended_gpu_memory_gb": max(80, total_model_memory * 1.5),
        "recommended_disk_space_gb": max(200, total_data_memory * 2)
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    print(f"Time format: {format_time(3725)}")
    print(f"Size format: {format_size(1234567890)}")
    
    mem_est = estimate_memory_requirements(200, 3)
    print("\nMemory estimation for 200 samples:")
    for key, value in mem_est.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✓ Utilities test passed")
