"""
Configuration file for the heterogeneous model alignment experiment.

V4: Added Step 9 probability probing paths and early exit config.

V3: Optimized — removed unused config items, fixed hardcoded projection paths,
    aligned config with actual code usage, added dynamic model-pair support.

Key changes from V2:
  - Removed: answer_patterns (code uses qa_utils._PATTERNS_*), answer_search_tail_chars,
    use_cpu_offload (not implemented), top_k (unused)
  - Fixed: OUTPUT_PATHS now supports dynamic model pairs via helper function
  - Added: get_projection_paths() for model-pair-specific projection file naming
  - Added: cka_plot path actually referenced by code
  - Clarified: top_k_overlap is the B-ball top-k, entropy_threshold is in nats
"""

import os


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODELS = {
    "qwen1.5B": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_dim": 1536,
        "num_layers": 28,
        "output_dir": "experiment_results/model_outputs/qwen1.5B",
    },
    "qwen3B": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "hidden_dim": 2048,
        "num_layers": 36,
        "output_dir": "experiment_results/model_outputs/qwen3B",
    },
    "qwen7B": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "hidden_dim": 3584,
        "num_layers": 28,
        "output_dir": "experiment_results/model_outputs/qwen7B",
    },
}


# ---------------------------------------------------------------------------
# Data configurations
# ---------------------------------------------------------------------------

DATA_CONFIG = {
    "dataset_name": "gsm8k_arc_mmlu",
    "task_type": "qa",

    "datasets": {
        "gsm8k": {
            "hf_path": "openai/gsm8k",
            "hf_split": "test",
            "num_samples": 100,
            "answer_type": "numerical",
        },
        "arc_challenge": {
            "hf_path": "allenai/ai2_arc",
            "hf_config": "ARC-Challenge",
            "hf_split": "test",
            "num_samples": 100,
            "answer_type": "multiple_choice",
        },
        "mmlu": {
            "hf_path": "cais/mmlu",
            "hf_subjects": [
                "abstract_algebra", "college_physics", "high_school_government_and_politics",
                "high_school_computer_science", "high_school_statistics",
                "international_law", "philosophy", "prehistory",
                "professional_law", "security_studies",
            ],
            "hf_split": "test",
            "num_samples": 100,
            "answer_type": "multiple_choice",
        },
    },

    "sampled_data_path": "./experiment_results/sampled_data/sampled_300.json",
    "alignment_data_path": "./experiment_results/sampled_data/alignment_100.json",
    "total_samples": 300,
    "num_alignment_samples": 100,
    "random_seed": 42,
}


# ---------------------------------------------------------------------------
# Generation configurations
# ---------------------------------------------------------------------------

GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
    "return_dict_in_generate": True,
    "output_scores": True,
    "output_attentions": False,
}


# ---------------------------------------------------------------------------
# Memory optimization configurations
# ---------------------------------------------------------------------------

MEMORY_CONFIG = {
    "save_hidden_states": True,
    "save_full_logits": False,
    "save_top_k_logits": 100,
    "cleanup_frequency": 10,
    "save_prefill_hidden_states": True,
    "skip_embedding_layer": True,
}


# ---------------------------------------------------------------------------
# Analysis configurations
# ---------------------------------------------------------------------------

ANALYSIS_CONFIG = {
    "top_k_overlap": 20,           # Top-k for overlap analysis
    "entropy_threshold": 2.5,      # Shannon entropy threshold (nats)
    "error_analysis_output": "experiment_results/analysis/error_analysis.csv",
    "error_analysis_3B_vs_7B": "experiment_results/analysis/error_analysis_qwen3B_vs_qwen7B.csv",
    "error_analysis_1.5B_vs_7B": "experiment_results/analysis/error_analysis_qwen1.5B_vs_qwen7B.csv",
    "three_model_comparison": "experiment_results/analysis/three_model_comparison.csv",
    "cka_output_dir": "experiment_results/analysis",
    "prob_plots_dir": "experiment_results/analysis/prob_dist_plots",
}


# ---------------------------------------------------------------------------
# Injection experiment configurations
# ---------------------------------------------------------------------------

INJECTION_CONFIG = {
    "alpha_values": [0.1, 0.3, 0.5, 0.8],
    "injection_layers": [1, 2, 3, 4],   # Last N layers to test
    "projection_dir": "experiment_results/projection_matrices",
    "results_output": "experiment_results/analysis/injection_results.csv",
}


# ---------------------------------------------------------------------------
# Probing configurations
# ---------------------------------------------------------------------------

PROBING_CONFIG = {
    "divergence_threshold": 0.1,   # P(GT) gap threshold for divergence detection
    "top_k_probe": 10,             # Top-k tokens to record per layer
    "output_dir": "experiment_results/probing",
    "early_exit": {
        "gt_confidence_threshold": 0.3,  # P(GT) above this -> confident, no switch
        "switch_confidence_gap": 0.15,    # P(GT) gap below this -> not worth switching
    },
}


# ---------------------------------------------------------------------------
# Hardware configurations
# ---------------------------------------------------------------------------

HARDWARE_CONFIG = {
    "device": "cuda",
    "dtype": "bfloat16",
    "max_memory": {0: "78GB"},
}


# ---------------------------------------------------------------------------
# Output paths — static paths
# ---------------------------------------------------------------------------

OUTPUT_PATHS = {
    "summary_report": "experiment_results/summary_report.json",
    "cka_matrix_1.5B_vs_7B": "experiment_results/analysis/cka_matrix_1.5Bvs7B.npy",
    "cka_matrix_7B_vs_3B": "experiment_results/analysis/cka_matrix_7Bvs3B.npy",
    "cka_matrix_1.5B_vs_3B": "experiment_results/analysis/cka_matrix_1.5Bvs3B.npy",
    "cka_plot": "experiment_results/analysis/cka_curves.png",
    "experiment_summary_plot": "experiment_results/analysis/experiment_summary.png",
    "probing_dir": "experiment_results/probing",
    "probing_aggregate": "experiment_results/probing/probing_aggregate.png",
    "probing_individual": "experiment_results/probing/probing_individual.png",
    "probing_divergence": "experiment_results/probing/probing_divergence.png",
    "probing_entropy": "experiment_results/probing/probing_entropy.png",
    "probing_early_exit": "experiment_results/probing/probing_early_exit.png",
    "probing_results_json": "experiment_results/probing/probing_results.json",
    "probing_per_layer_csv": "experiment_results/probing/probing_per_layer.csv",
}


# ---------------------------------------------------------------------------
# Dynamic path helpers
# ---------------------------------------------------------------------------

def get_projection_paths(small_model: str, large_model: str) -> dict:
    """Get projection matrix and bias paths for a specific model pair.

    Args:
        small_model: Model key (e.g., "qwen1.5B", "qwen3B").
        large_model: Model key (e.g., "qwen7B").

    Returns:
        Dict with "projection_matrix", "projection_bias", "layer_comparison" paths.
    """
    base = INJECTION_CONFIG["projection_dir"]
    pair_tag = f"{small_model}_to_{large_model}"
    return {
        "projection_matrix": os.path.join(base, f"W_up_{pair_tag}.pt"),
        "projection_bias": os.path.join(base, f"bias_{pair_tag}.pt"),
        "layer_comparison": os.path.join(base, f"layer_comparison_{pair_tag}.json"),
    }


def get_error_analysis_path(small_model: str, large_model: str) -> str:
    """Get the error analysis CSV path for a specific model pair."""
    base_dir = os.path.dirname(ANALYSIS_CONFIG["error_analysis_output"])
    return os.path.join(base_dir, f"error_analysis_{small_model}_vs_{large_model}.csv")


def get_injection_results_path(small_model: str, large_model: str) -> str:
    """Get the injection results CSV path for a specific model pair."""
    base_dir = os.path.dirname(INJECTION_CONFIG["results_output"])
    return os.path.join(base_dir, f"injection_results_{small_model}_vs_{large_model}.csv")
