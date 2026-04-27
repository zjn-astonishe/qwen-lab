"""
Configuration file for the heterogeneous model alignment experiment
"""

# Model configurations
MODELS = {
    "qwen1.5B": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_dim": 1536,
        "output_dir": "experiment_results/model_outputs/qwen1.5B"
    },
    "qwen7B": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "hidden_dim": 3584,
        "output_dir": "experiment_results/model_outputs/qwen7B"
    },
    "qwen14B": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "hidden_dim": 5120,
        "output_dir": "experiment_results/model_outputs/qwen14B"
    }
}

# Data configurations
DATA_CONFIG = {
    "dataset_name": "BFCL_v3",
    "data_dir": "./data/bfcl_v3",
    "sampled_data_path": "./experiment_results/sampled_data/sampled_200.json",
    "alignment_data_path": "./experiment_results/sampled_data/alignment_100.json",
    "num_samples": 200,
    "num_alignment_samples": 100,
    "random_seed": 42,
    "min_turns": 2  # Minimum number of conversation turns
}

# Generation configurations
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.0,
    "do_sample": False,
    "return_dict_in_generate": True,
    "output_scores": True,
    "output_hidden_states": True,
    "output_attentions": False  # Set to True if needed
}

# Analysis configurations
ANALYSIS_CONFIG = {
    "top_k": 30,  # Top-k tokens to analyze
    "top_k_overlap": 20,  # For B-ball dilemma detection
    "entropy_threshold": 2.5,  # Threshold for high entropy (平权分布)
    "error_analysis_output": "experiment_results/analysis/error_analysis.csv",
    "cka_output_dir": "experiment_results/analysis",
    "prob_plots_dir": "experiment_results/analysis/prob_dist_plots"
}

# Injection experiment configurations
INJECTION_CONFIG = {
    "alpha_values": [0.1, 0.3, 0.5, 0.8],  # Mixing coefficients
    "injection_layers": [1, 2, 3, 4],  # Last N layers to test
    "projection_dir": "experiment_results/projection_matrices",
    "results_output": "experiment_results/analysis/injection_results.csv"
}

# Hardware configurations
HARDWARE_CONFIG = {
    "device": "cuda",
    "dtype": "float16",  # or "bfloat16"
    "max_memory": {0: "78GB"}  # Adjust based on your GPU
}

# Output paths
OUTPUT_PATHS = {
    "summary_report": "experiment_results/summary_report.json",
    "cka_matrix_1.5B_vs_7B": "experiment_results/analysis/cka_matrix_1.5Bvs7B.npy",
    "cka_matrix_7B_vs_14B": "experiment_results/analysis/cka_matrix_7Bvs14B.npy",
    "cka_matrix_1.5B_vs_14B": "experiment_results/analysis/cka_matrix_1.5Bvs14B.npy",
    "cka_plot": "experiment_results/analysis/cka_plot.png",
    "projection_matrix": "experiment_results/projection_matrices/W_up_1.5B_to_7B.pt",
    "projection_bias": "experiment_results/projection_matrices/bias_1.5B_to_7B.pt"
}
