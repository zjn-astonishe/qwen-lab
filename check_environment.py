"""
Environment Checker
Verify system requirements and dependencies before running experiments
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Requires 3.10+)")
        return False


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("✗ CUDA not available (CPU mode will be very slow)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")
    try:
        stat = os.statvfs('.')
        free_space_gb = (stat.f_bavail * stat.f_frsize) / 1e9
        
        if free_space_gb > 200:
            print(f"✓ Available disk space: {free_space_gb:.1f} GB")
            return True
        elif free_space_gb > 100:
            print(f"⚠ Available disk space: {free_space_gb:.1f} GB (Recommended: 200+ GB)")
            return True
        else:
            print(f"✗ Available disk space: {free_space_gb:.1f} GB (Insufficient)")
            return False
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True


def check_dependencies():
    """Check required Python packages"""
    print("\nChecking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm')
    ]
    
    all_installed = True
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (Not installed)")
            all_installed = False
    
    return all_installed


def check_directory_structure():
    """Check if directory structure exists"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        'data',
        'models',
        'experiment_results',
        'experiment_results/sampled_data',
        'experiment_results/model_outputs',
        'experiment_results/analysis',
        'experiment_results/projection_matrices'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"⚠ {dir_path} (Will be created)")
    
    return True


def check_scripts():
    """Check if all required scripts exist"""
    print("\nChecking experiment scripts...")
    
    required_scripts = [
        'config.py',
        'step1_prepare_data.py',
        'step2_run_inference.py',
        'step3_error_analysis.py',
        'step4_cka_analysis.py',
        'step5_train_projection.py',
        'step6_injection_experiment.py',
        'step7_visualization.py',
        'step8_summary.py',
        'run_experiment.py'
    ]
    
    all_exist = True
    for script in required_scripts:
        if os.path.exists(script):
            print(f"✓ {script}")
        else:
            print(f"✗ {script} (Missing)")
            all_exist = False
    
    return all_exist


def estimate_requirements():
    """Estimate resource requirements"""
    print("\n" + "="*60)
    print("ESTIMATED RESOURCE REQUIREMENTS")
    print("="*60)
    print("\nFor full experiment (200 samples, 3 models):")
    print("  - GPU Memory: ~60-80 GB (A100 80GB recommended)")
    print("  - Disk Space: ~200 GB")
    print("  - RAM: ~64 GB")
    print("  - Time: ~4-8 hours (depending on hardware)")
    
    print("\nFor quick test (10 samples, synthetic data):")
    print("  - GPU Memory: ~20 GB")
    print("  - Disk Space: ~50 GB")
    print("  - RAM: ~32 GB")
    print("  - Time: ~30-60 minutes")


def main():
    print("="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    checks = []
    
    checks.append(("Python Version", check_python_version()))
    checks.append(("GPU Availability", check_gpu()))
    checks.append(("Disk Space", check_disk_space()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("Directory Structure", check_directory_structure()))
    checks.append(("Scripts", check_scripts()))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    for name, status in checks:
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{name:.<40} {status_str}")
    
    print(f"\nChecks passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ Environment is ready!")
        print("\nYou can now run the experiment:")
        print("  python run_experiment.py --all --use_synthetic --max_samples 10")
    else:
        print("\n✗ Some checks failed. Please resolve issues before running experiments.")
        if not checks[3][1]:  # Dependencies check failed
            print("\nTo install dependencies, run:")
            print("  pip install -r requirements.txt")
    
    estimate_requirements()


if __name__ == "__main__":
    main()
