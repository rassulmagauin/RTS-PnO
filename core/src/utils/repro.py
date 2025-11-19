# core/src/utils/repro.py
import os, random, numpy as np, torch

def set_deterministic(seed: int = 42, warn_only: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuBLAS matmul kernels
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Make BLAS threading stable (optional but recommended)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Torch-level determinism
    torch.use_deterministic_algorithms(not warn_only)
    if warn_only:
        # Torch >=2.0 allows warn-only mode to avoid hard errors
        torch.set_deterministic_debug_mode("warn_only")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Disable TF32 to avoid numeric drift on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
