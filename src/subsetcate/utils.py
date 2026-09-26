import os
import random
import warnings

import numpy as np
import torch


def set_worker_single_thread():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")


def set_worker_random_seed(seed=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="little")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_available_cpus() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        try:
            return int(slurm_cpus)
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass

    return os.cpu_count() or 1
