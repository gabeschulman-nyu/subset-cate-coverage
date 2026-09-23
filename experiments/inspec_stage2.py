"""Section 1 stage 2 (causalpfn env, GPU): per-unit posterior draws."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from causalpfn import CATEEstimator

S = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    d = np.load(f"logs/inspec/tasks_{a.tag}.npz")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    est = CATEEstimator(device=device, calibrate=False, verbose=False)
    T, N = d["t"].shape
    draws = np.zeros((T, N, S), dtype=np.float32)
    t0 = time.time()
    for i in range(T):
        X, t, y = d["X"][i], d["t"][i], d["y"][i]
        est.fit(X, t, y)
        o0 = est._predict_cepo(X, t, y, X, np.zeros(N, dtype=X.dtype), 1.0, S)
        o1 = est._predict_cepo(X, t, y, X, np.ones(N, dtype=X.dtype), 1.0, S)
        s0 = o0[1] if isinstance(o0, tuple) else o0
        s1 = o1[1] if isinstance(o1, tuple) else o1
        draws[i] = (np.asarray(s1) - np.asarray(s0)).squeeze()
        if i % 16 == 0:
            print(f"{a.tag} {i}/{T} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"logs/inspec/draws_{a.tag}.npz", draws=draws)
    print(f"{a.tag} done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
