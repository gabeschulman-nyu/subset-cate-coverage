"""CausalPFN's own ATE interval on ACIC 2016, as a check on the benchmark scoring.

benchmark_ladder rebuilds CausalPFN's aggregate interval from stored per-unit
draws. This calls the released code instead -- CATEEstimator's ATE interval --
on the same 40 datasets (settings 1-40, one replication each) and counts how
often it contains the true ATE, mean(mu1 - mu0). It runs with the temperature
calibration the stored draws used (calibrate=True) and with the API default
(calibrate=False).

Runs in the causalpfn environment, with bench_lib importable:
    SIM_LIB_DIR=<dir with bench_lib.py> python experiments/causalpfn_api_ate.py OUT.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("SIM_LIB_DIR", "benchmarks"))
from bench_lib import acic2016  # noqa: E402

SEEDS = range(1, 41)


def main(out_path: str) -> None:
    from causalpfn import CATEEstimator

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = []
    for calibrate in (True, False):
        est = CATEEstimator(device=device, verbose=False, calibrate=calibrate)
        t0 = time.time()
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            sim = acic2016(seed=seed)
            est.fit(sim.X, sim.z.astype(np.float32), sim.y.astype(np.float32))
            # estimate_ate_CI wraps this but raises KeyError('ate') in
            # causalpfn 0.1.4; the interval itself comes from here.
            ci = est._estimate_ate_cate_CI(sim.X, alpha=0.05)
            lo, hi = (float(np.asarray(ci[k]).squeeze())
                      for k in ("ate_lower_bound", "ate_upper_bound"))
            rows.append({"calibrate": calibrate, "seed": seed, "lo": lo, "hi": hi,
                         "truth": sim.true_ate, "covered": lo <= sim.true_ate <= hi,
                         "temperature": float(est.temperature)})
        hits = sum(r["covered"] for r in rows if r["calibrate"] == calibrate)
        print(f"calibrate={calibrate}: {hits}/{len(SEEDS)} cover "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
