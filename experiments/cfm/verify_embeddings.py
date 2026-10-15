"""Check that the hooked embeddings are the model's full per-unit state.

1. Exactness: logits recomputed from h with the model's own head reproduce
   the model's predicted CEPOs.
2. Agreement with the public API on a benchmark whose arms have at most
   1,024 rows, where the API's neighbour contexts cover the whole arm.

    SIM_LIB_DIR=<dir with sim_lib.py> python experiments/cfm/verify_embeddings.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SIM_LIB_DIR", "benchmarks"))
from causalpfn_embed import cate_embeddings, head_logits, load_causalpfn  # noqa: E402
from sim_lib import DGPS  # noqa: E402


def main():
    torch.manual_seed(0)
    icl = load_causalpfn("cpu")
    print(f"CausalPFN: {sum(p.numel() for p in icl.model.parameters()) / 1e6:.2f}M parameters, "
          f"{icl.model.nlayers} layers, width {icl.model.ninp}, {icl.model.nbins} bins")

    for dgp in ("HillA", "FriedmanConf"):
        sim = DGPS[dgp](seed=1)
        X, t, y = sim.X.astype(np.float32), sim.z.astype(np.float32), sim.y.astype(np.float32)
        e = cate_embeddings(icl, X, t, y)

        # 1. exactness of the head map
        worst = 0.0
        for arm in (0, 1):
            lg = head_logits(icl, e[f"h{arm}"])
            m = (torch.softmax(lg, -1) * icl.bin_centers).sum(-1)
            raw = m * e[f"scale{arm}"] + e[f"shift{arm}"]
            worst = max(worst, float((raw - e[f"mu{arm}"]).abs().max()))
            assert torch.equal(lg, e[f"logits{arm}"])
        cate_full = (e["mu1"] - e["mu0"]).numpy()

        # 2. the public API
        from causalpfn import CATEEstimator
        est = CATEEstimator(device="cpu", verbose=False)
        est.fit(X, t, y)
        cate_api = np.asarray(est.estimate_cate(X))
        d = np.abs(cate_api - cate_full)
        arms = (int((t == 0).sum()), int((t == 1).sum()))
        print(f"{dgp:13s} n={len(X)} arms={arms} d={X.shape[1]} | h: {tuple(e['h0'].shape)} | "
              f"head map reproduces CEPO to {worst:.1e} | full-context vs API CATE: "
              f"max {d.max():.4f}, median {np.median(d):.5f} (CATE sd {cate_full.std():.3f}) | "
              f"PEHE vs truth: full {np.sqrt(np.mean((cate_full - sim.tau) ** 2)):.3f}, "
              f"API {np.sqrt(np.mean((cate_api - sim.tau) ** 2)):.3f}")


if __name__ == "__main__":
    main()
