"""Dump CausalPFN's per-unit predictive distributions for PIT scoring.

For each DGP and seed, fits CausalPFN and writes one compressed .npz to
results_pfn_pit/<tag>/parts/:
    mu0, mu1            (n,)        predictive mean of Y(0), Y(1)
    samples0, samples1  (n, S)      samples from the predictive distribution
    logits0, logits1    (n, nbins)  histogram logits divided by the temperature
    bin_centers0/1      (n, nbins)  bin centers in original Y units
    y0_true, y1_true, tau_true, X, t, y, p_treat, true_ate, true_att,
    temperature, dgp, seed

Usage:
    python run_causalpfn_pit.py <n_sim> <tag> [dgps] [calibrate] [n_samples] [save_logits]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from sim_lib import DGPS


def run_one(dgp_name, seed, cate, n_samples, save_logits, out_path):
    sim = DGPS[dgp_name](seed=seed)
    X, T, Y = sim.X, sim.z.astype(np.float32), sim.y.astype(np.float32)
    cate.fit(X, T, Y)

    icl = cate.icl_model
    n = X.shape[0]
    X_q = np.concatenate([X, X], axis=0)
    t_q = np.concatenate([np.zeros(n, dtype=X.dtype), np.ones(n, dtype=X.dtype)], axis=0)

    captured = []
    orig_predict_cepo = icl.predict_cepo

    def patched(X_context, t_context, y_context, X_query, t_query,
                n_samples=None, temperature=None):
        y0_shift, y0_scale, y1_shift, y1_scale = icl._get_y0_y1_shift_scale(t_context, y_context)
        y_std = torch.where(
            t_context == 1, (y_context - y1_shift) / y1_scale,
            (y_context - y0_shift) / y0_scale,
        )
        xt_c = torch.cat([t_context.unsqueeze(-1), X_context], dim=2)
        xt_q = torch.cat([t_query.unsqueeze(-1), X_query], dim=2)
        xt = torch.cat([xt_c, xt_q], dim=1)
        x_src, y_src = icl.prepare_input(xt, y_std)
        logits = icl.model(x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(0, 1)
        logits = logits[:, :, -icl.model.nbins:]
        T_val = float(temperature.item()) if temperature is not None else 1.0
        logits_T = logits / T_val

        per_q_shift = torch.where(t_query == 1, y1_shift, y0_shift)
        per_q_scale = torch.where(t_query == 1, y1_scale, y0_scale)
        captured.append((
            logits_T.squeeze(0).detach().cpu().numpy(),
            icl.bin_centers.detach().cpu().numpy(),
            per_q_shift.squeeze(0).detach().cpu().numpy(),
            per_q_scale.squeeze(0).detach().cpu().numpy(),
            X_query.squeeze(0).detach().cpu().numpy(),
            t_query.squeeze(0).detach().cpu().numpy(),
        ))
        return orig_predict_cepo(X_context=X_context, t_context=t_context,
                                 y_context=y_context, X_query=X_query,
                                 t_query=t_query, n_samples=n_samples,
                                 temperature=temperature)

    icl.predict_cepo = patched
    try:
        mu_full, samples_full = cate._predict_cepo(
            X_context=cate.X_train, t_context=cate.t_train, y_context=cate.y_train,
            X_query=X_q, t_query=t_q, temperature=cate.temperature, n_samples=n_samples,
        )
    finally:
        icl.predict_cepo = orig_predict_cepo

    mu0, mu1 = mu_full[:n], mu_full[n:]
    samples0, samples1 = samples_full[:n], samples_full[n:]

    out = dict(
        X=X, t=T, y=Y,
        mu0=mu0.astype(np.float32), mu1=mu1.astype(np.float32),
        samples0=samples0.astype(np.float32), samples1=samples1.astype(np.float32),
        y0_true=sim.y0.astype(np.float32), y1_true=sim.y1.astype(np.float32),
        tau_true=sim.tau.astype(np.float32),
        p_treat=sim.p_treat.astype(np.float32),
        true_ate=np.float32(sim.true_ate),
        true_att=np.float32(sim.true_att),
        temperature=np.float32(cate.temperature),
        dgp=dgp_name, seed=seed,
    )

    if save_logits and captured:
        nbins = captured[0][0].shape[1]
        logits_buf = np.zeros((2 * n, nbins), dtype=np.float32)
        shift_buf = np.zeros((2 * n,), dtype=np.float32)
        scale_buf = np.zeros((2 * n,), dtype=np.float32)
        filled = np.zeros((2 * n,), dtype=bool)
        key_to_rows = {}
        for i in range(2 * n):
            key = (X_q[i].tobytes(), float(t_q[i]))
            key_to_rows.setdefault(key, []).append(i)
        for lg, _bc, sh, sc, xq, tq in captured:
            for j in range(xq.shape[0]):
                key = (xq[j].astype(X_q.dtype).tobytes(), float(tq[j]))
                if key in key_to_rows and key_to_rows[key]:
                    i = key_to_rows[key].pop(0)
                    logits_buf[i] = lg[j]
                    shift_buf[i] = sh[j]
                    scale_buf[i] = sc[j]
                    filled[i] = True
        if not filled.all():
            print(f"  warn: {(~filled).sum()}/{2*n} rows unmatched", flush=True)
        bc_std = captured[0][1]
        out["logits0"] = logits_buf[:n]
        out["logits1"] = logits_buf[n:]
        out["bin_centers0"] = (bc_std[None, :] * scale_buf[:n, None] + shift_buf[:n, None]).astype(np.float32)
        out["bin_centers1"] = (bc_std[None, :] * scale_buf[n:, None] + shift_buf[n:, None]).astype(np.float32)

    np.savez_compressed(out_path, **out)


def main():
    if len(sys.argv) < 3:
        print("Usage: run_causalpfn_pit.py <n_sim> <tag> [dgps] [calibrate] [n_samples] [save_logits]",
              file=sys.stderr)
        return 2
    n_sim = int(sys.argv[1])
    tag = sys.argv[2]
    dgps = sys.argv[3].split(",") if len(sys.argv) > 3 else ["DGP1", "DGP2"]
    calibrate = (len(sys.argv) > 4 and sys.argv[4].lower() in ("1", "true", "yes", "calibrate"))
    n_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    save_logits = (len(sys.argv) <= 6) or sys.argv[6].lower() in ("1", "true", "yes")

    out_dir = Path("results_pfn_pit") / tag
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} n_sim={n_sim} tag={tag} dgps={dgps} calibrate={calibrate} "
          f"n_samples={n_samples} save_logits={save_logits}", flush=True)

    from causalpfn import CATEEstimator
    cate = CATEEstimator(device=device, verbose=False, calibrate=calibrate)

    for dgp_name in dgps:
        t0 = time.time()
        for rep in range(1, n_sim + 1):
            seed = rep
            out_path = parts_dir / f"{dgp_name}_seed{seed:05d}.npz"
            if out_path.exists():
                continue
            try:
                run_one(dgp_name, seed, cate, n_samples, save_logits, out_path)
            except Exception as e:
                print(f"[{dgp_name} seed={seed}] FAILED: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue
            if rep % 5 == 0 or rep == n_sim:
                elapsed = time.time() - t0
                print(f"[{dgp_name}] {rep}/{n_sim} in {elapsed/60:.1f} min "
                      f"({elapsed/rep:.2f} s/rep)", flush=True)
    print(f"Artifacts in {out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
