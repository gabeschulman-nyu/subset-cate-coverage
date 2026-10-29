"""Train a subset head on frozen CausalPFN, on fresh prior tasks at every step.

CPU workers sample groups of prior tasks with equal (n, d); CausalPFN embeds
each group in one float32 pass on the GPU; the head takes a step on M subsets
per task against the exact theta_S. Every --eval_every steps a fixed held-out
set of prior tasks is scored (NLL, and coverage of nominal 95% intervals by
subset size next to CausalPFN's unit-level aggregation).

    python experiments/cfm/train_head.py --out_dir DIR [options]

The command for each trained head is in configs/cfm/<run>.sh.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalpfn_embed import cate_embeddings, head_logits, load_causalpfn  # noqa: E402
from prior import V2, PriorConfig, sample_task  # noqa: E402
from subset_head import (
    gaussian_nll,
    make_head,
    sample_subgroup_masks,
    unit_features,
    unit_level_interval,
)

BUCKETS = (1, 4, 16, 64, 256, "n", "subgroup")


class TaskStream(IterableDataset):
    """Endless groups of fresh tasks sharing (n, d); n log-uniform."""

    def __init__(self, cfg: PriorConfig, seed: int, rows: int, masks: int, p_random: float = 0.7, p_treated: float = 0.1):
        self.cfg, self.seed, self.rows, self.masks = cfg, seed, rows, masks
        self.p_random, self.p_treated = p_random, p_treated

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng([self.seed, wid])
        gen = torch.Generator().manual_seed(self.seed * 1000 + wid)
        lo, hi = self.cfg.n_range
        while True:
            n = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
            d = int(rng.integers(self.cfg.d_range[0], self.cfg.d_range[1] + 1))
            tasks = [sample_task(rng, self.cfg, n=n, d=d) for _ in range(max(1, self.rows // n))]
            # A real base table can have fewer than d columns. CausalPFN pads
            # every input with zero columns, so zero-padding does not change its output.
            w = max(t.X.shape[1] for t in tasks)
            Xs = [np.pad(t.X, ((0, 0), (0, w - t.X.shape[1]))) for t in tasks]
            out = {"X": torch.from_numpy(np.stack(Xs).astype(np.float32))}
            out.update({k: torch.from_numpy(np.stack([getattr(t, k) for t in tasks]).astype(np.float32))
                        for k in ("t", "y", "tau")})
            out["masks"] = torch.stack([sample_subgroup_masks(out["X"][b], out["t"][b], self.masks, gen,
                                                              p_random=self.p_random, p_treated=self.p_treated)
                                        for b in range(len(tasks))])
            yield out


def batch_unit_features(icl, X, t, y):
    """u [B, n, 770] for a group of equal-shape tasks, in one pass."""
    dev = X.device
    n = X.shape[1]
    Xq = torch.cat([X, X], 1)
    tq = torch.cat([torch.zeros_like(t), torch.ones_like(t)], 1)
    cap = {}
    hook = icl.model.head.register_forward_hook(lambda m, i, o: cap.__setitem__("h", i[0]))
    try:
        icl.predict_cepo(X_context=X, t_context=t, y_context=y, X_query=Xq, t_query=tq,
                         temperature=torch.tensor([1.0], device=dev))
    finally:
        hook.remove()
    h = cap["h"][n:].transpose(0, 1)                                   # [B, 2n, E]
    p = torch.softmax(head_logits(icl, h), -1)
    c = icl.bin_centers
    m = (p * c).sum(-1)
    v = ((p * c ** 2).sum(-1) - m ** 2).clamp(min=0)
    s0, sc0, s1, sc1 = icl._get_y0_y1_shift_scale(t, y)                # each [B, 1]
    cate = (m[:, n:] * sc1 + s1) - (m[:, :n] * sc0 + s0)
    sd = torch.sqrt(v[:, :n] * sc0 ** 2 + v[:, n:] * sc1 ** 2)
    s_y = y.std(dim=1, keepdim=True)
    u = torch.cat([h[:, :n], h[:, n:], (cate / s_y)[..., None], (sd / s_y)[..., None]], -1)
    return u, s_y[:, 0]


def build_validation(icl, cfg, n_tasks, seed, dev, gen):
    """Held-out tasks with fixed subsets and CausalPFN's unit-level intervals."""
    rng = np.random.default_rng(seed)
    ugen = torch.Generator(device=dev).manual_seed(seed)   # CausalPFN's per-unit draws
    val = []
    for _ in range(n_tasks):
        tk = sample_task(rng, cfg)
        n = tk.meta["n"]
        e = cate_embeddings(icl, tk.X.astype(np.float32), tk.t, tk.y)
        s_y = float(tk.y.std())
        u = unit_features(e, s_y, icl.bin_centers)
        masks, labels = [], []
        for k in BUCKETS[:-1]:
            kk = n if k == "n" else min(k, n)
            reps = 1 if kk == n else 8
            ranks = torch.rand(reps, n, generator=gen).argsort(1).argsort(1)
            masks.append((ranks < kk).to(dev))
            labels += [k] * reps
        X = torch.as_tensor(tk.X, dtype=torch.float32)
        t = torch.as_tensor(tk.t, dtype=torch.float32)
        sub = sample_subgroup_masks(X, t, 8, gen, p_random=0.0, p_treated=0.3).to(dev)
        masks.append(sub)
        labels += ["subgroup"] * sub.shape[0]
        mask = torch.cat(masks)
        tau = torch.as_tensor(tk.tau, dtype=torch.float32, device=dev)
        theta = (mask.float() @ tau) / mask.sum(1)
        lo, hi = unit_level_interval(e, icl.bin_centers, mask, 500, ugen)
        val.append(dict(u=u, mask=mask, theta=theta, s_y=s_y, labels=labels, tau=tau / s_y,
                        unit_cover=((lo <= theta) & (theta <= hi)).float(), unit_width=(hi - lo) / s_y))
    return val


@torch.no_grad()
def evaluate(head, val, latent: bool):
    """Held-out NLL and, per subset-size bucket, coverage and median width
    (sd(y) units) of the head's interval (SCM-marginalized for the latent head),
    of CausalPFN's unit-level interval, and of the latent head's unit-level
    interval."""
    head.eval()
    keys = ("head", "head_unit", "unit")
    cov = {k: {b: [] for b in BUCKETS} for k in keys}
    wid = {k: {b: [] for b in BUCKETS} for k in keys}
    nll = []
    for v in val:
        target = v["theta"] / v["s_y"]
        if latent:
            nll.append(float(head.loss(v["u"][None], v["mask"][None], v["tau"][None], target[None], K=64)[0]))
            lo, hi, ulo, uhi = head.intervals(v["u"], v["mask"])
            ints = {"head": (lo, hi), "head_unit": (ulo, uhi)}
        else:
            mean, sd = head(v["u"], v["mask"])
            nll.append(float(gaussian_nll(mean, sd, target)))
            ints = {"head": (mean - 1.96 * sd, mean + 1.96 * sd)}
        for name, (lo, hi) in ints.items():
            inside = (lo <= target) & (target <= hi)
            for i, lab in enumerate(v["labels"]):
                cov[name][lab].append(float(inside[i]))
                wid[name][lab].append(float(hi[i] - lo[i]))
        for i, lab in enumerate(v["labels"]):
            cov["unit"][lab].append(float(v["unit_cover"][i]))
            wid["unit"][lab].append(float(v["unit_width"][i]))
    head.train()
    rec = dict(val_nll=float(np.mean(nll)))
    for name in keys:
        if cov[name][BUCKETS[0]]:
            rec[f"{name}_cov"] = {str(b): float(np.mean(cov[name][b])) for b in BUCKETS}
            rec[f"{name}_width"] = {str(b): float(np.median(wid[name][b])) for b in BUCKETS}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--rows", type=int, default=8192, help="context rows per GPU pass")
    ap.add_argument("--masks", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip (0 = off)")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--head", choices=["cnp", "latent"], default="cnp")
    ap.add_argument("--latent_output", choices=["gaussian", "histogram"], default="gaussian",
                    help="latent head's per-unit output; histogram = the paper's latent NP")
    ap.add_argument("--hist_nbins", type=int, default=64)
    ap.add_argument("--hist_support_min", type=float, default=-4.0)
    ap.add_argument("--hist_support_max", type=float, default=4.0)
    ap.add_argument("--hist_smoothing", type=float, default=1.0, help="soft-target sigma in bins")
    ap.add_argument("--mask_p_random", type=float, default=0.7,
                    help="share of training subsets that are uniform random (log-uniform size); the rest are "
                         "treated-only and covariate-defined. 1.0 = the paper's random subsets only")
    ap.add_argument("--mask_p_treated", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=16, help="latent draws per training step")
    ap.add_argument("--lam", type=float, default=2.0, help="weight of the subset term (latent)")
    ap.add_argument("--val_tasks", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_range", type=int, nargs=2, default=(100, 2048))
    ap.add_argument("--d_range", type=int, nargs=2, default=(1, 60))
    ap.add_argument("--openml_prob", type=float, default=0.0)
    ap.add_argument("--openml_pack", default=None)
    ap.add_argument("--outcome_transform_prob", type=float, default=0.0)
    ap.add_argument("--prior_preset", choices=["v1", "v2"], default="v1",
                    help="v2 adds the prior's v2 mechanisms at their pre-specified rates (prior.V2)")
    a = ap.parse_args()
    cfg = PriorConfig(n_range=tuple(a.n_range), d_range=tuple(a.d_range),
                      openml_prob=a.openml_prob, openml_pack=a.openml_pack,
                      outcome_transform_prob=a.outcome_transform_prob,
                      **(V2 if a.prior_preset == "v2" else {}))
    prior_config = dataclasses.asdict(cfg)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({**vars(a), "prior_config": prior_config}, open(os.path.join(a.out_dir, "args.json"), "w"), indent=1)

    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    icl = load_causalpfn("cuda")
    for p in icl.parameters():
        p.requires_grad_(False)
    # CausalPFN stays in eval mode; its forward normalizes inputs differently
    # in training mode.

    gen = torch.Generator().manual_seed(a.seed + 7)
    t0 = time.time()
    val = build_validation(icl, cfg, a.val_tasks, seed=10_000 + a.seed, dev=dev, gen=gen)
    print(f"validation: {len(val)} tasks in {time.time() - t0:.0f} s", flush=True)

    latent = a.head == "latent"
    head = make_head(vars(a))[0].to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr / 100)
    loader = DataLoader(TaskStream(cfg, a.seed, a.rows, a.masks, a.mask_p_random, a.mask_p_treated),
                        batch_size=None, num_workers=a.workers,
                        prefetch_factor=4, persistent_workers=True)
    log = open(os.path.join(a.out_dir, "log.jsonl"), "a")
    best = math.inf
    t0, seen, skipped = time.time(), 0, 0
    for step, batch in enumerate(loader):
        if step >= a.steps:
            break
        X, t, y, tau, masks = (batch[k].to(dev, non_blocking=True) for k in ("X", "t", "y", "tau", "masks"))
        with torch.no_grad():
            u, s_y = batch_unit_features(icl, X, t, y)
        if not (torch.isfinite(u).all() and torch.isfinite(s_y).all()):
            skipped += 1
            continue
        theta = (masks.float() @ tau[..., None])[..., 0] / masks.sum(-1) / s_y[:, None]
        if latent:
            loss, _ = head.loss(u, masks, tau / s_y[:, None], theta, K=a.K, lam=a.lam)
        else:
            mean, sd = head(u, masks)
            loss = gaussian_nll(mean, sd, theta)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        opt.zero_grad()
        loss.backward()
        # A narrow interval with a large miss gives a large NLL gradient.
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), a.clip)
        opt.step()
        sched.step()
        seen += X.shape[0]
        if step % a.eval_every == 0 or step == a.steps - 1:
            ev = evaluate(head, val, latent)
            rate = seen / (time.time() - t0)
            rec = dict(step=step, train_nll=float(loss), tasks_seen=seen, tasks_per_s=rate,
                       skipped_batches=skipped, **ev)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            hc, uc = ev["head_cov"], ev["unit_cov"]
            print(f"step {step:6d} | {seen:8d} tasks, {rate:6.1f}/s | train {float(loss):.3f} val {ev['val_nll']:.3f} | "
                  "cov head/unit " + " ".join(f"{b}:{hc[str(b)]:.2f}/{uc[str(b)]:.2f}" for b in BUCKETS), flush=True)
            state = dict(head=head.state_dict(), args=vars(a), prior_config=prior_config, step=step, eval=ev)
            torch.save(state, os.path.join(a.out_dir, "last.pt"))
            if ev["val_nll"] < best:
                best = ev["val_nll"]
                torch.save(state, os.path.join(a.out_dir, "best.pt"))
    print(f"done: {seen} tasks in {(time.time() - t0) / 60:.1f} min; best val NLL {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
