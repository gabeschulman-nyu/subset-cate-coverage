"""Per-unit embeddings from the released CausalPFN, for heads trained on top of it.

CausalPFN (causalpfn==0.1.4) is a 20-layer, width-384 row transformer
(TabDPTLongContextModel, 18.8M parameters). Every row is one token: the
treatment is the first feature column, context rows also carry their
standardized outcome, and in every layer keys and values come from the
context rows only. A query row therefore sees the context and nothing else.
Its last hidden state -- the input to `model.head` -- is the embedding h(x, t)
used here; the head maps it to 1,024 logits over [-10, 10] in outcome units
standardized per treatment arm.

This module runs the model on the FULL context in one pass, so every query
unit is conditioned on the same data. The public API does not always do that:
`CausalEstimator._predict_cepo` gives each window of query units the nearest
neighbours of a gradient-boosting CATE estimate, at most 1,024 per arm by
default. When an arm has at most 1,024 rows every query sees the whole arm,
and the two paths agree up to a faiss padding quirk (a short arm's neighbour
list is padded with -1, which duplicates its last row). Above that, the API
conditions units on different context subsets; see `verify_embeddings.py`.

Runs in the causalpfn environment (environment-causalpfn.yml).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def load_causalpfn(device: str = "cpu"):
    """The released model, loaded exactly as the public API loads it."""
    from causalpfn import CATEEstimator

    est = CATEEstimator(device=device, verbose=False)
    est.load_model()
    icl = est.icl_model
    icl.eval()
    return icl


@dataclass
class CEPOEmbedding:
    h: torch.Tensor          # [Q, 384] last hidden state of each query row
    logits: torch.Tensor     # [Q, nbins] temperature-1 logits (standardized units)
    mean: torch.Tensor       # [Q] predicted CEPO in raw outcome units
    shift: torch.Tensor      # [Q] per-row arm shift (mean of that arm's context y)
    scale: torch.Tensor      # [Q] per-row arm scale (sd of that arm's context y)


def head_logits(icl, h: torch.Tensor) -> torch.Tensor:
    """The model's own map from last hidden state to histogram logits."""
    m = icl.model
    pred = m.head(h)
    pred = 30 * torch.tanh(pred / (7.5 * h.size(-1) ** 0.5))
    return pred[..., -m.nbins:]


@torch.no_grad()
def embed_cepo(icl, X_ctx, t_ctx, y_ctx, X_q, t_q) -> CEPOEmbedding:
    """One full-context pass; returns embeddings and the model's own outputs.

    X_ctx [n, d], t_ctx [n], y_ctx [n], X_q [Q, d], t_q [Q] as numpy or torch;
    d must be at most 99 (the API reduces wider X with a truncated SVD first).
    """
    dev = next(icl.parameters()).device
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev).unsqueeze(0)
    X_ctx, t_ctx, y_ctx, X_q, t_q = map(f, (X_ctx, t_ctx, y_ctx, X_q, t_q))
    if X_ctx.shape[-1] > icl.model.num_features - 1:
        raise ValueError(f"at most {icl.model.num_features - 1} covariates; reduce X first")

    captured = {}
    hook = icl.model.head.register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("h", inp[0]))
    try:
        mean = icl.predict_cepo(X_context=X_ctx, t_context=t_ctx, y_context=y_ctx,
                                X_query=X_q, t_query=t_q,
                                temperature=torch.tensor([1.0], device=dev))
    finally:
        hook.remove()
    n = X_ctx.shape[1]
    h = captured["h"][n:, 0, :]                      # sequence-first: [L, B, E]
    logits = head_logits(icl, h)
    y0_shift, y0_scale, y1_shift, y1_scale = icl._get_y0_y1_shift_scale(t_ctx, y_ctx)
    tq = t_q[0]
    shift = torch.where(tq == 1, y1_shift[0, 0], y0_shift[0, 0])
    scale = torch.where(tq == 1, y1_scale[0, 0], y0_scale[0, 0])
    return CEPOEmbedding(h=h, logits=logits, mean=mean.reshape(-1), shift=shift, scale=scale)


def cate_embeddings(icl, X, t, y, X_q=None):
    """h(x_i, 0) and h(x_i, 1) for every query unit (default: the context units)."""
    X_q = X if X_q is None else X_q
    Q = len(X_q)
    X2 = np.concatenate([X_q, X_q])
    t2 = np.concatenate([np.zeros(Q), np.ones(Q)])
    e = embed_cepo(icl, X, t, y, X2, t2)
    return {
        "h0": e.h[:Q], "h1": e.h[Q:],
        "logits0": e.logits[:Q], "logits1": e.logits[Q:],
        "mu0": e.mean[:Q], "mu1": e.mean[Q:],
        "scale0": e.scale[:Q], "scale1": e.scale[Q:],
        "shift0": e.shift[:Q], "shift1": e.shift[Q:],
    }
