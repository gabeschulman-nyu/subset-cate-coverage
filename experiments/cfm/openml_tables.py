"""Fetch CausalPFN's OpenML base tables as numeric covariate tables.

The catalogue is assets/openml_catalogue.csv in CausalPFN's repository (337
OpenML datasets). Per dataset: download the Parquet file through OpenML's REST
API; drop ignored, row-identifier and ID-like columns and columns more than
half missing; encode categoricals as integer codes; impute with the column
median; drop constant columns; subsample to at most --max_rows rows and
--max_cols columns (seeded by dataset id); save float32 X as <out>/<did>.npz
with the column names and OpenML metadata. Datasets whose name matches a
benchmark covariate source (IHDP / Infant Health and Development, ACIC /
Collaborative Perinatal Project, LaLonde, Twins, Jobs) are excluded and listed.

Runs in the cfmdata environment (pyarrow, pandas, requests):
    python experiments/cfm/openml_tables.py CATALOGUE.csv OUT_DIR
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

API = "https://www.openml.org/api/v1/json"
EXCLUDE = re.compile(r"ihdp|infant|\bacic|perinatal|\bcpp\b|lalonde|nsw|psid|twins|\bjobs\b|causal", re.I)


def get_json(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def get_bytes(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            return r.content
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def process(did, out, max_rows, max_cols):
    path = os.path.join(out, f"{did}.npz")
    if os.path.exists(path):
        return dict(did=did, status="cached")
    desc = get_json(f"{API}/data/{did}")["data_set_description"]
    name = desc.get("name", "")
    if EXCLUDE.search(name):
        return dict(did=did, name=name, status="excluded")
    try:
        feats = get_json(f"{API}/data/features/{did}")["data_features"]["feature"]
        drop = {f["name"] for f in feats if f.get("is_ignore") == "true" or f.get("is_row_identifier") == "true"}
    except requests.HTTPError:
        drop = set()    # no feature metadata; the ID-like check below still applies
    df = pd.read_parquet(io.BytesIO(get_bytes(desc["parquet_url"])))
    df = df.drop(columns=[c for c in df.columns if c in drop])
    rng = np.random.default_rng(did)
    if len(df) > max_rows:
        df = df.iloc[np.sort(rng.choice(len(df), max_rows, replace=False))].reset_index(drop=True)
    cols, names = [], []
    for c in df.columns:
        s = df[c]
        if s.isna().mean() > 0.5:
            continue
        if s.dtype.kind in "biuf":
            v = s.astype(float).to_numpy()
        else:
            if s.nunique(dropna=True) > 0.9 * len(s):          # ID-like strings
                continue
            v = s.astype("category").cat.codes.astype(float).to_numpy()
            v[v < 0] = np.nan
        if np.all(np.isnan(v)):
            continue
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        if np.nanstd(v) == 0:
            continue
        cols.append(v)
        names.append(str(c))
    if len(cols) < 2:
        return dict(did=did, name=name, status="too few columns")
    X = np.stack(cols, 1).astype(np.float32)
    if X.shape[1] > max_cols:
        keep = np.sort(rng.choice(X.shape[1], max_cols, replace=False))
        X, names = X[:, keep], [names[i] for i in keep]
    np.savez_compressed(path, X=X, columns=np.array(names), did=did, name=name,
                        licence=desc.get("licence", ""), target=desc.get("default_target_attribute") or "")
    return dict(did=did, name=name, status="ok", rows=X.shape[0], cols=X.shape[1], licence=desc.get("licence", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue")
    ap.add_argument("out")
    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--max_cols", type=int, default=100)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dids = pd.read_csv(a.catalogue)["did"].astype(int).tolist()
    results = []
    with cf.ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(process, d, a.out, a.max_rows, a.max_cols): d for d in dids}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(dict(did=futs[fut], status=f"failed: {type(e).__name__}: {str(e)[:120]}"))
            if i % 25 == 0:
                print(f"{i}/{len(dids)}", flush=True)
    json.dump(sorted(results, key=lambda r: r["did"]), open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    by = pd.Series([r["status"].split(":")[0] for r in results]).value_counts().to_dict()
    print("status:", by)
    for r in results:
        if r["status"] == "excluded":
            print(f"  excluded {r['did']}: {r['name']}")
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.endswith(".npz"))
    print(f"total {size / 1e6:.0f} MB")
    sys.exit(0)


if __name__ == "__main__":
    main()
