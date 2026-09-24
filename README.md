# Unit-level aggregation breaks coverage in causal foundation models

Code and derived artifacts for the paper. Everything the paper reports can be
rebuilt from what is in this repository; see *Reproducing the paper* below for
the two boundaries where raw inputs are too large or not ours to ship.

## Layout

```
configs/                training commands for the six final checkpoints
src/metadentify/        model and training code
experiments/            the eval scripts, and the three figure generators
benchmarks/             vendored loaders for the six benchmarks (no data)
docs/latex/             the table generator; writes docs/latex/tables/
docs/figures/           the three figures as they appear in the paper
logs/inspec/            derived measurement artifacts the figures read
```

## Install

```
conda env create -f environment.yml && conda activate metadentify
pip install -e .
```

## Reproducing the paper

Figures and tables rebuild from the shipped artifacts in `logs/inspec/`, with
no GPU and no model download:

```
make verify
```

This checks `artifacts.sha256`, then rebuilds
`docs/figures/{benchmark_ladder,twopath,design_effect}.pdf` and
`docs/latex/tables/{legacy_breaking,n2000,nsweep}.tex`. Every producer is
seeded and the figures are byte-reproducible, so a clean run reproduces the
paper's images and tables exactly. `make all` skips the checksum step.

`MANIFEST.md` lists every artifact with the script and arguments that
produced it. It also decodes the `twopath_` filenames, which are named
after the sweep a cell came from rather than after what it measures --
`g1_s0` and `lad025_s1` are not meant to be read cold. Each file also
carries its own `arm`, `head`, `n`, `selection` and `checkpoint` fields,
which are authoritative if the table and a filename ever disagree.

Versions used for the results in the paper: Python 3.10.16, torch 2.5.1+cu121,
numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, matplotlib 3.10.9, lightning 2.6.5,
scikit-learn 1.6.1.

### Rebuilding the artifacts themselves

The measurement artifacts are derived from two upstream sources, neither of
which is shipped.

**Our models.** `experiments/two_path_ablation.py` and
`experiments/subset_selection_coverage.py` read trained checkpoints. The
paper uses six: the Gaussian-head CNP and the latent NP, three seeds each.
`configs/` holds the exact training command and recorded configuration for
each; every command was verified against `run-experiment`'s parser. Each run
is 30,000 steps and takes one to two GPU-hours.

**CausalPFN rows.** `experiments/inspec_stage{1,2,3}.py` generate tasks, draw
per-unit posteriors from the released CausalPFN checkpoint, and score them.
Stage 2 needs the `causalpfn` package and a GPU.
`experiments/benchmark_ladder.py` reads per-unit draws for the six benchmarks
as `{DGP}_seed{NNNNN}.npz`; producing them is about 9 GB, so only the derived
summary (`logs/inspec/benchmark_ladder.json`) is shipped. Point `PIT_PARTS` at
the directory if you regenerate them.

### Benchmark data

The loaders in `benchmarks/` are included; the datasets are not, because their
licences are not ours to redistribute. `bench_lib.py` documents what each
loader expects on disk — IHDP as the `ihdp_npci_1-100` splits, ACIC 2016 as the
covariate table plus the `aciccomp2016` generator output. The four simulated
designs need no download.

## What is not here

Training checkpoints, raw posterior draws, the ablation grid that did not reach
the paper, models trained with other output heads, sample sizes or budgets,
and the transformer arm that was evaluated and cut. This repository
is the paper and nothing else.
