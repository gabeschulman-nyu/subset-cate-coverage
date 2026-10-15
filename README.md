# Unit-level aggregation breaks coverage in causal foundation models

Gabe Schulman, Lucius E.J. Bynum, Yoav Wald

Code, trained models and derived results for the paper. Every figure and
table rebuilds from what is in this repository in about ten seconds on a CPU.
Rebuilding the results themselves needs the benchmark data (fetched by the
scripts in `benchmarks/`) and a GPU; the steps and their measured runtimes
are below.

The model and training code in `src/subsetcate/` builds on
[metadentify](https://github.com/lbynum/metadentify), the code of
*Computational Identifiability* ([arXiv:2606.19361](https://arxiv.org/abs/2606.19361)),
released under the MIT licence; the package is renamed `subsetcate` here.
`LICENSE` carries its notice. CausalPFN, which the paper characterizes and on
which the CFM-scale heads are trained, is installed from PyPI and used under
its own licence; nothing of it is redistributed here.

## Layout

```
configs/              training config and command for each of the six synthetic-family models
configs/cfm/          the same for the six CFM-scale heads
checkpoints/          the six synthetic-family models (CNP and latent NP, three seeds each)
checkpoints/cfm/      the six CFM-scale heads on frozen CausalPFN (CNP and latent NP, three seeds each)
src/subsetcate/       model and training code
experiments/          evaluation scripts and figure generators
experiments/cfm/      CFM-scale heads: prior, training, evaluation, figures and tables
benchmarks/           benchmark generators and loaders, data download scripts
docs/latex/           the table generator; writes docs/latex/tables/
docs/figures/         the figures as they appear in the paper
logs/inspec/          derived results of CausalPFN and the synthetic-family models
logs/cfm/             derived results of the CFM-scale heads and of CausalPFN in the same pass
```

## Install

```
conda env create -f environment.yml && conda activate subsetcate
pip install -e .
```

`environment.yml` pins the versions that produced the results (Python
3.10.16, torch 2.5.1, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3,
matplotlib 3.10.9, pytorch-lightning 2.6.5, scikit-learn 1.6.1). It is all
that rebuilding the figures and tables needs. Two more environments pin the
other steps exactly:

| environment | for |
|---|---|
| `environment-causalpfn.yml` | everything that runs CausalPFN: its per-unit draws, and training and evaluating the CFM-scale heads. Pins `causalpfn==0.1.4` (see the notes on CausalPFN below). |
| `environment-cfmdata.yml` | fetching the OpenML base tables of the CFM-scale prior |

## Rebuilding the figures and tables

```
make verify
```

checks `artifacts.sha256`, deletes the figures and tables, and regenerates
them from `logs/`. All generators are seeded and the output is byte-identical
to the files shipped here. `make all` skips the checksum step.

| paper | file | generator | reads | time |
|---|---|---|---|---|
| Figure 1 | `docs/figures/benchmark_ladder.pdf` | `experiments/make_figure_benchmarks.py` | `logs/inspec/benchmark_ladder.json` | 3 s |
| Figure 2 | `docs/figures/twopath.pdf` | `experiments/make_figure_twopath.py` | `logs/inspec/twopath_{g1,lat}_s{0,1,2}.json` | 2 s |
| Figure 3 | `docs/figures/cfm_heads_vs_causalpfn.pdf` | `experiments/cfm/make_figure_cfm_heads.py` | `logs/inspec/benchmark_ladder.json`, `logs/cfm/final/v2_*.json` | 3 s |
| Figure 4 | `docs/figures/design_effect.pdf` | `experiments/make_figure_collapse.py` | the 60 `twopath_*` cells of our models, `res_*.json`, `ladder_*.json` | 3 s |
| Figure 5 | `docs/figures/cfm_fixed_weight_cnp.pdf` | `experiments/cfm/make_figure_fixed_weight.py --head cnp` | `logs/cfm/final/v2_cnp_rand_s{0,1,2}.json` | 3 s |
| Figure 6 | `docs/figures/cfm_fixed_weight_latent.pdf` | `experiments/cfm/make_figure_fixed_weight.py --head latent` | `logs/cfm/final/v2_lathist_s{0,1,2}.json` | 3 s |
| Table 1 | written in the paper | -- | -- | -- |
| Table 2 | `docs/latex/tables/threearm.tex` | `docs/latex/build_tex_tables.py` | `res_n500.json`, `ladder_propensity_m_ovsoft500.json`, `benchmark_ladder.json`, `twopath_g1_s*.json` | <1 s for all five tables |
| Table 3 | `docs/latex/tables/subset_selection.tex` | same | `ladder_*_n500.json`, `ladder_*_ovsoft500.json`, `res_n500.json`, `res_ovsoft500.json` | |
| Table 4 | `docs/latex/tables/legacy_breaking.tex` | same | `benchmark_ladder.json` | |
| Table 5 | `docs/latex/tables/n2000.tex` | same | `twopath_n2krepro_*.json` | |
| Table 6 | `docs/latex/tables/nsweep.tex` | same | `res_n{125,250,500,1000,2000}.json` | |
| Table 7 | `docs/latex/tables/heads_coverage.tex` | `experiments/cfm/final_tables.py` | `logs/cfm/final/*.json`, `logs/inspec/benchmark_ladder.json` | 1 s for both, and the summary below |
| Table 8 | `docs/latex/tables/heads_moments.tex` | same | same | |
| -- | `docs/cfm_arxiv_numbers.md` | same | same | |

`docs/cfm_arxiv_numbers.md` collects the CFM-scale numbers the text quotes,
with standard errors clustered by dataset, m_S and rho_S under two further
estimators, and how closely the predicted curves of Figures 5 and 6 follow
the observed ones. Times are on one CPU core. `experiments/artifacts.py` decides which result
file supplies each setting of the synthetic-family figures and tables.
`MANIFEST.md` lists every result file with the script that produced it.

## Rebuilding the results

Times are measured on one NVIDIA L40S, or on CPUs where stated.

**Training (the six synthetic-family models).** `configs/<name>.sh` is the
exact training command for each model and `configs/<name>.json` its recorded
configuration; see `configs/README.md`. One seed takes about 1 GPU-hour for
the CNP and 1.7 for the latent NP. The trained models are shipped in
`checkpoints/`:

| model | checkpoint |
|---|---|
| CNP, seeds 0-2 | `checkpoints/sweep_gauss1_s{0,1,2}_E128_S30000/*/epoch=*.ckpt` |
| latent NP, seeds 0-2 | `checkpoints/sweep_latfix_s{0,1,2}_E128_S30000/*/epoch=*.ckpt` |

These are the checkpoints the result files name in their `checkpoint`
field. They hold the weights and the configuration of `configs/`; optimizer
state is not included.

**Fixed-weight evaluation (Figures 2 and 4, Table 2).**
`experiments/two_path_ablation.py --checkpoint <ckpt> --tag <tag> ...` scores
one cell, about 25 s each. The arguments behind each shipped cell are in
`MANIFEST.md` and in each file's own fields. Tasks are preprocessed the way the
checkpoint was trained: the CNP on the raw scale, the latent NP standardized
per task. The models behind Table 5 were trained separately and are not
shipped; its cells are.

**CausalPFN on the synthetic family (Tables 2, 3, 6, Figure 4).**
`experiments/inspec_stage1.py` generates tasks, `inspec_stage2.py` (CausalPFN
environment) draws per-unit posterior samples, `inspec_stage3.py` scores them,
and `subset_selection_coverage.py` scores the selection rules. About 1.5 min
per setting.

**CausalPFN on the benchmarks (Figures 1 and 3, Tables 2 and 4).**

1. Fetch the data: `benchmarks/get_ihdp.sh` downloads the IHDP splits and
   checks them against the files we used; `benchmarks/get_acic2016.sh`
   installs the official `aciccomp2016` generator (needs R and git) and
   writes the 77 settings. Its output matches the files we used to within
   1e-9. The four simulated designs need no download. The datasets are not
   included because their licences are not ours to redistribute.
2. Draw per-unit samples (CausalPFN environment, from `benchmarks/`):
   `python run_causalpfn_pit.py 40 pit_calib HillA,HillB,HillBHeavy,FriedmanConf,KangSchafer,IHDP,ACIC2016 true 2000 true`.
   About 6 s per dataset (17 s for ACIC 2016), 40 min in all. The output is
   about 9 GB and is not shipped.
3. Score: `PIT_PARTS=benchmarks/results_pfn_pit/pit_calib/parts python experiments/benchmark_ladder.py`,
   about 16 min on 4 CPUs. Every unit is scored against its conditional
   effect: IHDP and ACIC 2016 store it, and for the simulated designs
   `experiments/benchmark_targets.py` recomputes it with
   `benchmarks/sim_lib.py` and refuses any target equal to y1 - y0.

**CFM-scale heads on frozen CausalPFN (Figures 3, 5 and 6, Tables 7 and 8).** Everything below
runs from the repository root in the CausalPFN environment unless stated.

1. Fetch the prior's real base tables (cfmdata environment):
   `benchmarks/get_openml_tables.sh`. It downloads CausalPFN's catalogue of
   OpenML datasets at commit `7da4afa`, fetches the 330 tables with a Parquet
   file, writes `data/cfm/openml_pack.{f32,json}` and checks the pack against
   the one the shipped heads were trained on. About 70 MB; the tables are not
   shipped, each keeping its own OpenML licence. Only training needs them.
2. Train: `configs/cfm/<run>.sh` is the exact command for each head (see
   `configs/cfm/README.md`): about 5.0M fresh prior tasks per run, 6.3 h for
   the CNP head and 9.2 h for the latent-NP head. The six heads are shipped
   in `checkpoints/cfm/<run>.pt` (weights, recorded arguments, prior
   configuration, selected step).
3. Evaluate, with the benchmark data (step 1 of the previous section):
   `python experiments/cfm/eval_final.py --method v2_cnp_rand_s0=checkpoints/cfm/v2_cnp_rand_s0.pt --out logs/cfm/final/v2_cnp_rand_s0.json`,
   and likewise for the other five heads: about 5 min per CNP head and 25 min
   per latent-NP head, for six benchmarks of 40 datasets. Each output keeps
   coverage and interval width per dataset and subset size for both
   aggregations, and the per-unit quantities m_S, rho_S and nRMSE. Random
   draws are seeded per dataset, so a rerun in the pinned environment on an
   L40S reproduces every coverage, width, m_S, rho_S and nRMSE in the shipped
   files exactly (a few per-size moments differ in the last bit). CausalPFN's
   own row, in the same pass and at the temperatures its calibration fitted
   in step 2 of the previous section:
   `python experiments/cfm/eval_final.py --method causalpfn --paper_temperatures $PIT_PARTS --out logs/cfm/final/causalpfn_paperT.json`
   (about 10 min).
4. `python experiments/cfm/verify_embeddings.py` checks that the per-unit
   states the heads read are CausalPFN's full state (its own output layer
   reproduces its predictions from them) and that the single full-context
   pass agrees with the public API wherever the API's neighbour contexts
   cover a whole treatment arm.

## Notes on our models

- The package contains only the model variants the paper uses: the CNP
  (`route_b`, one-component mixture head), the latent NP (`route_a`), and the
  two models behind Table 5 (`route_b` with a histogram head, and the
  unit-level-only `vanilla` model). The command-line parser still lists the
  options of variants that were removed, so that the recorded configurations
  parse; selecting a removed variant raises an error.
- The latent NP is trained with `route_a_objective: combined` (Appendix B).
  Its configurations also record `kl_free_bits` and `kl_warmup_steps`; the
  combined objective ignores both, because it has no KL term.
- The latent NP's checkpoints contain a posterior network over the latent,
  which is used neither in training nor at inference. It is kept so that the
  checkpoints load.
- The CFM-scale heads read, per unit, CausalPFN's last hidden state under
  both treatments (the input to its own output layer) and its effect mean and
  standard deviation, from one forward pass with every unit in context.
  CausalPFN stays frozen and in evaluation mode. The CNP head decodes a
  Gaussian for the subset effect; the latent-NP head decodes per-unit
  histograms given a shared latent and is trained with the same combined
  objective as the synthetic-family latent NP. Their training differs from
  the synthetic-family models' in the ways the paper lists: the prior and
  backbone, about 20 tasks per step for 250,000 steps with 32 subsets per
  task, gradient clipping at norm 1.0, size inputs (log k, k/n) to the CNP
  decoder, effects standardized by sd(y), and selection of the latent NP on
  its validation loss.

## Notes on CausalPFN

- Every result uses `causalpfn==0.1.4` from PyPI, whose source is identical to
  commit `3a40991` of [github.com/vdblm/CausalPFN](https://github.com/vdblm/CausalPFN).
  The appendix cites line numbers at commit `7da4afa`, a later commit still
  versioned 0.1.4; the cited lines are the same in both. The two commits
  differ in two places, neither of which changes any computation here: a
  floor of 2 on `min_samples_leaf` in the booster that picks the API's
  neighbour contexts (the same value for every dataset here, all of which
  have at least 747 units), and an optional `temperature` argument (always
  passed here).
- The weights are downloaded by the package from Hugging Face
  (`vdblm/causalpfn`, file `causalpfn_v0.pt`, sha256
  `4f0f53717f1c62426fe98faa0459d33bc8f9862a87e8484ca3869ca899e7841d`, the
  same at revisions `ccfc508` and `83aad07`).
- On the benchmarks CausalPFN runs with its built-in temperature calibration
  (`calibrate=True`, the fourth argument above); on the synthetic family it
  runs at its default temperature. It is not retrained.
- CausalPFN's aggregate interval draws one sample per unit from its sampler,
  averages the samples over the subset, and takes percentiles. The scripts
  above apply that construction to every subset from stored per-unit samples.
- `experiments/causalpfn_api_ate.py` checks the ATE interval on ACIC 2016
  against CausalPFN's own code: it contains the true ATE in 8 of the 40
  datasets with calibration, as in Table 4, and in 18 of 40 at the default
  temperature (10 min). In `causalpfn==0.1.4`, `CATEEstimator.estimate_ate_CI`
  raises `KeyError: 'ate'`: it looks up a key that the method it wraps,
  `_estimate_ate_cate_CI`, does not return. The script calls
  `_estimate_ate_cate_CI` directly, which computes the interval.
- Figure 1 uses CausalPFN's public API, which on ACIC 2016 (treatment arms
  over 1,024 units) conditions each window of units on its nearest neighbours;
  the CFM-scale heads use one full-context pass. `docs/cfm_arxiv_numbers.md`
  compares the two on the same datasets.

## What is not here

Raw posterior draws, the benchmark datasets, the OpenML tables, the models
behind Table 5, and models or result files that did not reach the paper.
