# Unit-level aggregation breaks coverage in causal foundation models

Code, trained models and derived results for the paper. Every figure and
table rebuilds from what is in this archive in about ten seconds on a CPU.
Rebuilding the results themselves needs the benchmark data (fetched by the
scripts in `benchmarks/`) and a GPU; the steps and their measured runtimes
are below.

## Layout

```
configs/                    training config and command for each of the six models
checkpoints/                the six trained models (CNP and latent NP, three seeds each)
src/subsetcate/            model and training code
experiments/                evaluation scripts and the three figure generators
benchmarks/                 benchmark generators and loaders, data download scripts
docs/latex/                 the table generator; writes docs/latex/tables/
docs/figures/               the three figures as they appear in the paper
logs/inspec/                derived results the figures and tables are built from
```

## Install

```
conda env create -f environment.yml && conda activate subsetcate
pip install -e .
```

`environment.yml` pins the versions that produced the results (Python
3.10.16, torch 2.5.1, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3,
matplotlib 3.10.9, pytorch-lightning 2.6.5, scikit-learn 1.6.1). The CausalPFN
steps run in a second environment, `environment-causalpfn.yml`, which pins
`causalpfn==0.1.4`.

## Rebuilding the figures and tables

```
make verify
```

checks `artifacts.sha256`, deletes the figures and tables, and regenerates
them from `logs/inspec/`. All generators are seeded and the output is
byte-identical to the files shipped here. `make all` skips the checksum step.

| paper | file | generator | reads (`logs/inspec/`) | time |
|---|---|---|---|---|
| Figure 1 | `docs/figures/benchmark_ladder.pdf` | `experiments/make_figure_benchmarks.py` | `benchmark_ladder.json` | 3 s |
| Figure 2 | `docs/figures/twopath.pdf` | `experiments/make_figure_twopath.py` | `twopath_{g1,lat}_s{0,1,2}.json` | 2 s |
| Figure 3 | `docs/figures/design_effect.pdf` | `experiments/make_figure_collapse.py` | the 60 `twopath_*` cells of our models, `res_*.json`, `ladder_*.json` | 3 s |
| Table 1 | written in the paper | -- | -- | -- |
| Table 2 | `docs/latex/tables/threearm.tex` | `docs/latex/build_tex_tables.py` | `res_n500.json`, `ladder_propensity_m_ovsoft500.json`, `benchmark_ladder.json`, `twopath_g1_s*.json` | <1 s for all five tables |
| Table 3 | `docs/latex/tables/subset_selection.tex` | same | `ladder_*_n500.json`, `ladder_*_ovsoft500.json`, `res_n500.json`, `res_ovsoft500.json` | |
| Table 4 | `docs/latex/tables/legacy_breaking.tex` | same | `benchmark_ladder.json` | |
| Table 5 | `docs/latex/tables/n2000.tex` | same | `twopath_n2krepro_*.json` | |
| Table 6 | `docs/latex/tables/nsweep.tex` | same | `res_n{125,250,500,1000,2000}.json` | |

Times are on one CPU core. `experiments/artifacts.py` decides which result
file supplies each setting, for every figure and table alike.
`MANIFEST.md` lists every result file with the script that produced it.

## Rebuilding the results

Times are measured on one NVIDIA L40S, or on CPUs where stated.

**Training (the six models).** `configs/<name>.sh` is the exact training
command for each model and `configs/<name>.json` its recorded configuration;
see `configs/README.md`. One seed takes about 1 GPU-hour for the CNP and 1.7
for the latent NP. The trained models are shipped in `checkpoints/`:

| model | checkpoint |
|---|---|
| CNP, seeds 0-2 | `checkpoints/sweep_gauss1_s{0,1,2}_E128_S30000/*/epoch=*.ckpt` |
| latent NP, seeds 0-2 | `checkpoints/sweep_latfix_s{0,1,2}_E128_S30000/*/epoch=*.ckpt` |

These are the checkpoints the result files name in their `checkpoint`
field. They hold the weights and the configuration of `configs/`; optimizer
state is not included.

**Fixed-weight evaluation (Figures 2 and 3, Table 2).**
`experiments/two_path_ablation.py --checkpoint <ckpt> --tag <tag> ...` scores
one cell, about 25 s each. The arguments behind each shipped cell are in
`MANIFEST.md` and in each file's own fields. Tasks are preprocessed the way the
checkpoint was trained: the CNP on the raw scale, the latent NP standardized
per task. The models behind Table 5 were trained separately and are not
shipped; its cells are.

**CausalPFN on the synthetic family (Tables 2, 3, 6, Figure 3).**
`experiments/inspec_stage1.py` generates tasks, `inspec_stage2.py` (CausalPFN
environment) draws per-unit posterior samples, `inspec_stage3.py` scores them,
and `subset_selection_coverage.py` scores the selection rules. About 1.5 min
per setting.

**CausalPFN on the benchmarks (Figure 1, Tables 2 and 4).**

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

## Notes on CausalPFN

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

## What is not here

Raw posterior draws, the benchmark datasets, the models behind Table 5, and
models or result files that did not reach the paper.
