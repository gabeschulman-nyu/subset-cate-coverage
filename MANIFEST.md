# Manifest

The result files the figures and tables read, and the scripts that produced
them. `make verify` checks them against `artifacts.sha256`.

| files | script | contents |
|---|---|---|
| `logs/inspec/benchmark_ladder.json` | `experiments/benchmark_ladder.py` | CausalPFN on the six benchmarks and Hill-B heavy-tailed, 40 datasets each |
| `logs/inspec/ladder_*.json` (6) | `experiments/subset_selection_coverage.py` | CausalPFN, subset-selection rules, n=500 |
| `logs/inspec/res_*.json` (17) | `experiments/inspec_stage{1,2,3}.py` | CausalPFN on the synthetic family, by n and sharpness |
| `logs/inspec/twopath_*.json` (63) | `experiments/two_path_ablation.py` | CNP and latent NP on the synthetic family (tags below) |
| `logs/cfm/final/causalpfn_paperT.json` | `experiments/cfm/eval_final.py --method causalpfn` | CausalPFN with the temperatures of the benchmark run (`--paper_temperatures`) |
| `logs/cfm/final/v2_*.json` (6) | `experiments/cfm/eval_final.py --method <run>=checkpoints/cfm/<run>.pt` | heads on CausalPFN, six benchmarks, 40 datasets each |

## `twopath_` tags

A trailing `_s0`, `_s1` or `_s2` is the training seed, `soft` marks evaluation
at sharpness s=4, and `n2krepro` the n=2000 models of Table 5. Each file
records its settings, including the checkpoint it evaluated.

| tag | model | head | n | subset selection | pool fraction | seeds |
|---|---|---|---|---|---|---|
| `blad010` | CNP | Gaussian | 500 | neighbor_ball | 0.10 | 3 |
| `blad025` | CNP | Gaussian | 500 | neighbor_ball | 0.25 | 3 |
| `blad050` | CNP | Gaussian | 500 | neighbor_ball | 0.50 | 3 |
| `elad010` | CNP | Gaussian | 500 | effect_top | 0.10 | 3 |
| `elad025` | CNP | Gaussian | 500 | effect_top | 0.25 | 3 |
| `elad050` | CNP | Gaussian | 500 | effect_top | 0.50 | 3 |
| `g1` | CNP | Gaussian | 500 | random | 0.25 | 3 |
| `g1corr` | CNP | Gaussian | 500 | x1_quartile | 0.25 | 3 |
| `g1soft` | CNP | Gaussian | 500 | random | 0.25 | 3 |
| `lad010` | CNP | Gaussian | 500 | propensity_stratum | 0.10 | 3 |
| `lad025` | CNP | Gaussian | 500 | propensity_stratum | 0.25 | 3 |
| `lad050` | CNP | Gaussian | 500 | propensity_stratum | 0.50 | 3 |
| `lat` | latent NP | histogram | 500 | random | 0.25 | 3 |
| `latcorr` | latent NP | histogram | 500 | x1_quartile | 0.25 | 3 |
| `llad010` | latent NP | histogram | 500 | propensity_stratum | 0.10 | 3 |
| `llad025` | latent NP | histogram | 500 | propensity_stratum | 0.25 | 3 |
| `llad050` | latent NP | histogram | 500 | propensity_stratum | 0.50 | 3 |
| `n2krepro_a` | latent NP | histogram | 2000 | random | 0.25 | 1 |
| `n2krepro_b` | CNP | histogram | 2000 | random | 0.25 | 1 |
| `n2krepro_nv` | unit-level only | histogram | 2000 | random | 0.25 | 1 |
| `olad010` | CNP | Gaussian | 500 | effect_oracle | 0.10 | 3 |
| `olad025` | CNP | Gaussian | 500 | effect_oracle | 0.25 | 3 |
| `olad050` | CNP | Gaussian | 500 | effect_oracle | 0.50 | 3 |
