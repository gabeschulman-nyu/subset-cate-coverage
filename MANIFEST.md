# Artifact manifest

Every measurement the figures and tables read, with the code that
produced it. Checksums are in `artifacts.sha256`; `make verify`
checks them and rebuilds every figure and table.

All producers are seeded, and the figures are byte-reproducible:
rerunning a generator on the same artifacts gives an identical image.

| artifact | producer | detail |
|---|---|---|
| `logs/inspec/benchmark_ladder.json` | `experiments/benchmark_ladder.py` | six benchmarks and Hill-B heavy-tailed, 40 reps each, scored against the conditional effect |
| `logs/inspec/ladder_effect_oracle_m_n500.json` | `experiments/subset_selection_coverage.py` | selection-rule ladder, n=500 |
| `logs/inspec/ladder_neighbor_ball_m_n500.json` | `experiments/subset_selection_coverage.py` | selection-rule ladder, n=500 |
| `logs/inspec/ladder_nested_n500.json` | `experiments/subset_selection_coverage.py` | selection-rule ladder, n=500 |
| `logs/inspec/ladder_nested_ovsoft500.json` | `experiments/subset_selection_coverage.py` | selection-rule ladder, n=500 |
| `logs/inspec/ladder_propensity_m_n500.json` | `experiments/subset_selection_coverage.py` | selection-rule ladder, n=500 |
| `logs/inspec/ladder_propensity_m_ovsoft500.json` | `experiments/subset_selection_coverage.py` | selection-rule ladder, n=500 |
| `logs/inspec/res_n1000.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=1000, 64 tasks |
| `logs/inspec/res_n125.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=125, 64 tasks |
| `logs/inspec/res_n2000.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=2000, 64 tasks |
| `logs/inspec/res_n250.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=250, 64 tasks |
| `logs/inspec/res_n500.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=500, 64 tasks |
| `logs/inspec/res_ov20_n1000.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=1000, 64 tasks |
| `logs/inspec/res_ov20_n2000.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=2000, 64 tasks |
| `logs/inspec/res_ov20_n250.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=250, 64 tasks |
| `logs/inspec/res_ov20_n500.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=500, 64 tasks |
| `logs/inspec/res_ov40_n1000.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=1000, 64 tasks |
| `logs/inspec/res_ov40_n2000.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=2000, 64 tasks |
| `logs/inspec/res_ov40_n250.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=250, 64 tasks |
| `logs/inspec/res_ov60_n1000.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=1000, 64 tasks |
| `logs/inspec/res_ov60_n2000.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=2000, 64 tasks |
| `logs/inspec/res_ov60_n250.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=250, 64 tasks |
| `logs/inspec/res_ov60_n500.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=500, 64 tasks |
| `logs/inspec/res_ovsoft500.json` | `experiments/inspec_stage{1,2,3}.py` | shipped CausalPFN, n=500, 64 tasks |
| `logs/inspec/twopath_blad010_s0.json` | `experiments/two_path_ablation.py` | --tag blad010_s0 --selection neighbor_ball, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_blad010_s1.json` | `experiments/two_path_ablation.py` | --tag blad010_s1 --selection neighbor_ball, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_blad010_s2.json` | `experiments/two_path_ablation.py` | --tag blad010_s2 --selection neighbor_ball, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_blad025_s0.json` | `experiments/two_path_ablation.py` | --tag blad025_s0 --selection neighbor_ball, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_blad025_s1.json` | `experiments/two_path_ablation.py` | --tag blad025_s1 --selection neighbor_ball, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_blad025_s2.json` | `experiments/two_path_ablation.py` | --tag blad025_s2 --selection neighbor_ball, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_blad050_s0.json` | `experiments/two_path_ablation.py` | --tag blad050_s0 --selection neighbor_ball, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_blad050_s1.json` | `experiments/two_path_ablation.py` | --tag blad050_s1 --selection neighbor_ball, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_blad050_s2.json` | `experiments/two_path_ablation.py` | --tag blad050_s2 --selection neighbor_ball, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_elad010_s0.json` | `experiments/two_path_ablation.py` | --tag elad010_s0 --selection effect_top, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_elad010_s1.json` | `experiments/two_path_ablation.py` | --tag elad010_s1 --selection effect_top, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_elad010_s2.json` | `experiments/two_path_ablation.py` | --tag elad010_s2 --selection effect_top, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_elad025_s0.json` | `experiments/two_path_ablation.py` | --tag elad025_s0 --selection effect_top, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_elad025_s1.json` | `experiments/two_path_ablation.py` | --tag elad025_s1 --selection effect_top, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_elad025_s2.json` | `experiments/two_path_ablation.py` | --tag elad025_s2 --selection effect_top, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_elad050_s0.json` | `experiments/two_path_ablation.py` | --tag elad050_s0 --selection effect_top, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_elad050_s1.json` | `experiments/two_path_ablation.py` | --tag elad050_s1 --selection effect_top, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_elad050_s2.json` | `experiments/two_path_ablation.py` | --tag elad050_s2 --selection effect_top, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_g1_s0.json` | `experiments/two_path_ablation.py` | --tag g1_s0 --selection random, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_g1_s1.json` | `experiments/two_path_ablation.py` | --tag g1_s1 --selection random, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_g1_s2.json` | `experiments/two_path_ablation.py` | --tag g1_s2 --selection random, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_g1corr.json` | `experiments/two_path_ablation.py` | --tag g1corr --selection x1_quartile, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_g1corr_s1.json` | `experiments/two_path_ablation.py` | --tag g1corr_s1 --selection x1_quartile, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_g1corr_s2.json` | `experiments/two_path_ablation.py` | --tag g1corr_s2 --selection x1_quartile, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_g1soft_s0.json` | `experiments/two_path_ablation.py` | --tag g1soft_s0 --selection random, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_g1soft_s1.json` | `experiments/two_path_ablation.py` | --tag g1soft_s1 --selection random, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_g1soft_s2.json` | `experiments/two_path_ablation.py` | --tag g1soft_s2 --selection random, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_lad010_s0.json` | `experiments/two_path_ablation.py` | --tag lad010_s0 --selection propensity_stratum, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_lad010_s1.json` | `experiments/two_path_ablation.py` | --tag lad010_s1 --selection propensity_stratum, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_lad010_s2.json` | `experiments/two_path_ablation.py` | --tag lad010_s2 --selection propensity_stratum, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_lad025_s0.json` | `experiments/two_path_ablation.py` | --tag lad025_s0 --selection propensity_stratum, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_lad025_s1.json` | `experiments/two_path_ablation.py` | --tag lad025_s1 --selection propensity_stratum, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_lad025_s2.json` | `experiments/two_path_ablation.py` | --tag lad025_s2 --selection propensity_stratum, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_lad050_s0.json` | `experiments/two_path_ablation.py` | --tag lad050_s0 --selection propensity_stratum, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_lad050_s1.json` | `experiments/two_path_ablation.py` | --tag lad050_s1 --selection propensity_stratum, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_lad050_s2.json` | `experiments/two_path_ablation.py` | --tag lad050_s2 --selection propensity_stratum, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_lat_s0.json` | `experiments/two_path_ablation.py` | --tag lat_s0 --selection random, ckpt sweep_latfix_s0_E128_S30000 |
| `logs/inspec/twopath_lat_s1.json` | `experiments/two_path_ablation.py` | --tag lat_s1 --selection random, ckpt sweep_latfix_s1_E128_S30000 |
| `logs/inspec/twopath_lat_s2.json` | `experiments/two_path_ablation.py` | --tag lat_s2 --selection random, ckpt sweep_latfix_s2_E128_S30000 |
| `logs/inspec/twopath_latcorr.json` | `experiments/two_path_ablation.py` | --tag latcorr --selection x1_quartile, ckpt sweep_latfix_s0_E128_S30000 |
| `logs/inspec/twopath_latcorr_s1.json` | `experiments/two_path_ablation.py` | --tag latcorr_s1 --selection x1_quartile, ckpt sweep_latfix_s1_E128_S30000 |
| `logs/inspec/twopath_latcorr_s2.json` | `experiments/two_path_ablation.py` | --tag latcorr_s2 --selection x1_quartile, ckpt sweep_latfix_s2_E128_S30000 |
| `logs/inspec/twopath_llad010_s0.json` | `experiments/two_path_ablation.py` | --tag llad010_s0 --selection propensity_stratum, ckpt sweep_latfix_s0_E128_S30000 |
| `logs/inspec/twopath_llad010_s1.json` | `experiments/two_path_ablation.py` | --tag llad010_s1 --selection propensity_stratum, ckpt sweep_latfix_s1_E128_S30000 |
| `logs/inspec/twopath_llad010_s2.json` | `experiments/two_path_ablation.py` | --tag llad010_s2 --selection propensity_stratum, ckpt sweep_latfix_s2_E128_S30000 |
| `logs/inspec/twopath_llad025_s0.json` | `experiments/two_path_ablation.py` | --tag llad025_s0 --selection propensity_stratum, ckpt sweep_latfix_s0_E128_S30000 |
| `logs/inspec/twopath_llad025_s1.json` | `experiments/two_path_ablation.py` | --tag llad025_s1 --selection propensity_stratum, ckpt sweep_latfix_s1_E128_S30000 |
| `logs/inspec/twopath_llad025_s2.json` | `experiments/two_path_ablation.py` | --tag llad025_s2 --selection propensity_stratum, ckpt sweep_latfix_s2_E128_S30000 |
| `logs/inspec/twopath_llad050_s0.json` | `experiments/two_path_ablation.py` | --tag llad050_s0 --selection propensity_stratum, ckpt sweep_latfix_s0_E128_S30000 |
| `logs/inspec/twopath_llad050_s1.json` | `experiments/two_path_ablation.py` | --tag llad050_s1 --selection propensity_stratum, ckpt sweep_latfix_s1_E128_S30000 |
| `logs/inspec/twopath_llad050_s2.json` | `experiments/two_path_ablation.py` | --tag llad050_s2 --selection propensity_stratum, ckpt sweep_latfix_s2_E128_S30000 |
| `logs/inspec/twopath_n2krepro_a.json` | `experiments/two_path_ablation.py` | --tag n2krepro_a --selection random, ckpt sweep_n2ka2_E128_S30000 |
| `logs/inspec/twopath_n2krepro_b.json` | `experiments/two_path_ablation.py` | --tag n2krepro_b --selection random, ckpt sweep_n2kb_E128_S30000 |
| `logs/inspec/twopath_n2krepro_nv.json` | `experiments/two_path_ablation.py` | --tag n2krepro_nv --selection random, ckpt sweep_n2knv_E128_S30000 |
| `logs/inspec/twopath_olad010_s0.json` | `experiments/two_path_ablation.py` | --tag olad010_s0 --selection effect_oracle, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_olad010_s1.json` | `experiments/two_path_ablation.py` | --tag olad010_s1 --selection effect_oracle, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_olad010_s2.json` | `experiments/two_path_ablation.py` | --tag olad010_s2 --selection effect_oracle, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_olad025_s0.json` | `experiments/two_path_ablation.py` | --tag olad025_s0 --selection effect_oracle, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_olad025_s1.json` | `experiments/two_path_ablation.py` | --tag olad025_s1 --selection effect_oracle, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_olad025_s2.json` | `experiments/two_path_ablation.py` | --tag olad025_s2 --selection effect_oracle, ckpt sweep_gauss1_s2_E128_S30000 |
| `logs/inspec/twopath_olad050_s0.json` | `experiments/two_path_ablation.py` | --tag olad050_s0 --selection effect_oracle, ckpt sweep_gauss1_s0_E128_S30000 |
| `logs/inspec/twopath_olad050_s1.json` | `experiments/two_path_ablation.py` | --tag olad050_s1 --selection effect_oracle, ckpt sweep_gauss1_s1_E128_S30000 |
| `logs/inspec/twopath_olad050_s2.json` | `experiments/two_path_ablation.py` | --tag olad050_s2 --selection effect_oracle, ckpt sweep_gauss1_s2_E128_S30000 |

## Reading a `twopath_` tag

These tags name the sweep a cell came from rather than what it
measures, and they are not renamed: several encode which checkpoint
was scored, which a name built from the measurement alone cannot
distinguish. The table below decodes every family that ships. A
trailing `_s0`/`_s1`/`_s2` is the training seed; each file also
records its own `arm`, `head`, `n`, `selection` and `checkpoint`
fields, which are authoritative.

Recurring fragments: `g1` is the Gaussian-head CNP and `lat` the
latent NP, both at n=500; `lad` is a pool-fraction ladder and the
digits are that fraction (`lad025` = 0.25), with a leading `b`, `e`,
`o` or `l` naming the rule it walks (neighbour ball, effect-top,
effect-oracle, or the latent NP's propensity ladder); `corr` marks
the covariate-quartile rule and `soft` evaluation at overlap s=4;
`n2krepro` cells are the n=2000 table's.

| tag family | arm | head | n | subset selection | pool fraction | seeds |
|---|---|---|---|---|---|---|
| `blad010` | CNP (direct decode) | mixture | 500 | neighbor_ball | 0.10 | 3 |
| `blad025` | CNP (direct decode) | mixture | 500 | neighbor_ball | 0.25 | 3 |
| `blad050` | CNP (direct decode) | mixture | 500 | neighbor_ball | 0.50 | 3 |
| `elad010` | CNP (direct decode) | mixture | 500 | effect_top | 0.10 | 3 |
| `elad025` | CNP (direct decode) | mixture | 500 | effect_top | 0.25 | 3 |
| `elad050` | CNP (direct decode) | mixture | 500 | effect_top | 0.50 | 3 |
| `g1` | CNP (direct decode) | mixture | 500 | random | 0.25 | 3 |
| `g1corr` | CNP (direct decode) | mixture | 500 | x1_quartile | 0.25 | 3 |
| `g1soft` | CNP (direct decode) | mixture | 500 | random | 0.25 | 3 |
| `lad010` | CNP (direct decode) | mixture | 500 | propensity_stratum | 0.10 | 3 |
| `lad025` | CNP (direct decode) | mixture | 500 | propensity_stratum | 0.25 | 3 |
| `lad050` | CNP (direct decode) | mixture | 500 | propensity_stratum | 0.50 | 3 |
| `lat` | latent NP | histogram | 500 | random | 0.25 | 3 |
| `latcorr` | latent NP | histogram | 500 | x1_quartile | 0.25 | 3 |
| `llad010` | latent NP | histogram | 500 | propensity_stratum | 0.10 | 3 |
| `llad025` | latent NP | histogram | 500 | propensity_stratum | 0.25 | 3 |
| `llad050` | latent NP | histogram | 500 | propensity_stratum | 0.50 | 3 |
| `n2krepro_a` | latent NP | histogram | 2000 | random | 0.25 | 1 |
| `n2krepro_b` | CNP (direct decode) | histogram | 2000 | random | 0.25 | 1 |
| `n2krepro_nv` | unit-level only | histogram | 2000 | random | 0.25 | 1 |
| `olad010` | CNP (direct decode) | mixture | 500 | effect_oracle | 0.10 | 3 |
| `olad025` | CNP (direct decode) | mixture | 500 | effect_oracle | 0.25 | 3 |
| `olad050` | CNP (direct decode) | mixture | 500 | effect_oracle | 0.50 | 3 |

Every shipped artifact has a surviving producer.

