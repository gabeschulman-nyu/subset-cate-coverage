# Training configurations of the CFM-scale heads

One pair of files per shipped head: the CNP head and the latent-NP head, three
seeds each, trained on the frozen, released CausalPFN.

| file | what it is |
|---|---|
| `<run>.json` | the configuration recorded in the head's checkpoint: its arguments, the prior they built (`prior_config`), and the step at which it was selected |
| `<run>.sh` | the `experiments/cfm/train_head.py` command that reproduces it |

The commands are generated from the recorded arguments by
`devtools/export_cfm_configs.py`, one flag per option that differs from the
parser's default, and each was checked by parsing it with `train_head.py`'s own
parser: every argument comes back as recorded.

Each run streams about 5.0M fresh tasks from the prior (250,000 steps of 8,192
context rows) and keeps the checkpoint with the lowest loss on 256 held-out
prior tasks, scored every 5,000 steps: the Gaussian NLL of the subset effect for
the CNP head, the training objective (64 draws) for the latent-NP head. One run
takes about 6.3 hours (CNP) or 9.2 hours (latent NP) on one NVIDIA L40S. Runs
need the OpenML base tables at `data/cfm/openml_pack` (`benchmarks/get_openml_tables.sh`)
and write to `logs/cfm/<run>/` (`best.pt`, `last.pt`, `log.jsonl`, `args.json`).

The task stream is seeded by `--seed` and by the data-loader worker index, so
reproducing a run's exact stream needs the same `--workers` (14).

| run | head | seed | selected step |
|---|---|---|---|
| `v2_cnp_rand_s0` | CNP | 0 | 245,000 |
| `v2_cnp_rand_s1` | CNP | 1 | 220,000 |
| `v2_cnp_rand_s2` | CNP | 2 | 215,000 |
| `v2_lathist_s0` | latent NP | 0 | 249,999 |
| `v2_lathist_s1` | latent NP | 1 | 230,000 |
| `v2_lathist_s2` | latent NP | 2 | 195,000 |
