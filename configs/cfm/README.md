# Training configurations of the heads on CausalPFN

The CNP and latent NP heads on the frozen CausalPFN, three seeds each.
`<run>.json` holds the arguments recorded in the head's checkpoint, the prior
configuration they produce (`prior_config`), and the selected step. `<run>.sh`
is the `experiments/cfm/train_head.py` command that reproduces the run,
generated from the recorded arguments and parsed back with `train_head.py`'s
parser to confirm every argument.

Each run trains for 250,000 steps on about 5.0M tasks drawn from the prior,
with 8,192 context rows per step. The selected checkpoint has the lowest loss
on 256 held-out prior tasks, evaluated every 5,000 steps (the Gaussian NLL of
the subset-CATE for the CNP head, the training objective with 64 draws for the
latent NP head). A run takes about 6.3 hours (CNP) or 9.2 hours (latent NP) on
one NVIDIA L40S, reads the OpenML tables from `data/cfm/openml_pack`
(`benchmarks/get_openml_tables.sh`), and writes to `logs/cfm/<run>/`. The task
stream depends on `--seed` and on the number of data-loader workers, so an
exact rerun needs `--workers 14`.

| run | head | seed | selected step |
|---|---|---|---|
| `v2_cnp_rand_s0` | CNP | 0 | 245,000 |
| `v2_cnp_rand_s1` | CNP | 1 | 220,000 |
| `v2_cnp_rand_s2` | CNP | 2 | 215,000 |
| `v2_lathist_s0` | latent NP | 0 | 249,999 |
| `v2_lathist_s1` | latent NP | 1 | 230,000 |
| `v2_lathist_s2` | latent NP | 2 | 195,000 |
