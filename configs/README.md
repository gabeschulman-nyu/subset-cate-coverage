# Training configurations

One pair of files per final checkpoint: the Gaussian-head CNP and the latent
NP, three seeds each.

| file | what it is |
|---|---|
| `<name>.json` | the configuration recorded in the checkpoint at training time |
| `<name>.sh` | the `run-experiment` command that reproduces it |

The commands are generated from the recorded configurations, one flag per
option that differs from the parser's default, and each was verified by
splitting it as a shell would and parsing it with `run-experiment`'s own
parser: every option comes back as recorded. Two recorded keys,
`empirical_data_dir` and `empirical_max_len`, are not command-line options and
are unused by these models.

Checkpoint directories are set to `checkpoints/<name>`, and the three
experiment-tracking settings are omitted: tracking was off for every run and
is not part of the released code. Nothing else was changed from the recorded
values.

Selection follows Appendix B: the CNP keeps the checkpoint with the best
validation negative log-likelihood, the latent NP the one with the best
validation RMSE, each over the full 30,000-step run.
