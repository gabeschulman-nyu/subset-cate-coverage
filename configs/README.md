# Training configurations

The CNP (Gaussian head) and latent NP on the synthetic family, three seeds
each. `<name>.json` is the configuration recorded in the checkpoint, and
`<name>.sh` is the `run-experiment` command that reproduces it. Each command
was generated from the recorded configuration and parsed back with
`run-experiment`'s parser to confirm every option.

The commands write to `checkpoints/<name>` and omit the experiment-tracking
settings, which were off for every run. The recorded keys `empirical_data_dir`
and `empirical_max_len` are not command-line options and are unused.

Model selection follows Appendix B. The CNP uses the checkpoint with the best
validation negative log-likelihood and the latent NP the one with the best
validation RMSE, each over the full 30,000-step run.
