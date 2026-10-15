#!/bin/bash
# Training command for the CFM-scale head 'v2_cnp_rand_s0', generated from the arguments
# recorded in its checkpoint and verified against train_head.py's parser.
# See configs/cfm/README.md. Runs in the CausalPFN environment, from the repo root.
python experiments/cfm/train_head.py \
  --out_dir logs/cfm/v2_cnp_rand_s0 \
  --steps 250000 \
  --mask_p_random 1.0 \
  --eval_every 5000 \
  --workers 14 \
  --n_range 100 5000 \
  --openml_prob 0.5 \
  --openml_pack data/cfm/openml_pack \
  --outcome_transform_prob 0.3 \
  --prior_preset v2
