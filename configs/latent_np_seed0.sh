#!/bin/bash
# Training command for the final checkpoint 'latent_np_seed0', generated from its
# recorded configuration and verified against run-experiment's parser.
# See configs/README.md.
run-experiment \
  --experiment_setup_path experiments/exact_polynomial.py \
  --disable_wandb \
  --experiment_name polynomial-confounder \
  --query_name ITE \
  --model_variant route_a \
  --lr_schedule cosine \
  --route_a_objective combined \
  --fml_lambda 2.0 \
  --estimator_type histogram \
  --embed_dim 128 \
  --standardize \
  --deterministic_datagen \
  --dataset_size 500 \
  --num_train_tasks 8192 \
  --num_val_tasks 256 \
  --num_test_tasks 128 \
  --online_data \
  --total_steps 30000 \
  --lr 0.0003 \
  --patience 100000 \
  --checkpoint_dir checkpoints/latent_np_seed0
