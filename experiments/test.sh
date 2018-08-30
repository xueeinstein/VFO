#!/bin/bash
# Usage: ./test.sh [unique_name]
seed=0
PY=$VIRTUAL_ENV"/bin/python"
echo "Using virtual env at "$VIRTUAL_ENV

echo "Running with seed: "$seed
env OPENAI_LOGDIR="logs/vfo_standard_maze_test"$1 $PY -m baselines.run \
  --alg vfo --env standard_maze --num_timesteps 1e7 --seed $seed \
  --save_path "models/vfo_standard_maze_test"$1 --network cnn \
  --num_env=16 \
  --lr=0.001 --vf_coef=0.01 \
  --start_op_at=0.01
