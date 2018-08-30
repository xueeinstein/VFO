#!/bin/bash
# Usage: ./test1.sh [unique_name]
seed=0
PY=$VIRTUAL_ENV"/bin/python"
echo "Using virtual env at "$VIRTUAL_ENV

echo "Running with seed: "$seed
env OPENAI_LOGDIR="logs/a2c_standard_maze_test"$1 $PY -m baselines.run \
  --alg a2c --env standard_maze --num_timesteps 1e7 --seed $seed \
  --save_path "models/a2c_standard_maze_test"$1 --network cnn \
  --num_env=16 \
  --lr=0.001 --vf_coef=0.01
