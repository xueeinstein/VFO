#!/bin/bash
# ========================================================
#              Exploration at 100 Episode
#
# Code # which fix maze layout but sample actions with
# different seeds
#
# Requirements:
# 1. run ./experiments/test.sh for a while,
#    manually stop it after get option training snapshot
# ========================================================
seed=0
local_steps=5
episodes=100
PY=$VIRTUAL_ENV"/bin/python"
echo "Using virtual env at "$VIRTUAL_ENV
echo "Running with seed: "$seed

iter_num=1000
while [ $iter_num -lt $episodes ]
do
  echo "Ep: "$iter_num
  env OPENAI_LOGDIR="logs/exploration_100/episode_"$iter_num $PY -m baselines.run \
    --alg vfo --env standard_maze --num_timesteps 0 --seed $seed \
    --load_path="logs/vfo_standard_maze_test_no_distil/snapshot" \
    --play

  env OPENAI_LOGDIR="logs/exploration_100/op_episode_"$iter_num $PY -m baselines.run \
    --alg vfo --env standard_maze --num_timesteps 0 --seed $seed \
    --load_path="logs/vfo_standard_maze_test_no_distil/snapshot" \
    --selective_option_play

  iter_num=$(( $iter_num + 1 ))
done

python experiments/plot_exploration_100.py \
  --fig logs/exploration_100/mf_policy_exploration.png \
  -p "logs/exploration_100/episode_*/log.txt" \
  --steps 5

python experiments/plot_exploration_100.py \
  --fig logs/exploration_100/selective_option_exploration.png \
  -p "logs/exploration_100/op_episode_*/log.txt" \
  --steps 5
