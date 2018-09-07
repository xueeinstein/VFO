#!/bin/bash
# Usage: ./experiments/play.sh [env_name] [model_name]
# Then this script use xxx checkpoint under models for inference
seed=0
episodes=5
env_name="standard_maze"
model_name="vfo_standard_maze_0"
PY=$VIRTUAL_ENV"/bin/python"
echo "Using virtual env at "$VIRTUAL_ENV

if [ "$1" != "" ]; then
    env_name=$1
fi

if [ "$2" != "" ]; then
    model_name=$2
fi

iter_num=0
while [ $iter_num -lt $episodes ]
do
  echo "Ep: "$iter_num
  echo "Running with seed: "$seed
  env OPENAI_LOGDIR="plays/"$model_name"/episode_"$iter_num $PY -m baselines.run \
    --alg vfo --env $env_name --num_timesteps 0 --seed $seed \
    --load_path="models/"$model_name \
    --play

  env OPENAI_LOGDIR="plays/"$model_name"/ops_episode_"$iter_num $PY -m baselines.run \
    --alg vfo --env $env_name --num_timesteps 0 --seed $seed \
    --load_path="models/"$model_name \
    --options_play

  env OPENAI_LOGDIR="plays/"$model_name"/selective_op_episode_"$iter_num $PY \
      -m baselines.run --alg vfo --env $env_name --num_timesteps 0 --seed $seed \
    --load_path="models/"$model_name \
    --selective_option_play

  iter_num=$(( $iter_num + 1 ))
  seed=$(( $seed + 1 ))
done
