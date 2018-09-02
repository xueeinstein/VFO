#!/bin/bash
seeds=(0 100 200)
seeds_num=${#seeds[@]}
PY=$VIRTUAL_ENV"/bin/python"
echo "Using virtual env at "$VIRTUAL_ENV

iter_num=0
while [ $iter_num -lt $seeds_num ]
do
  seed=${seeds[$iter_num]}
  echo "Running with seed: "$seed
  env OPENAI_LOGDIR="logs/a2c_freeway_"$iter_num $PY -m baselines.run \
    --alg a2c --env FreewayNoFrameskip-v4 --num_timesteps 2e7 --seed $seed \
    --save_path "models/a2c_freeway_"$iter_num --network cnn \
    --num_env=16 \
    --lr=0.001 --vf_coef=10

  iter_num=$(( $iter_num + 1 ))
done
