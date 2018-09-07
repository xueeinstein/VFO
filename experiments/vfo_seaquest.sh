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
  env OPENAI_LOGDIR="logs/vfo_seaquest_"$iter_num $PY -m baselines.run \
    --alg vfo --env SeaquestNoFrameskip-v4 --num_timesteps 3e7 --seed $seed \
    --save_path "models/vfo_seaquest_"$iter_num --network cnn \
    --num_env=16 \
    --start_op_at=0.5 --options_update_iter=1

  iter_num=$(( $iter_num + 1 ))
done
