#!/bin/bash

# NOTE: If you run into OOM issues, try reducing --num_envs
export LD_LIBRARY_PATH=""

#│ 'ant', 'ant_random_start', 'ant_ball', 'ant_push',         │
#│ 'humanoid', 'reacher', 'cheetah', 'pusher_easy', 'pusher_hard', 'pusher_reacher', 'pusher2', 'arm_reach', 'arm_grasp',   │
#│ 'arm_push_easy', 'arm_push_hard', 'arm_binpick_easy', 'arm_binpick_hard', 'ant_ball_maze', 'ant_u_maze', 'ant_big_maze', │
#│ 'ant_hardest_maze', 'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze', 'simple_u_maze', 'simple_big_maze', │
#│ 'simple_hardest_maze'

#ant ant_ball humanoid pusher_easy arm_reach arm_grasp arm_binpick_easy ant_u_maze simple_big_maze humanoid_u_maze
for env in ant_ball arm_reach ; do
  for seed in 1 2 3 4 5 ; do
    for fn in norm ; do
      JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl python ../run.py crl \
        --wandb_project_name dcrl-2 --wandb_group ${env} --exp_name ${env}-${fn}-${seed} --log_wandb --total_env_steps 10000000 \
        --seed ${seed} \
        --env ${env} \
        --energy_fn ${fn}
    done
  done
done

echo "All runs have finished."
