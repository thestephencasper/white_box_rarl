
for ENV_NAME in Simglucose # Cancer HalfCheetah-v3 Hopper-v3
do

  for id_num in 1 2 3 4 5 6 7 8 9 10
  do

    # runs the three types of agents simultaneously
    python -W ignore wbrarl.py --experiment_type=ctrl --id=${id_num} --env=${ENV_NAME} --device='cuda:0' &
    python -W ignore wbrarl.py --experiment_type=rarl --id=${id_num} --env=${ENV_NAME} --n_advs=3 --device='cuda:0' &
    python -W ignore wbrarl.py --experiment_type=act_lat_rarl --id=${id_num} --env=${ENV_NAME} --n_advs=3 --device='cuda:0'

    # tests the three types of agents simultaneously
    python -W ignore wbrarl.py --env=${ENV_NAME} --mode=eval --agent_ckpt=best_agent_control_${ENV_NAME}_1000000_id=${id_num} --env_ckpt=agent_control_${ENV_NAME}_1000000_id=${id_num}_eval_env --device='cuda:0' &
    python -W ignore wbrarl.py --env=${ENV_NAME} --mode=eval --agent_ckpt=best_agent_rarl_${ENV_NAME}_1000000_id=${id_num} --env_ckpt=agent_rarl_${ENV_NAME}_1000000_id=${id_num}_eval_env --device='cuda:0' &
    python -W ignore wbrarl.py --env=${ENV_NAME} --mode=eval --agent_ckpt=best_agent_lat_act_rarl_${ENV_NAME}_1000000_id=${id_num} --env_ckpt=agent_lat_act_rarl_${ENV_NAME}_1000000_id=${id_num}_eval_env --device='cuda:0'

  done
done

