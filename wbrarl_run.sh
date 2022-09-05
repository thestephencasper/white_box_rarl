
for ENV_NAME in HalfCheetah-v3 Hopper-v3
do

  for id_num in 1 2
  do

    python wbrarl.py --experiment_type=ctrl --id=${id_num} --env=${ENV_NAME} &
    python wbrarl.py --experiment_type=rarl --id=${id_num} --env=${ENV_NAME} --n_advs=3 &
    python wbrarl.py --experiment_type=act_lat_rarl --id=${id_num} --env=${ENV_NAME} --n_advs=3

    python wbrarl.py --env=${ENV_NAME} --mode=eval --agent_ckpt=best_agent_control_${ENV_NAME}_2000000_id=${id_num} --env_ckpt=agent_control_${ENV_NAME}_2000000_id=${id_num}_eval_env &
    python wbrarl.py --env=${ENV_NAME} --mode=eval --agent_ckpt=best_agent_rarl_${ENV_NAME}_2000000_id=${id_num} --env_ckpt=agent_rarl_${ENV_NAME}_2000000_id=${id_num}_eval_env &
    python wbrarl.py --env=${ENV_NAME} --mode=eval --agent_ckpt=best_agent_lat_act_rarl_${ENV_NAME}_2000000_id=${id_num} --env_ckpt=agent_lat_act_rarl_${ENV_NAME}_2000000_id=${id_num}_eval_env

  done
done
