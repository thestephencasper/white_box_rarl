# White-Box Adversarial Policies in Deep Reinforcement Learning

This repository contains code for White-Box RARL experiments from the paper, White-Box Adversarial Policies in Deep Reinforcement Learning (Casper et al., 2022)

Paper link and bibtex coming soon...

## Instructions

Set up [mujoco-py](https://github.com/openai/mujoco-py).

```
python -m pip install -r requirements.txt
bash wbrarl_run.sh
python wbrarl_plotting.py
```

This will train 2 control, rarl, and white-box rarl agents for the HalfCheetah-v3 and Hopper-v3 environments each. Then it will plot results like those in the paper. This may be slow, so making sure that gpus are in use and modifying the shell to run more commands in parallel may be useful. Please contact us with any questions. 
