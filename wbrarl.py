import sys
import os
import time
import random
import argparse
import multiprocessing
import pickle
import copy
from multiprocessing import freeze_support
import numpy as np
import torch
import gym
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import warnings


LAST_LAYER_DIM = 256

HYPERS_SAC = {'Hopper-v3': {'learning_starts': 4000, 'learning_rate': 0.0002}}
HYPERS_PPO = {'HalfCheetah-v3': {'batch_size': 64,
                             'ent_coef': 0.0025,
                             'n_steps': 128,  # orig was 512, made smaller because n_envs is high
                             'gamma': 0.98,
                             'learning_rate': 2.0633e-05,
                             'gae_lambda': 0.92,
                             'n_epochs': 12,  # orig was 20
                             'max_grad_norm': 0.5,
                             'vf_coef': 0.58096,
                             'clip_range': 0.06,
                             'policy_kwargs': {'log_std_init': -2.0, 'ortho_init': False,
                                          'activation_fn': torch.nn.ReLU,
                                          'net_arch': [dict(pi=[256, 256], vf=[256, 256])]}}}
ADV_HYPERS_SAC = {'Hopper-v3': {'ent_coef': 0.15, 'learning_starts': 4000}}
ADV_HYPERS_PPO = {'HalfCheetah-v3': {'ent_coef': 0.0075}}

COEF_DICT = {'HalfCheetah-v3': {'mass': [0.2, 0.3, 0.4, 0.5, 1.5, 2.0, 2.5, 3.0],
                                'friction': [0.05, 0.1, 0.2, 0.3, 1.3, 1.5, 1.7, 1.9]},
             'Hopper-v3': {'mass': [0.2, 0.3, 0.4, 0.5, 1.05, 1.1, 1.15, 1.2],
                           'friction': [0.2, 0.3, 0.4, 0.5, 1.4, 1.6, 1.8, 2.0]},
             }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_type', type=str, default='')  # 'ctrl', 'rarl', or 'rarl' with 'act', 'val', and/or 'lat' as prefixes
    parser.add_argument('--agent_ckpt', type=str, default='')
    parser.add_argument('--env_ckpt', type=str, default='')
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='./models/')
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--n_test_episodes', type=int, default=10)
    parser.add_argument('--n_envs', type=int, default=16)  # epoch size is n_steps * n_envs
    parser.add_argument('--n_train', type=int, default=int(2e6))
    parser.add_argument('--n_train_per_iter', type=int, default=10000)  # how often to switch advs and report results
    parser.add_argument('--test_each', type=int, default=2)  # how often to test
    parser.add_argument('--start_adv_training', type=int, default=200000)  # when to start the adv
    parser.add_argument('--n_advs', type=int, default=1)  # how many adversaries to train in an ensemble
    parser.add_argument('--delta_action', type=float, default=0.075)  # how much to let the adv maximally perturb
    parser.add_argument('--lam', type=float, default=0.05)  # how much to penalize the adversary's action L1 norm
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--perturb_style', type=str, default='action')  # 'body' or 'action' depending on what the adversary perturbs
    parser.add_argument('--n_report', type=int, default=2)

    args = parser.parse_args()
    return args


def get_seed():
    # gets a random seed from the current time
    return int(str(time.time()).replace('.', '')[-5:])


class DummyAdvEnv(gym.Wrapper):
    # this is used for initializing adversarial policies

    def __init__(self, env, lat, act, val, act_space):
        self.env = env
        obs_dict = {'ob': self.env.observation_space}
        if lat:
            lat_size = LAST_LAYER_DIM
            obs_dict['lat'] = gym.spaces.Box(np.float32(-np.inf * np.ones(lat_size)),
                                             np.float32(np.inf * np.ones(lat_size)))
        if act:
            obs_dict['act'] = self.env.action_space
        if val:
            obs_dict['val'] = gym.spaces.Box(np.float32(np.array([-np.inf])), np.float32(np.array([np.inf])))
        self.observation_space = gym.spaces.Dict(obs_dict)
        self.action_space = act_space


class RARLEnv(gym.Wrapper):
    # this can be an env for either the protagonist or adversary depending on whether agent_mode or adv_mode is called

    def __init__(self, env, args, agent_ckpt, adv_ckpts, mode):

        super().__init__(env)
        self.env = env
        self.args = copy.deepcopy(args)
        self.sd = get_seed()
        self.lat = 'lat' in self.args.experiment_type
        self.act = 'act' in self.args.experiment_type
        self.val = 'val' in self.args.experiment_type
        self.observation = None
        self.agent_action = None
        self.agent_ckpt = agent_ckpt
        if isinstance(adv_ckpts, str):
            adv_ckpts = [adv_ckpts]
        self.adv_ckpts = adv_ckpts

        if mode == 'agent':
            self.agent_mode()
        elif mode == 'adv':
            self.adv_mode()

    def agent_mode(self):

        # get observation space, action space, agents, step, and reset
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        if self.adv_ckpts[0]:
            self.advs = [args.alg.load(self.args.model_dir + self.adv_ckpts[i], device='cpu')
                         for i in range(args.n_advs)]
        else:
            dummy_adv_env = DummyAdvEnv(copy.deepcopy(self.env), self.lat, self.act, self.val, self.get_adv_action_space())
            self.advs = [self.args.alg('MultiInputPolicy', dummy_adv_env, seed=self.sd, device='cpu', **self.args.adv_hypers[self.args.env])
                        for _ in range(args.n_advs)]
        if self.agent_ckpt:
            self.agent = self.args.alg.load(self.args.model_dir + self.agent_ckpt, device='cpu')
        else:
            self.agent = self.args.alg('MlpPolicy', self, device='cpu', seed=self.sd, **self.args.hypers[self.args.env])

        self.step = self.step_agent
        self.reset = self.reset_agent
        self.adv_i = 0

    def adv_mode(self):

        # get observation space, action space, agents, step, and reset
        obs_dict = {'ob': self.env.observation_space}
        if self.lat:
            lat_size = LAST_LAYER_DIM
            obs_dict['lat'] = gym.spaces.Box(np.float32(-np.inf * np.ones(lat_size)),
                                             np.float32(np.inf * np.ones(lat_size)))
        if self.act:
            obs_dict['act'] = self.env.action_space
        if self.val:
            obs_dict['val'] = gym.spaces.Box(np.float32(np.array([-np.inf])), np.float32(np.array([np.inf])))
        self.observation_space = gym.spaces.Dict(obs_dict)
        self.action_space = self.get_adv_action_space()
        if self.agent_ckpt:
            self.agent = self.args.alg.load(self.args.model_dir + self.agent_ckpt, device='cpu')
        else:
            self.agent = self.args.alg('MlpPolicy', self.env, device='cpu', seed=self.sd, **self.args.hypers[self.args.env])

        self.step = self.step_adv
        self.reset = self.reset_adv

    def reset_agent(self):
        self.observation = self.env.reset()
        self.adv_i = random.randint(0, len(self.advs)-1)
        return self.observation

    def reset_adv(self):
        self.observation = self.env.reset()
        self.agent_action = self.agent.predict(self.observation, deterministic=True)[0]
        return self.get_adv_obs(self.agent_action)

    def get_adv_obs(self, agent_action):
        obs = {'ob': self.observation}
        if self.lat:
            tens_ob = torch.unsqueeze(torch.from_numpy(self.observation), dim=0).float()
            if self.args.alg == SAC:
                latent_pi_val = self.agent.policy.actor.latent_pi(tens_ob)
            else:
                features = self.agent.policy.extract_features(tens_ob)
                latent_pi_val, _ = self.agent.policy.mlp_extractor(features)
            self.agent_latent = latent_pi_val.detach().numpy()
            obs['lat'] = np.squeeze(self.agent_latent)
        if self.act:
            obs['act'] = agent_action
        if self.val:
            raise NotImplementedError
        return obs

    def step_agent(self, agent_action):
        adv_obs = self.get_adv_obs(agent_action)
        adv_action = self.advs[self.adv_i].predict(adv_obs, deterministic=False)[0]
        if self.args.perturb_style == 'body':
            self.adv_to_xfrc(adv_action)
        if self.args.perturb_style == 'action':
            agent_action += adv_action
            agent_action = np.clip(agent_action, self.env.action_space.low, self.env.action_space.high)
        return self.env.step(agent_action)

    def step_adv(self, adv_action):
        if self.args.perturb_style == 'body':
            self.adv_to_xfrc(adv_action)
        self.observation, reward, done, infos = self.env.step(self.agent_action)
        norm_penalty = args.lam * np.mean(np.abs(adv_action))
        adv_reward = -1 * reward - norm_penalty
        self.agent_action = self.agent.predict(self.observation, deterministic=False)[0]
        if self.args.perturb_style == 'action':
            self.agent_action += adv_action
            self.agent_action = np.clip(self.agent_action, self.env.action_space.low, self.env.action_space.high)
        obs = self.get_adv_obs(self.agent_action)
        return obs, adv_reward, done, infos

    def get_adv_action_space(self):
        if self.args.perturb_style == 'body':
            high_adv = np.float32(np.ones(self.n_dim * len(self.body_idx)) * self.args.delta_body)
            return gym.spaces.Box(-high_adv, high_adv)
        elif self.args.perturb_style == 'action':
            high_adv = self.env.action_space.high * self.args.delta_action
            return gym.spaces.Box(-high_adv, high_adv)
        else:
            raise NotImplementedError



def make_rarl_env(wrapper, args, agent_ckpt, adv_ckpts, mode, rank):

    def _init():
        gym_env = gym.make(args.env)
        env = wrapper(gym_env, args, agent_ckpt, adv_ckpts, mode)
        env.seed(rank)
        return env

    set_random_seed(rank)
    return _init


def make_env(args, rank, mc=1.0, fc=1.0):

    def _init():
        env = gym.make(args.env)
        env.seed(rank)
        body_mass = env.model.body_mass * mc
        env.model.body_mass[:] = body_mass
        geom_friction = env.model.geom_friction * fc
        env.model.geom_friction[:] = geom_friction
        return env

    set_random_seed(rank)
    return _init


def get_save_suff(args, iter):

    savename = f'rarl_{args.env}_{iter * args.n_train_per_iter}_id={args.id}'
    if 'act' in args.experiment_type:
        savename = 'act_' + savename
    if 'val' in args.experiment_type:
        savename = 'val_' + savename
    if 'lat' in args.experiment_type:
        savename = 'lat_' + savename
    return savename


def simple_eval(policy, eval_env, n_episodes):
    all_rewards = []
    observation = eval_env.reset()
    for _ in range(n_episodes):
        done = False
        ep_reward = 0.0
        while not done:
            action = policy.predict(observation=observation, deterministic=False)[0]
            observation, reward, done, infos = eval_env.step(action)
            done = done[0]
            ep_reward += reward[0]
        all_rewards.append(ep_reward)
        observation = eval_env.reset()
    return sum(all_rewards) / n_episodes


def train_rarl(args):

    env_wrapper = RARLEnv
    n_iters = (args.n_train // args.n_train_per_iter)
    sd = get_seed()
    agent_rewards = []
    adv_improvements = []
    last_saved_agent = ''
    last_saved_adv = ''
    best_mean_reward = -np.inf

    adv_envs_raw = [SubprocVecEnv([make_rarl_env(env_wrapper, args, last_saved_agent, last_saved_adv, 'adv', sd + i)
                                   for i in range(args.n_envs)]) for _ in range(args.n_advs)]
    adv_envs = [VecNormalize(adv_envs_raw[j], norm_reward=False) for j in range(args.n_advs)]
    adv_policies = [args.alg('MultiInputPolicy', adv_envs[j], device=args.device, seed=sd, **args.adv_hypers[args.env]) for j in range(args.n_advs)]
    agent_env_raw = SubprocVecEnv([make_rarl_env(env_wrapper, args, last_saved_agent, last_saved_adv, 'agent', sd + i)
                                   for i in range(args.n_envs)])
    agent_env = VecNormalize(agent_env_raw, norm_reward=False)
    agent_policy = args.alg('MlpPolicy', agent_env, device=args.device, seed=sd, **args.hypers[args.env])
    last_saved_agent = 'agent_' + get_save_suff(args, 0)
    agent_policy.save(args.model_dir + last_saved_agent + '.zip')
    adv_eval_envs_raw = [SubprocVecEnv([make_rarl_env(env_wrapper, args, last_saved_agent, last_saved_adv, 'adv', 42)])
                         for _ in range(args.n_advs)]
    adv_eval_envs = [VecNormalize(adv_eval_envs_raw[j], norm_reward=False) for j in range(args.n_advs)]
    agent_eval_env_raw = SubprocVecEnv([make_env(args, 42)])
    agent_eval_env = VecNormalize(agent_eval_env_raw, norm_reward=False)
    last_saved_advs = []  # for deleting files no longer needed

    for i in range(1, n_iters + 1):

        save_suff = get_save_suff(args, i)
        n_train_this_iter = args.n_train_per_iter + args.hypers[args.env].get('learning_starts', 0)

        # train adv
        if ((args.perturb_style == 'body' and args.delta_body > 0.0) or
                (args.perturb_style == 'action' and args.delta_action > 0.0)) and \
                args.n_train_per_iter * i > args.start_adv_training:

            for adv_policy, adv_env, adv_eval_env in zip(adv_policies, adv_envs, adv_eval_envs):
                adv_env_raw = SubprocVecEnv([make_rarl_env(env_wrapper, args, last_saved_agent, last_saved_adv, 'adv', sd + i)
                                             for i in range(args.n_envs)])
                adv_env_state = adv_env.__getstate__()
                adv_env.__setstate__(adv_env_state)
                adv_env.set_venv(adv_env_raw)
                adv_policy.env = adv_env
                adv_eval_env_raw = SubprocVecEnv([make_rarl_env(env_wrapper, args, last_saved_agent, adv_policy, 'adv', 42)])
                adv_eval_env.__setstate__(adv_env_state)
                adv_eval_env.set_venv(adv_eval_env_raw)

            mean_rewards_pre = [simple_eval(adv_policy, adv_eval_envs[j], args.n_test_episodes) for j, adv_policy in enumerate(adv_policies)]

            for adv_policy in adv_policies:
                adv_policy.learn(n_train_this_iter)

            for adv_policy, adv_env, adv_eval_env in zip(adv_policies, adv_envs, adv_eval_envs):
                adv_env_state = adv_env.__getstate__()
                adv_eval_env_raw = SubprocVecEnv([make_rarl_env(env_wrapper, args, last_saved_agent, adv_policy, 'adv', 42)])
                adv_eval_env.__setstate__(adv_env_state)
                adv_eval_env.set_venv(adv_eval_env_raw)

            if (i - 1) % args.test_each == 0:
                mean_rewards_post = [simple_eval(adv_policy, adv_eval_envs[j], args.n_test_episodes) for j, adv_policy in enumerate(adv_policies)]
                adv_improvements.append(round((sum(mean_rewards_post) - sum(mean_rewards_pre)) / args.n_advs))
            if i % args.n_report == 0:
                print(f'{args.experiment_type} id={args.id} adv_improvements:', adv_improvements, sum(adv_improvements))
                sys.stdout.flush()

        for lsa in last_saved_advs:
            os.remove(args.model_dir + lsa + '.zip')
        last_saved_advs = [f'adv{j}_' + save_suff for j in range(args.n_advs)]
        for i_policy, adv_policy in enumerate(adv_policies):
            adv_policy.save(args.model_dir + last_saved_advs[i_policy] + '.zip')

        # train agent
        agent_env_raw = SubprocVecEnv([make_rarl_env(env_wrapper, args, last_saved_agent, last_saved_advs, 'agent', sd + j)
                                       for j in range(args.n_envs)])
        agent_env_state = agent_env.__getstate__()
        agent_env.__setstate__(agent_env_state)
        agent_env.set_venv(agent_env_raw)
        agent_policy.env = agent_env

        agent_policy.learn(n_train_this_iter)

        agent_env_state = agent_env.__getstate__()
        agent_eval_env_raw = SubprocVecEnv([make_env(args, 42)])
        agent_eval_env.__setstate__(agent_env_state)
        agent_eval_env.set_venv(agent_eval_env_raw)

        if (i - 1) % args.test_each == 0:
            mean_reward = simple_eval(agent_policy, agent_eval_env, args.n_test_episodes)
            if mean_reward >= best_mean_reward:
                best_mean_reward = mean_reward
                best_save_suff = get_save_suff(args, n_iters)
                agent_savename = 'best_agent_' + best_save_suff
                agent_policy.save(args.model_dir + agent_savename + '.zip')
            agent_rewards.append(round(mean_reward))

        if i % args.n_report == 0:
            print(f'{args.env} {args.experiment_type} id={args.id} timestep: {i * args.n_train_per_iter}, mean agent rewards: {agent_rewards}')
            sys.stdout.flush()

        os.remove(args.model_dir + last_saved_agent + '.zip')
        last_saved_agent = 'agent_' + save_suff
        agent_policy.save(args.model_dir + last_saved_agent + '.zip')

    savename = 'agent_' + get_save_suff(args, n_iters)
    agent_policy.save(args.model_dir + savename + '.zip')
    with open(args.results_dir + savename + '_rewards.pkl', 'wb') as f:
        pickle.dump(agent_rewards, f)
    agent_eval_env.save(args.model_dir + savename + '_eval_env')


def train_control(args):

    n_iters = (args.n_train // args.n_train_per_iter)
    sd = get_seed()
    env = VecNormalize(SubprocVecEnv([make_env(args, sd + i) for i in range(args.n_envs)]), norm_reward=False)
    eval_env = VecNormalize(SubprocVecEnv([make_env(args, 42)]), norm_reward=False)
    policy = args.alg('MlpPolicy', env, device=args.device, seed=sd, **args.hypers[args.env])

    best_mean_reward = -np.inf
    savename = f'best_agent_control_{args.env}_{args.n_train}_id={args.id}'

    rewards = []
    for i in range(1, n_iters + 1):
        n_train_this_iter = args.n_train_per_iter + args.hypers[args.env].get('learning_starts', 0)
        policy.learn(n_train_this_iter)

        # update the state of the eval env to be the same as the regular env
        env_state = env.__getstate__()
        eval_env_raw = SubprocVecEnv([make_env(args, 42)])
        eval_env.__setstate__(env_state)
        eval_env.set_venv(eval_env_raw)

        if i % args.n_report == 0:
            mean_reward = simple_eval(policy, eval_env, args.n_test_episodes)
            rewards.append(round(mean_reward))
            if mean_reward >= best_mean_reward:
                best_mean_reward = mean_reward
                policy.save(args.model_dir + savename + '.zip')
        if i % args.n_report == 0:
            print(f'{args.env} {args.experiment_type} id={args.id} timestep: {i * args.n_train_per_iter}, mean agent rewards: {rewards}')
            sys.stdout.flush()

    with open(args.results_dir + f'agent_control_{args.env}_{args.n_train}_id={args.id}' + '_rewards.pkl', 'wb') as f:
        pickle.dump(rewards, f)

    eval_env.save(args.model_dir + f'agent_control_{args.env}_{args.n_train}_id={args.id}_eval_env')

    env.close()
    eval_env.close()


def eval_agent_grid(args):

    mass_coeffs = COEF_DICT[args.env]['mass']
    friction_coeffs = COEF_DICT[args.env]['friction']
    assert args.agent_ckpt, 'Must give --agent_ckpt to test an agent'
    assert args.env_ckpt, 'Must give --env_ckpt to test an agent'
    all_mean_rewards = []

    for mc in mass_coeffs:
        all_mean_rewards.append([])
        for fc in friction_coeffs:
            eval_env = SubprocVecEnv([make_env(args, 42, mc, fc)])
            eval_env = VecNormalize.load(args.model_dir + args.env_ckpt, eval_env)
            agent_policy = args.alg.load(args.model_dir + args.agent_ckpt, device=args.device)
            mean_reward = simple_eval(agent_policy, eval_env, 16)
            print(f'{args.agent_ckpt} mass={mc} friction={fc} mean eval reward: {mean_reward}')
            all_mean_rewards[-1].append(mean_reward)

    with open(args.results_dir + args.agent_ckpt + f'_eval.pkl', 'wb') as f:
        pickle.dump(all_mean_rewards, f)


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    freeze_support()
    multiprocessing.set_start_method('spawn')
    args = parse_args()

    if 'HalfCheetah' in args.env:
        args.alg = PPO
        args.hypers = HYPERS_PPO
        args.adv_hypers = ADV_HYPERS_PPO
    else:
        args.alg = SAC
        args.hypers = HYPERS_SAC
        args.adv_hypers = ADV_HYPERS_SAC

    if args.mode == 'eval':
        eval_agent_grid(args)
    elif 'rarl' in args.experiment_type:
        train_rarl(args)
    elif args.experiment_type == 'ctrl':
        train_control(args)
    else:
        raise NotImplementedError()

    print('Done :)')

