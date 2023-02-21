import numpy as np
import torch

# import sys
# sys.path.append('..')

import gym


################ Cancer Simulator ####################
# This cancer simulator is built on the paper
#   * Ribba 2012 (https://pubmed.ncbi.nlm.nih.gov/22761472/)
#   * Yauney 2018 (http://proceedings.mlr.press/v85/yauney18a/yauney18a.pdf)

# The simulator approximate the dynamics of tumor growth, with actions consisting administration of chemotherapy TMZ at each timestep representing one month.

# * State space: 4-dimensional where the first dimension associates with drug concerntration and the last three with tumor diameter
# * Action space: TODO
# * Maximum Treatment cycle: 30


class Environment(object):
    """ generic base class """
    def __init__(self):
        pass

    def reset(self):
        pass

    def is_done(self):
        """ return True/False """
        pass

    def observe(self):
        """ return state """
        pass

    def transition(self, action):
        """ return state, reward """
        pass

    def get_reward(self):
        """ return reward """
        pass


# class EnvCancer(Environment):
class EnvCancer(gym.Env):

    ############# Initialization #############
    # dose_penalty -- 0 - no penalty; 1 - small; 10 - large; default no penalty;
    # max_steps -- max treatment cycle; default 30 days
    # transition_noise -- default 0 (i.e. deterministic)

    actions = np.array([
        [0],
        [1]
    ])
    map_actions_to_idx = {tuple(a): a_idx for a_idx, a in enumerate(actions)}
    action_idx_to_value = {v: k for k, v in map_actions_to_idx.items()}
    dim_S = 5
    n_actions = len(actions)

    def __init__(self, dose_penalty=10, initial_cycle=0, treatment_cycle=30, transition_noise=0.0, init_params=None,
                 w=0.5):
        # PDE parameter initialization (See Ribba 2012 Sec model & Table 1)
        if init_params is not None:
            self.lambda_p = init_params['lambda_p']
            self.k_qpp = init_params['k_qpp']
            self.k_pq = init_params['k_pq']
            self.gamma = init_params['gamma']
            self.delta_qp = init_params['delta_qp']
            self.P_init = 1.
            self.Q_init = 10.
        else:
            self.lambda_p = 0.121
            self.k_qpp = 0.0031
            self.k_pq = 0.0295
            self.gamma = 0.729
            self.delta_qp = 0.00867
            self.P_init = 7.13
            self.Q_init = 41.2

        self.kde = 0.24
        self.k = 100
        self.w = w  # w = 1 -> focus on tumor size; w = 0 -> focus on drug concentration
        self.alpha = 0.15
        self.last_action = None

        self.dose_penalty = dose_penalty
        self.initial_cycle = initial_cycle
        self.treatment_cycle = treatment_cycle
        # self.observation_cycle = observation_cycle
        self.transition_noise = transition_noise
        self.T = initial_cycle + treatment_cycle

        self.observation_space = gym.spaces.Box(-np.inf * np.ones(5), np.inf * np.ones(5))
        self.action_space = gym.spaces.Box(np.zeros(1), np.ones(1))

    def reset(self, random_init_state=False):
        Pm = self.P_init
        Qm = self.Q_init
        if random_init_state:
            assert False
            while True:
                P = np.random.lognormal(mean=-0.5 * np.log(0.94 ** 2 + 1) + np.log(Pm),
                                        sigma=np.sqrt(np.log(0.94 ** 2 + 1)))
                Q = np.random.lognormal(mean=-0.5 * np.log(0.54 ** 2 + 1) + np.log(Qm),
                                        sigma=np.sqrt(np.log(0.54 ** 2 + 1)))
                # maximum tumor size <= 100mm
                # if P+Q <= 100 and P >= self.P_init and Q >= self.Q_init: break
                if P + Q <= 100: break
        else:
            P = Pm
            Q = Qm

        C = 0
        Q_p = 0

        # current time index
        self.time_step = 0
        self.last_action = None
        self.state = np.array([C, P, Q, Q_p, self.time_step])
        self.state = np.array([float(ob) for ob in self.state])
        return self.state

    def is_done(self):
        return self.time_step >= self.T

    def observe(self):
        return self.state

    def get_reward(self, s, a, s_next):
        return self.w * self.R1(s, a, s_next) + (1 - self.w) * self.R2(s, a, s_next)

    def R1(self, s, a, s_next):
        # Focus on tumor size
        _, P, Q, Q_p, time_step = s
        _, P_next, Q_next, Q_p_next, time_step_next = s_next

        P_star = P + Q + Q_p
        P_star_next = P_next + Q_next + Q_p_next

        if time_step_next < self.T:
            reward = (P_star - P_star_next)
        else:
            P_init = self.P_init
            Q_init = self.Q_init
            Q_p_init = 0.
            P_star_init = P_init + Q_init + Q_p_init
            reward = 10 * (P_star_init - P_star_next)
        return reward

    def R2(self, s, a, s_next):
        # Focus on drug concentration
        C_next, _, _, _, time_step_next = s_next
        if time_step_next < self.T:
            reward = -self.dose_penalty * C_next
        else:
            reward = 0.

        return reward

    def transition(self, action):
        # current state
        C, P, Q, Q_p, time_step = state = self.state
        P_star = P + Q + Q_p

        if time_step < self.initial_cycle or time_step >= self.T:
            # assert action == 0
            action = 0.

        # perform_action
        time_step += 1
        if self.last_action is None:
            amt_added = action
        else:
            amt_added = self.alpha * action + (1 - self.alpha) * self.last_action
        self.last_action = amt_added
        C += amt_added
        C = C - self.kde * C
        P = (
                P + self.lambda_p * P * (1 - P_star / self.k)
                + self.k_qpp * Q_p
                - self.k_pq * P - self.gamma * C * self.kde * P
        )
        Q = (
                Q + self.k_pq * P
                - self.gamma * C * self.kde * Q
        )
        Q_p = (
                Q_p + self.gamma * C * self.kde * Q
                - self.k_qpp * Q_p
                - self.delta_qp * Q_p
        )

        # new state
        next_state = np.array([C, P, Q, Q_p, time_step])

        # add noise if any
        noise = 1 + self.transition_noise * np.random.randn(len(self.state))
        next_state *= noise
        next_state[-1] = time_step
        next_state = np.array([float(ob) for ob in next_state])
        self.state = next_state
        self.time_step = time_step

        # # define reward function
        # P_star_new = P + Q + Q_p
        # if self.is_done():
        #     P_init = 7.13
        #     Q_init = 41.2
        #     Q_p_init = 0.
        #     P_star_init = P_init + Q_init + Q_p_init
        #     reward = 10*(P_star_init - P_star_new)
        # else:
        #     reward = (P_star - P_star_new) - self.dose_penalty * C
        # return s_{t+1}, r
        reward = self.get_reward(state, action, next_state)
        return next_state, reward, self.is_done(), {}

    def step(self, action):
        return self.transition(action)
