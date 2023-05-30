#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: rainbow.py
# @Description: This script implements the Rainbow algorithm following the paper 'Rainbow: Combining Improvements in Deep Reinforcement Learning'. The Rainbow algorithm is a combination of several improvements to the DQN algorithm, including Double Q-learning, Prioritized Experience Replay, Dueling Network, Multi-step Learning, Distributional RL, and Noisy Nets.
# @Time: 2023/05/25
# @Author: Yueyuan Li


from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from rllib.interface import AgentBase
from rllib.interface import ConfigBase
from rllib.buffer import PrioritizedReplayBuffer
from rllib.exploration import EpsilonGreedy


class NoisyLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class DuelingNetwork(nn.Module):
    """Extension: Rainbow uses a dueling network architecture.

    The Dueling DQN network is from the paper 'Dueling Network Architectures for Deep Reinforcement Learning'
    """

    def __init__(self, num_channels: int, action_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(num_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, 1))

        self.advantage = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, action_dim)
        )

    def forward(self, state: torch.Tensor):
        feature = self.feature(state)
        value = self.value(feature)
        advantage = self.advantage(feature)
        x = value + advantage - advantage.mean(-1, keepdim=True)
        return x

    def action(self, state: torch.Tensor) -> int:
        x = self.forward(state)
        action = x.argmax(1)[0]
        action = action.detach().cpu().numpy()
        return action


class RainbowConfig(ConfigBase):
    def __init__(self, configs: dict):
        super().__init__()

        self.set_env(configs)

        # model
        ## hyper-parameters
        self.explore_func = EpsilonGreedy
        self.explore_kwargs = {
            "reduce_epsilon": True,
            "initial_epsilon": 1,
            "final_epsilon": 0.1,
            "step_decay": int(1e6),
        }

        ## networks
        self.lr = 6.25e-5
        self.epsilon = 1.5e-4
        self.q_net = DuelingNetwork
        self.q_net_kwargs = {}
        self.buffer_kwargs = {}

        self.merge_configs(configs)


class Rainbow(AgentBase):
    name = "Rainbow"

    def __init__(self, configs: RainbowConfig, device: torch.device = torch.device("cpu")):
        super().__init__(configs, device)

        # networks
        self.policy_net = self.configs.q_net(**self.configs.q_net_kwargs).to(self.device)
        self.target_net = deepcopy(self.policy_net)

        # optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.configs.lr)

        # Extension: prioritized replay
        self.buffer = PrioritizedReplayBuffer(self.configs.buffer_size)

        # exploration
        self.explore = self.configs.explore_func(**self.configs.explore_kwargs)

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action = self.policy_net.action(state)

        return action

    def train(self):
        if len(self.buffer) < self.configs.n_initial_exploration_steps:
            return

        batches = self.buffer.sample(self.configs.batch_size)
        state = torch.FloatTensor(batches["state"]).to(self.device)
        action = torch.FloatTensor(batches["action"]).to(self.device)
        reward = torch.FloatTensor(batches["reward"]).to(self.device)
        next_state = torch.FloatTensor(batches["next_state"]).to(self.device)
        done = torch.FloatTensor(batches["done"]).to(self.device)

        # Extention: Double DQN's loss function
        q_value = self.policy_net(state).gather(1, action.long().unsqueeze(-1))
        next_q_value = self.target_net(next_state).gather(1, self.policy_net(next_state).argmax(1))
        q_target = reward + self.configs.gamma * next_q_value * (1 - done)
        loss = F.smooth_l1_loss(q_value, q_target.detach())

        # update policy net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target net
        self.update_cnt += 1
        if self.update_cnt % self.configs.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        return super().save(path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        self.target_net = deepcopy(self.policy_net)
