#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: dqn.py
# @Description: This script implements the original DQN algorithm following the Nature released version of DQN paper 'Human-level control through deep reinforcement learning'.
# @Time: 2023/05/22
# @Author: Yueyuan Li


from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from rllib.interface import AgentBase
from rllib.interface import ConfigBase
from rllib.buffer import RandomReplayBuffer
from rllib.exploration import EpsilonGreedy


class QNetwork(nn.Module):
    """The Q-network used in the original DQN paper"""

    def __init__(self, num_channels: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)
        return x

    def action(self, state: torch.Tensor) -> int:
        x = self.forward(state)
        action = x.argmax()
        action = action.detach().cpu().numpy()
        return action


class DQNConfig(ConfigBase):
    def __init__(self, configs: dict):
        super().__init__()

        self.set_env(configs)

        # model
        ## hyper-parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.explore = True
        self.explore_func = EpsilonGreedy
        self.explore_kwargs = {
            "reduce_epsilon": True,
            "initial_epsilon": 1,
            "final_epsilon": 0.1,
            "step_decay": int(1e6),
        }
        self.n_initial_exploration_steps = 5e4
        self.buffer_size: int = int(1e6)

        ## networks
        self.lr = 2.5e-4
        self.eps = 0.01
        self.q_net = QNetwork
        self.q_net_kwargs = {"num_channels": 4, "action_dim": self.action_dim}
        self.target_update_freq = 1e4

        self.merge_configs(configs)


class DQN(AgentBase):
    name = "DQN"

    def __init__(self, configs: DQNConfig, device: torch.device = torch.device("cpu")):
        super().__init__(configs, device)

        # networks
        self.policy_net = self.configs.q_net(**self.configs.q_net_kwargs).to(self.device)
        self.target_net = deepcopy(self.policy_net)
        self.update_cnt = 0

        # optimizer
        self.optimizer = torch.optim.RMSprop(
            self.policy_net.parameters(), lr=self.configs.lr, eps=self.configs.eps
        )

        # the replay buffer
        self.buffer = RandomReplayBuffer(self.configs.buffer_size)

        # exploration method
        if self.configs.explore:
            self.explore_func = self.configs.explore_func(**self.configs.explore_kwargs)

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action = self.policy_net.action(state)
        if self.configs.explore:
            action = self.explore_func.explore(action, self.configs.action_space)

        return action

    def train(self):
        if len(self.buffer) < self.configs.n_initial_exploration_steps:
            return

        batches = self.buffer.sample(self.configs.batch_size)
        state = torch.FloatTensor(batches["state"]).to(self.device)
        action = torch.FloatTensor(batches["action"]).to(self.device)
        next_state = torch.FloatTensor(batches["next_state"]).to(self.device)
        reward = torch.FloatTensor(batches["reward"]).to(self.device)
        done = torch.FloatTensor(batches["done"]).to(self.device)

        # loss function
        q_value = self.policy_net(state)[range(self.configs.batch_size), action.long()]
        next_q_value = self.target_net(next_state).max(-1)[0]
        q_target = reward + self.configs.gamma * (1 - done) * next_q_value
        loss = F.smooth_l1_loss(q_value, q_target)

        # update policy net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target net
        self.update_cnt += 1
        if self.update_cnt % self.configs.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.configs.debug:
            return loss.item()

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
