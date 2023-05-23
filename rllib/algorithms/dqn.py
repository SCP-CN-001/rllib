#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: dqn.py
# @Description: This file implements the original DQN algorithm following the Nature released version of DQN paper 'Human-level control through deep reinforcement learning'
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """The Q-network used in the original DQN paper"""
    def __init__(self, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)
        return x

    def action(self, state: torch.Tensor) -> int:
        x = self.forward(state)
        action = x.argmax(1)[0]
        action = action.detach().cpu().numpy()
        return action


class DQNConfig(ConfigBase):
    """Configuration of the DQN model"""
    def __init__(self, configs: dict):
        super().__init__()

        for key in ["state_space", "action_space"]:
            try:
                setattr(self, key, configs[key])
            except:
                raise AttributeError(f"{key} is not defined for DQN Config!")
        if "action_dim" not in configs.keys():
            self.num_actions = self.action_space.n
        else:
            self.num_actions = configs["action_dim"]

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
        self.replay_start_size = 5e4
        self.buffer_size: int = int(1e6)

        ## networks
        self.lr = 2.5e-4
        self.momentum = 0.95
        self.eps = 0.01
        self.q_net = QNetwork
        self.q_net_kwargs = {"num_actions": self.num_actions}
        self.target_update_freq = 1e4

        # tricks
        self.gradient_clip = False
        self.gradient_clip_range = 1

        self.merge_configs(configs)


class DQN(AgentBase):
    name = "DQN"

    def __init__(self, configs: DQNConfig):
        self.configs = configs

        # networks
        self.policy_net = self.configs.q_net(**self.configs.q_net_kwargs).to(device)
        self.target_net = deepcopy(self.policy_net)
        self.update_cnt = 0

        # optimizer
        self.optimizer = torch.optim.RMSprop(
            self.policy_net.parameters(),
            lr=self.configs.lr,
            momentum=self.configs.momentum,
            eps=self.configs.eps,
        )

        # the replay buffer
        self.buffer = RandomReplayBuffer(self.configs.buffer_size)

        # exploration method
        if self.configs.explore:
            self.explore_func = self.configs.explore_func(**self.configs.explore_kwargs)

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        action = self.policy_net.action(state)
        if self.configs.explore:
            action = self.explore_func.explore(action, self.configs.action_space)

        return action

    def train(self):
        if len(self.buffer) < self.configs.replay_start_size:
            return

        batches = self.buffer.sample(self.configs.batch_size)
        state = torch.FloatTensor(batches["state"]).to(device)
        action = torch.FloatTensor(batches["action"]).to(device)
        reward = torch.FloatTensor(batches["reward"]).to(device)
        next_state = torch.FloatTensor(batches["next_state"]).to(device)
        done = torch.FloatTensor(batches["done"]).to(device)

        # loss function
        q_value = self.policy_net(state)[range(self.configs.batch_size), action.long()]
        q_next = self.target_net(next_state).max(-1)[0]
        q_target = reward + self.configs.gamma * done * q_next
        loss = F.smooth_l1_loss(q_value, q_target)

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        if self.configs.gradient_clip:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update target net
        self.update_cnt += 1
        if self.update_cnt % self.configs.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.target_net = deepcopy(self.policy_net)
