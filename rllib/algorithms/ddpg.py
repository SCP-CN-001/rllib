#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: ddpg.py
# @Description: This script implements the original DDPG algorithm following the paper 'Continuous control with deep reinforcement learning'.
# @Time: 2023/05/24
# @Author: Yueyuan Li

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rllib.interface import AgentBase
from rllib.interface import ConfigBase
from rllib.buffer import RandomReplayBuffer
from rllib.exploration.ornstein_uhlenbeck_noise import OrnsteinUhlenbeckNoise


def uniform_init(layer, init_weight: float):
    nn.init.uniform_(layer.weight.data, -init_weight, init_weight)
    nn.init.uniform_(layer.bias.data, -init_weight, init_weight)
    return layer


class DDPGActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size1: int,
        hidden_size2: int,
        init_weight: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            uniform_init(nn.Linear(state_dim, hidden_size1), 1 / np.sqrt(state_dim)),
            nn.ReLU(),
            uniform_init(nn.Linear(hidden_size1, hidden_size2), 1 / np.sqrt(hidden_size1)),
            nn.ReLU(),
            uniform_init(nn.Linear(hidden_size2, action_dim), init_weight),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)
        return x

    def action(self, state: torch.Tensor) -> np.ndarray:
        x = self.forward(state)
        action = x.detach().cpu().numpy()
        return action


class DDPGCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size1: int,
        hidden_size2: int,
        init_weight: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            uniform_init(
                nn.Linear(state_dim + action_dim, hidden_size1), 1 / np.sqrt(state_dim + action_dim)
            ),
            nn.ReLU(),
            uniform_init(nn.Linear(hidden_size1, hidden_size2), 1 / np.sqrt(hidden_size1)),
            nn.ReLU(),
            uniform_init(nn.Linear(hidden_size2, 1), init_weight),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = self.net(x)

        return x


class DDPGConfig(ConfigBase):
    def __init__(self, configs: dict):
        super().__init__()

        self.set_env(configs)

        # The default parameters are referred to the original paper's setting in the low dimension scenarios.

        # model
        ## hyper-parameters
        self.gamma: float = 0.99
        self.batch_size = 64
        self.tau: float = 1e-3  # soft-update factors
        self.buffer_size: int = int(1e6)
        self.ou_theta = 0.15  # for exploration based on Ornstein–Uhlenbeck process
        self.ou_sigma = 0.2
        self.ou_step = 0.002

        ## actor net
        self.lr_actor = 1e-4
        self.actor_net = DDPGActor
        self.actor_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size1": 400,
            "hidden_size2": 300,
            "init_weight": 3e-3,
        }

        ## critic net
        self.lr_critic = 1e-3
        self.critic_net = DDPGCritic
        self.critic_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size1": 400,
            "hidden_size2": 300,
            "init_weight": 3e-4,
        }

        self.merge_configs(configs)


class DDPG(AgentBase):
    name = "DDPG"

    def __init__(self, configs: DDPGConfig, device: torch.device = torch.device("cpu")):
        super().__init__(configs, device)

        # networks
        ## actor net
        self.actor_net = self.configs.actor_net(**self.configs.actor_kwargs).to(self.device)
        self.actor_target_net = deepcopy(self.actor_net)

        ## critic net
        self.critic_net = self.configs.critic_net(**self.configs.critic_kwargs).to(self.device)
        self.critic_target_net = deepcopy(self.critic_net)

        ## optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), self.configs.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), self.configs.lr_critic
        )

        # the replay buffer
        self.buffer = RandomReplayBuffer(self.configs.buffer_size)

        # exploration
        self.noise_generator = OrnsteinUhlenbeckNoise(
            self.configs.action_dim,
            0,
            self.configs.ou_theta,
            self.configs.ou_sigma,
            self.configs.ou_step,
        )

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        action = self.actor_net.action(state)
        action += self.noise_generator()

        return action

    def soft_update(self, target_net, current_net):
        for target, current in zip(target_net.parameters(), current_net.parameters()):
            target.data.copy_(
                current.data * self.configs.tau + target.data * (1.0 - self.configs.tau)
            )

    def train(self):
        if len(self.buffer) < self.configs.batch_size:
            return

        batches = self.buffer.sample(self.configs.batch_size)
        state = torch.FloatTensor(batches["state"]).to(self.device)
        action = torch.FloatTensor(batches["action"]).to(self.device)
        next_state = torch.FloatTensor(batches["next_state"]).to(self.device)
        reward = torch.FloatTensor(batches["reward"]).to(self.device)
        done = torch.FloatTensor(batches["done"]).to(self.device)

        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)

        # update the critic network
        next_action = self.actor_target_net(state)
        q_next = self.critic_target_net(next_state, next_action)
        q_target = reward + (1 - done) * self.configs.gamma * q_next
        q_value = self.critic_net(state, action)
        critic_loss = F.mse_loss(q_value, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update the actor network
        actor_loss = -self.critic_net(state, self.actor_net(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update target networks
        self.soft_update(self.actor_target_net, self.actor_net)
        self.soft_update(self.critic_target_net, self.critic_net)

        if self.configs.debug:
            return actor_loss.item(), critic_loss.item()

    def save(self, path: str):
        torch.save(
            {
                "actor_net": self.actor_net.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_net": self.critic_net.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        self.actor_target_net = deepcopy(self.actor_net)
        self.critic_target_net = deepcopy(self.critic_net)
