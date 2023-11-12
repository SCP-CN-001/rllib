#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: rainbow.py
# @Description: This script implements the Rainbow algorithm following the paper 'Rainbow: Combining Improvements in Deep Reinforcement Learning'. The Rainbow algorithm is a combination of several improvements to the DQN algorithm, including Double Q-learning, Prioritized Experience Replay, Dueling Network, Multi-step Learning, Distributional RL, and Noisy Nets.
# @Time: 2023/05/25
# @Author: Yueyuan Li


from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rllib.interface import AgentBase
from rllib.interface import ConfigBase
from rllib.buffer import PrioritizedReplayBuffer


class NoisyLinear(nn.Module):
    """Extension: Noisy Nets.

    In rainbow all the linear layers are replaced by noisy linear layers. The noisy linear layer is from the paper 'Noisy Networks for Exploration'.
    """

    def __init__(self, in_features, out_features, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_(0, 1)
        self.bias_epsilon.normal_(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )


class RainbowQNetwork(nn.Module):
    def __init__(self, num_channels: int, action_dim: int, n_atoms: int, support: torch.Tensor):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.support = support

        # Extension: Dueling network architecture.
        self.feature = nn.Sequential(
            nn.Conv2d(num_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Extension: Noisy linear layer.
        self.value_hidden = NoisyLinear(7 * 7 * 64, 512)
        self.value = NoisyLinear(512, n_atoms)

        self.advantage_hidden = NoisyLinear(7 * 7 * 64, 512)
        self.advantage = NoisyLinear(512, action_dim * n_atoms)

    def dist(self, state: torch.Tensor):
        feature = self.feature(state)

        value_hidden = F.relu(self.value_hidden(feature))
        value = self.value(value_hidden).view(-1, 1, self.n_atoms)
        advantage_hidden = F.relu(self.advantage_hidden(feature))
        advantage = self.advantage(advantage_hidden).view(-1, self.action_dim, self.n_atoms)

        q_atoms = value + advantage - advantage.mean(-1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # avoid nans

        return dist

    def forward(self, state: torch.Tensor):
        dist = self.dist(state)
        x = torch.sum(dist * self.support, dim=-1)
        return x

    def action(self, state: torch.Tensor) -> int:
        x = self.forward(state)
        action = x.argmax()
        action = action.detach().cpu().numpy()
        return action

    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()


class RainbowConfig(ConfigBase):
    def __init__(self, configs: dict):
        super().__init__()

        self.set_env(configs)

        # model
        ## hyper-parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.n_initial_exploration_steps = int(8e4)
        self.buffer_size = int(1e6)
        self.buffer_kwargs = {"alpha": 0.5, "beta": 0.4, "beta_increment": 1e-6}
        self.multi_step = 3
        self.v_min = -10
        self.v_max = 10
        self.n_atoms = 51

        ## networks
        self.lr = 6.25e-5
        self.epsilon = 1.5e-4
        self.q_net = RainbowQNetwork
        self.q_net_kwargs = {
            "num_channels": 4,
            "action_dim": self.action_dim,
            "n_atoms": self.n_atoms,
        }
        self.target_update_freq = 3.2e4

        self.merge_configs(configs)


class Rainbow(AgentBase):
    name = "Rainbow"

    def __init__(self, configs: RainbowConfig, device: torch.device = torch.device("cpu")):
        super().__init__(configs, device)

        # Extension: Category Distributional RL
        self.support = torch.linspace(
            self.configs.v_min, self.configs.v_max, self.configs.n_atoms
        ).to(self.device)
        self.configs.q_net_kwargs["support"] = self.support

        # networks
        self.policy_net = self.configs.q_net(**self.configs.q_net_kwargs).to(self.device)
        self.target_net = deepcopy(self.policy_net)
        self.update_cnt = 0

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.configs.lr, eps=self.configs.epsilon
        )

        # buffer
        ## Extension: Prioritized Experience Replay
        self.buffer = PrioritizedReplayBuffer(
            self.configs.buffer_size, **self.configs.buffer_kwargs
        )
        ## Extension: Multi-step Learning
        self.n_step_buffer = deque(maxlen=self.configs.multi_step)

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action = self.policy_net.action(state)

        return action

    def push(self, transition: tuple):
        # Extension: Multi-step Learning
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.configs.multi_step:
            return

        state, action = self.n_step_buffer[0][:2]
        next_state, reward, done = self.n_step_buffer[-1][2:]

        for t in reversed(list(self.n_step_buffer)[:-1]):
            next_s, r, d = t[2:]
            reward = r + self.configs.gamma * reward * (1 - d)
            next_state = next_s if d == 1 else next_state
            done = d if d == 1 else done

        self.buffer.push((state, action, next_state, reward, done))

    def train(self):
        if len(self.buffer) < self.configs.n_initial_exploration_steps:
            return

        batch, leaf_idx = self.buffer.sample(self.configs.batch_size)
        state = torch.FloatTensor(batch["state"]).to(self.device)
        action = torch.FloatTensor(batch["action"]).to(self.device)
        next_state = torch.FloatTensor(batch["next_state"]).to(self.device)
        reward = torch.FloatTensor(batch["reward"]).to(self.device)
        done = torch.FloatTensor(batch["done"]).to(self.device)
        weight = torch.FloatTensor(batch["weight"]).to(self.device)

        # Extention: Double DQN's loss function
        next_action = self.policy_net(next_state).argmax(1)
        next_dist = self.target_net.dist(next_state)[range(self.configs.batch_size), next_action]

        # Extension: N-step Learning
        n_step_gamma = self.configs.gamma**self.configs.multi_step

        # Extension: Category Distributional RL
        delta_z = float(self.configs.v_max - self.configs.v_min) / (self.configs.n_atoms - 1)

        q_target = reward.unsqueeze(-1) + n_step_gamma * self.support * (1 - done)
        q_target = q_target.clamp(min=self.configs.v_min, max=self.configs.v_max)
        b = (q_target - self.configs.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = (
            torch.linspace(
                0, (self.configs.batch_size - 1) * self.configs.n_atoms, self.configs.batch_size
            )
            .long()
            .unsqueeze(1)
            .expand(self.configs.batch_size, self.configs.n_atoms)
            .to(self.device)
        )

        proj_dist = torch.zeros(next_dist.size(), device=self.device)
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

        dist = self.policy_net.dist(state)
        log_p = torch.log(dist[range(self.configs.batch_size), action.long()])
        elementwise_loss = -(proj_dist * log_p).sum(1).squeeze()

        ## add weight to the loss
        loss = torch.mean(elementwise_loss * weight)

        # update policy net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target net
        self.update_cnt += 1
        if self.update_cnt % self.configs.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Extension: Prioritized Experience Replay
        ## update priority
        loss_priority = elementwise_loss.detach().cpu().numpy()
        self.buffer.update_priority(leaf_idx, loss_priority)

        # Extension: Noisy Nets
        ## reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        if self.configs.debug:
            return loss.item()

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "support": self.support,
                "configs": self.configs,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        for key in ["support", "configs"]:
            if key in checkpoint:
                setattr(self, key, checkpoint[key])
