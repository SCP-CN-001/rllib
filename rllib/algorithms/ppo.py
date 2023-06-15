#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: ppo.py
# @Description: This script implement the PPO algorithm following the paper 'Proximal Policy Optimization Algorithms'. Because the original paper is not very clear, the implementation also refers to the paper 'Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO'.
# @Time: 2023/05/26
# @Author: Yueyuan Li

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from rllib.interface import AgentBase
from rllib.interface import ConfigBase

from rllib.buffer import RandomReplayBuffer


def orthogonal_init(layer, gain: float = np.sqrt(2), constant: float = 0.0):
    nn.init.orthogonal_(layer.weight.data, gain)
    nn.init.constant_(layer.bias.data, constant)
    return layer


class PPOActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, continuous: bool):
        super().__init__()

        self.continuous = continuous

        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_size, action_dim), 0.01),
            nn.Tanh() if continuous else nn.Softmax(dim=-1),
        )

        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor):
        x = self.net(state)
        return x

    def get_dist(self, state: torch.Tensor):
        if self.continuous:
            mean = self.forward(state)
            std = self.log_std.expand_as(mean).exp()
            dist = Normal(mean, std)
        else:
            probs = self.forward(state)
            dist = Categorical(probs)

        return dist

    def action(self, state: torch.Tensor):
        dist = self.get_dist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action = action.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        dist = self.get_dist(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy


class PPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_size, 1)),
        )

    def forward(self, state: torch.Tensor):
        x = self.net(state)
        return x


class PPOConfig(ConfigBase):
    def __init__(self, configs: dict):
        super().__init__()

        self.set_env(configs)

        # model
        ## hyper-parameters
        ## Here use the PPO hyper-parameters from Mujoco as default values
        self.continuous = True
        self.num_env = 1
        self.num_epochs = 10
        self.horizon = 2048
        self.batch_size = 64
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.vf_coef = 0.5
        self.entropy_coef = 0.0

        ## actor net
        self.lr_actor = 3e-4
        self.actor_net = PPOActor
        self.actor_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 64,
            "continuous": self.continuous if "continuous" not in configs else configs["continuous"],
        }

        ## critic net
        self.lr_critic = 3e-4
        self.critic_net = PPOCritic
        self.critic_kwargs = {"state_dim": self.state_dim, "hidden_size": 64}

        self.merge_configs(configs)

        # implementation details
        self.norm_advantage = True
        self.clip_grad_norm = True


class PPO(AgentBase):
    name = "PPO"

    def __init__(self, configs: PPOConfig, device: torch.device = torch.device("cpu")):
        super().__init__(configs, device)

        # networks
        ## actor net
        self.actor_net = self.configs.actor_net(**self.configs.actor_kwargs).to(self.device)

        ## critic net
        self.critic_net = self.configs.critic_net(**self.configs.critic_kwargs).to(self.device)

        ## optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor_net.parameters(), "lr": self.configs.lr_actor, "eps": 1e-5},
                {"params": self.critic_net.parameters(), "lr": self.configs.lr_critic, "eps": 1e-5},
            ]
        )

        ## buffer
        self.buffer_cnt = 0
        self.buffer = [
            RandomReplayBuffer(self.configs.horizon, extra_items=["log_prob", "value"])
            for _ in range(self.configs.num_env)
        ]

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        action, log_prob = self.actor_net.action(state)
        value = self.critic_net(state)

        return action, log_prob, value

    def push(self, transitions: list):
        for i in range(self.configs.num_env):
            self.buffer[i].push(transitions[i])

        self.buffer_cnt += 1

    def train(self):
        if self.buffer_cnt < self.configs.horizon:
            return

        state = None
        action = None
        log_prob = None
        advantages = None
        returns = None

        for buffer in self.buffer:
            batches = buffer.all()
            _state = torch.FloatTensor(batches["state"]).to(self.device)
            _action = torch.FloatTensor(batches["action"]).to(self.device)
            _next_state = torch.FloatTensor(batches["next_state"]).to(self.device)
            _reward = torch.FloatTensor(batches["reward"]).to(self.device)
            _done = torch.FloatTensor(batches["done"]).to(self.device)
            _log_prob = torch.FloatTensor(batches["log_prob"]).to(self.device)
            _value = torch.FloatTensor(batches["value"]).to(self.device)

            # generalized advantage estimation
            with torch.no_grad():
                _advantages = torch.zeros_like(_reward).to(self.device)
                _lastgaelam = 0

                for t in reversed(range(self.configs.horizon)):
                    if t == self.configs.horizon - 1:
                        _next_done = 0
                        _next_value = self.critic_net(_next_state[-1])
                    else:
                        _next_done = _done[t + 1]
                        _next_value = _value[t + 1]

                    _delta = (
                        _reward[t] + self.configs.gamma * _next_value * (1 - _next_done) - _value[t]
                    )
                    _advantages[t] = _lastgaelam = (
                        _delta
                        + self.configs.gamma
                        * self.configs.gae_lambda
                        # * (1 - _next_done)
                        * _lastgaelam
                    )

                _advantages = _advantages.unsqueeze(-1)
                _value = _value.unsqueeze(-1)
                _returns = _advantages + _value

                state = _state if state is None else torch.cat([state, _state])
                action = _action if action is None else torch.cat([action, _action])
                log_prob = _log_prob if log_prob is None else torch.cat([log_prob, _log_prob])
                advantages = (
                    _advantages if advantages is None else torch.cat([advantages, _advantages])
                )
                returns = _returns if returns is None else torch.cat([returns, _returns])

        # update networks with mini-batch
        idxs = np.arange(self.configs.horizon)
        for _ in range(self.configs.num_epochs):
            np.random.shuffle(idxs)

            for i in range(0, self.configs.horizon, self.configs.batch_size):
                minibatch_idx = idxs[i : i + self.configs.batch_size]
                new_log_prob, entropy = self.actor_net.evaluate(
                    state[minibatch_idx], action[minibatch_idx]
                )
                new_value = self.critic_net(state[minibatch_idx])

                ratios = torch.exp(new_log_prob - log_prob[minibatch_idx])

                advantage_batch = advantages[minibatch_idx]
                if self.configs.norm_advantage:
                    advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                        advantage_batch.std() + 1e-8
                    )

                loss_clip = torch.min(
                    ratios * advantage_batch,
                    torch.clamp(
                        ratios, 1 - self.configs.clip_epsilon, 1 + self.configs.clip_epsilon
                    )
                    * advantage_batch,
                ).mean()

                loss_vf = F.mse_loss(new_value, returns[minibatch_idx]).mean()

                loss_entropy = entropy.mean()

                loss = -(
                    loss_clip
                    - self.configs.vf_coef * loss_vf
                    + self.configs.entropy_coef * loss_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.configs.clip_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                self.optimizer.step()

        for buffer in self.buffer:
            buffer.clear()
        self.buffer_cnt = 0

        return loss

    def save(self, path: str):
        torch.save(
            {
                "actor_net": self.actor_net.state_dict(),
                "critic_net": self.critic_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
