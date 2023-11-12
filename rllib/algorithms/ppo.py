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
            policy = self.forward(state)
            dist = Categorical(policy)

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
        self.num_envs = 1
        self.num_epochs = 10
        self.horizon = 2048
        self.batch_size = 64
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.adam_epsilon = 1e-5
        self.clip_epsilon = 0.2
        self.vf_coef = 0.5
        self.entropy_coef = 0.0

        ## actor net
        self.actor_net = PPOActor
        self.actor_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 64,
            "continuous": self.continuous if "continuous" not in configs else configs["continuous"],
        }

        ## critic net
        self.critic_net = PPOCritic
        self.critic_kwargs = {"state_dim": self.state_dim, "hidden_size": 64}

        # implementation details
        self.adv_norm = True
        self.gradient_clip = True
        self.max_step = None

        self.merge_configs(configs)


class PPO(AgentBase):
    name = "PPO"

    def __init__(self, configs: PPOConfig, device: torch.device = torch.device("cpu")):
        super().__init__(configs, device)

        self.current_step = 0
        self.current_epsilon = self.configs.clip_epsilon

        # networks
        ## actor net
        self.actor_net = self.configs.actor_net(**self.configs.actor_kwargs).to(self.device)

        ## critic net
        self.critic_net = self.configs.critic_net(**self.configs.critic_kwargs).to(self.device)

        ## optimizer
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.actor_net.parameters(),
                    "lr": self.configs.lr,
                    "eps": self.configs.adam_epsilon,
                },
                {
                    "params": self.critic_net.parameters(),
                    "lr": self.configs.lr,
                    "eps": self.configs.adam_epsilon,
                },
            ]
        )

        ## buffer
        self.buffer = [
            RandomReplayBuffer(self.configs.horizon + 1, extra_items=["log_prob", "value"])
            for _ in range(self.configs.num_envs)
        ]

    def get_action(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)

        actions, log_probs = self.actor_net.action(states)
        values = self.critic_net(states)
        values = values.detach().cpu().numpy()

        return actions, log_probs, values

    def push(self, transitions: list):
        observations, states, actions, log_probs, values = transitions
        next_states, rewards, terminated, truncated, info = observations

        for i in range(self.configs.num_envs):
            state = states[i]
            action = actions[i]
            next_state = next_states[i]
            if "_final_observation" in info and info["_final_observation"][i]:
                next_state = info["_final_observation"][i]

            reward = rewards[i]
            done = int(terminated[i] or truncated[i])
            log_prob = log_probs[i]
            value = values[i]

            transition = (state, action, next_state, reward, done, log_prob, value)
            self.buffer[i].push(transition)

    def _compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation"""

        advantages = np.zeros_like(rewards)
        old_advantage = 0.0

        for t in reversed(range(len(rewards) - 1)):
            if dones[t]:
                old_advantage = 0.0

            delta = rewards[t] + self.configs.gamma * values[t + 1] * (1 - dones[t + 1]) - values[t]
            old_advantage = delta + self.configs.gamma * self.configs.gae_lambda * old_advantage
            advantages[t] = old_advantage

        advantages = np.expand_dims(advantages[:-1], axis=-1)
        value_targets = advantages + values[:-1]

        return advantages, value_targets

    def train(self):
        if len(self.buffer[0]) < self.configs.horizon + 1:
            return

        states, actions, log_probs, advantages, value_targets = None, None, None, None, None

        for buffer in self.buffer:
            batches = buffer.all()

            _advantages, _value_targets = self._compute_gae(
                batches["reward"], batches["value"], batches["done"]
            )

            states = (
                batches["state"][:-1]
                if states is None
                else np.concatenate((states, batches["state"][:-1]))
            )
            actions = (
                batches["action"][:-1]
                if actions is None
                else np.concatenate((actions, batches["action"][:-1]))
            )
            log_probs = (
                batches["log_prob"][:-1]
                if log_probs is None
                else np.concatenate((log_probs, batches["log_prob"][:-1]))
            )
            advantages = (
                _advantages if advantages is None else np.concatenate((advantages, _advantages))
            )
            value_targets = (
                _value_targets
                if value_targets is None
                else np.concatenate((value_targets, _value_targets))
            )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        value_targets = torch.FloatTensor(value_targets).to(self.device)

        # update networks with mini-batch
        idxs = np.arange(self.configs.horizon)

        for _ in range(self.configs.num_epochs):
            np.random.shuffle(idxs)

            for i in range(0, self.configs.horizon, self.configs.batch_size):
                minibatch_idx = idxs[i : i + self.configs.batch_size]

                advantage_batch = advantages[minibatch_idx]
                if self.configs.adv_norm:
                    advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                        advantage_batch.std() + 1e-8
                    )

                new_log_prob, entropy = self.actor_net.evaluate(
                    states[minibatch_idx], actions[minibatch_idx]
                )
                new_value = self.critic_net(states[minibatch_idx])

                if (
                    advantage_batch.shape != log_probs[minibatch_idx].shape
                    and log_probs[minibatch_idx].shape[1] > advantage_batch.shape[1]
                ):
                    old_log_prob = torch.sum(log_probs[minibatch_idx], dim=1, keepdim=True)
                    new_log_prob = torch.sum(new_log_prob, dim=1, keepdim=True)

                ratios = torch.exp(new_log_prob - old_log_prob)
                ratios = torch.reshape(ratios, advantage_batch.shape)

                loss_clip = torch.min(
                    ratios * advantage_batch,
                    torch.clamp(ratios, 1 - self.current_epsilon, 1 + self.current_epsilon)
                    * advantage_batch,
                ).mean()

                loss_vf = F.mse_loss(new_value, value_targets[minibatch_idx]).mean()

                loss_entropy = entropy.mean()

                loss = -(
                    loss_clip
                    - self.configs.vf_coef * loss_vf
                    + self.configs.entropy_coef * loss_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.configs.gradient_clip:
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                self.optimizer.step()

        if self.configs.max_step is not None:
            self.current_step += self.configs.horizon
            alpha = self.current_step / self.configs.max_step
            self.current_epsilon = self.configs.clip_epsilon * (1 - alpha)
            for g in self.optimizer.param_groups:
                g["lr"] = self.configs.lr * (1 - alpha)

        for i in range(self.configs.num_envs):
            self.buffer[i].clear()

        if self.configs.debug:
            return loss.item(), loss_clip.item(), loss_vf.item(), loss_entropy.item()

    def save(self, path: str):
        torch.save(
            {
                "actor_net": self.actor_net.state_dict(),
                "critic_net": self.critic_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "current_step": self.current_step,
                "current_epsilon": self.current_epsilon,
                "configs": self.configs,
            },
            path,
        )

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        for key in ["current_step", "current_epsilon", "configs"]:
            if key in checkpoint:
                setattr(self, key, checkpoint[key])
