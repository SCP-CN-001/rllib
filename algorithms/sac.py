from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from rllib.algorithms.base.agent import AgentBase
from rllib.algorithms.base.config import ConfigBase
from rllib.utils.replay_buffer.replay_buffer import ReplayBuffer


class SACActor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_size: int, 
        log_std_min: float, log_std_max: float, epsilon: float
    ):
        super(SACActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state: torch.Tensor)  -> torch.Tensor:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    def evaluate(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        """Implement the re-parameterization trick f()
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        z = z.to(mean.get_device())
        action = torch.tanh(mean + std * z)
        log_prob = normal.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + self.epsilon)

        return action, log_prob

class SACCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(SACCritic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = self.net(x)

        return x


class SACConfig(ConfigBase):
    """Configuration of the SAC model
    """
    def __init__(self, configs: dict):
        super().__init__()

        for key in ["state_space", "action_space"]:
            if key in configs:
                setattr(self, key, configs[key])
            else:
                raise AttributeError("[%s] is not defined for SACConfig!" % key)
        self.state_dim = self.state_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # model
        ## hyper-parameters
        self.gamma: float = 0.99
        self.batch_size: int = 128
        self.tau: float = 1e-2          # soft-update factor
        self.buffer_size: int = int(1e6)

        ## actor net
        self.lr_actor = 3e-4
        self.actor_net = SACActor
        self.actor_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 256,
            "log_std_min": -20,
            "log_std_max": 2,
            "epsilon": 1e-6
        }

        ## critic net
        self.lr_critic = 3e-4
        self.critic_net = SACCritic
        self.critic_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 256
        }

        ## alpha
        self.initial_alpha = 0.2
        self.lr_alpha = 3e-4

        # tricks

        self.merge_configs(configs)


class SAC(AgentBase):
    """Soft Actor-Critic (SAC)
    Implementing based on the 2nd version of SAC paper 'Soft Actor-Critic Algorithms and Applications'
    """
    def __init__(self, configs: dict):
        super().__init__(SACConfig, configs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # networks
        ## actor net
        self.policy_net = self.configs.actor_net(**self.configs.actor_kwargs).to(self.device)

        ## critic nets
        self.q1_net = self.configs.critic_net(**self.configs.critic_kwargs).to(self.device)
        self.q1_target_net = deepcopy(self.q1_net)

        self.q2_net = self.configs.critic_net(**self.configs.critic_kwargs).to(self.device)
        self.q2_target_net = deepcopy(self.q2_net)

        ## alpha
        self.alpha = self.configs.initial_alpha

        ## optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), self.configs.lr_critic)
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), self.configs.lr_critic)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), self.configs.lr_actor
        )
        
         # the replay buffer
        self.buffer = ReplayBuffer(self.configs.buffer_size)

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        action = self.policy_net.action(state)

        return action

    def soft_update(self, target_net, current_net):
        for target, current in zip(target_net.parameters(), current_net.parameters()):
            target.data.copy_(current.data * self.configs.tau + target.data * (1. - self.configs.tau))

    def train(self):
        if len(self.buffer) < self.configs.batch_size:
            return

        batches = self.buffer.sample(self.configs.batch_size)
        state = torch.FloatTensor(batches["state"]).to(self.device)
        action = torch.FloatTensor(batches["action"]).to(self.device)
        reward = torch.FloatTensor(batches["reward"]).unsqueeze(-1).to(self.device)
        next_state = torch.FloatTensor(batches["next_state"]).to(self.device)
        done = torch.FloatTensor(batches["done"]).unsqueeze(-1).to(self.device)

        # Soft Q loss
        with torch.no_grad():
            next_action, next_log_prob = self.policy_net.evaluate(next_state)
            q1_target = self.q1_target_net(next_state, next_action)
            q2_target = self.q2_target_net(next_state, next_action)
            q_target = reward + done * self.configs.gamma * (torch.min(q1_target, q2_target) - self.alpha * next_log_prob)
        current_q1 = self.q1_net(state, action)
        current_q2 = self.q2_net(state, action)
        q1_loss = F.mse_loss(current_q1, q_target.detach())
        q2_loss = F.mse_loss(current_q2, q_target.detach())

        # Update critic networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy loss
        action_, log_prob = self.policy_net.evaluate(state)
        q1_value = self.q1_net(state, action_)
        q2_value = self.q2_net(state, action_)
        policy_loss = (self.alpha * log_prob - torch.min(q1_value, q2_value)).mean()

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update target networks
        self.soft_update(self.q1_target_net, self.q1_net)
        self.soft_update(self.q2_target_net, self.q2_net)

    def evaluate(self):
        return super().evaluate()