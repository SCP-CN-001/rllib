from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from model.agent_base import ConfigBase, AgentBase
from model.network import Network
from model.replay_memory import ReplayMemory


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SACConfig(ConfigBase):
    """Hyper parameters of the SAC model
    """
    def __init__(self, configs):
        super().__init__()
        # model
        self.tau = 1e-2
        self.gamma: float = 0.99
        self.batch_size:int = 128
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.adam_epsilon = 1e-8

        self.log_std_min = -20
        self.log_std_max = 2
        self.alpha = 1e-2
        self.initial_alpha = 1.0
        self.alpha_lr = 1e-3
        self.adaptive_entropy = True

        self.explore = True
        self.explore_configs = {
            "type": "epsilon_greedy",
            "epsilon": 0.1
        }

        self.hidden_size = 256
        self.memory_size = int(1e6)

        # tricks
        self.adaptive_alpha = True

        self.merge_configs(configs)

        if hasattr(self, "activation"):
            if self.activation == "tanh":
                activation = nn.Tanh()
            elif self.activation == "relu":
                activation = nn.ReLU()
            else:
                raise NotImplementedError
        else:
            activation = nn.ReLU()

        if not hasattr(self, "actor_layers"):
            self.actor_layers = [
                ("linear1", nn.Linear(self.state_dim, self.hidden_size)),
                ("activate1", activation),
                ("linear2", nn.Linear(self.hidden_size, self.hidden_size)),
                ("activate2", activation),
                ("linear3", nn.Linear(self.hidden_size, self.action_dim*2))
            ]
        
        if not hasattr(self, "critic_layers"):
            self.critic_layers = [
                ("linear1", nn.Linear(self.state_dim+self.action_dim, self.hidden_size)),
                ("activate1", activation),
                ("linear2", nn.Linear(self.hidden_size, self.hidden_size)),
                ("activate2", activation),
                ("linear3", nn.Linear(self.hidden_size, 1))
            ]


class SAC(AgentBase):
    """Soft Actor-Critic (SAC)
    Implementing based on the 2nd version of SAC paper 'Soft Actor-Critic Algorithms and Applications'
    """
    def __init__(self, configs: dict):
        super().__init__(SACConfig, configs)

        # Initialize networks
        ## critic nets
        self.q1_net = Network(deepcopy(self.configs.critic_layers)).to(device)
        self.q1_target_net = deepcopy(self.q1_net)

        self.q2_net = Network(deepcopy(self.configs.critic_layers)).to(device)
        self.q2_target_net = deepcopy(self.q2_net)

        ## actor net
        self.policy_net = Network(deepcopy(self.configs.actor_layers)).to(device)

        ## alpha
        self.alpha = self.configs.alpha

        ## optimizers
        self.q1_optimizer = torch.optim.Adam(
            self.q1_net.parameters(), self.configs.lr_critic, eps=self.configs.adam_epsilon
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2_net.parameters(), self.configs.lr_critic, eps=self.configs.adam_epsilon
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), self.configs.lr_actor, eps=self.configs.adam_epsilon
        )

        # Initialize the replay memory
        self.memory = ReplayMemory(self.configs.memory_size)

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)

        policy_dist = self.policy_net(state)
        mean, log_std = torch.chunk(policy_dist, 2, dim=-1)
        log_std = torch.clamp(log_std, self.configs.log_std_min, self.configs.log_std_max)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    def reparameterize_policy(self, state, epsilon=1e-6):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)

        policy_dist = self.policy_net(state)
        mean, log_std = torch.chunk(policy_dist, 2, dim=-1)
        log_std = torch.clamp(log_std, self.configs.log_std_min, self.configs.log_std_max)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(mean, std)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = normal.log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob

    def update(self):
        if len(self.memory) < self.configs.batch_size:
            return

        # Get memory batches
        batches = self.memory.sample(self.configs.batch_size)
        state = torch.FloatTensor(batches["state"]).to(device)
        action = torch.FloatTensor(batches["action"]).to(device)
        reward = torch.FloatTensor(batches["reward"]).unsqueeze(-1).to(device)
        next_state = torch.FloatTensor(batches["next_state"]).to(device)
        done = torch.FloatTensor(batches["done"]).unsqueeze(-1).to(device)

        with torch.autograd.set_detect_anomaly(True):
            # Soft Q loss
            with torch.no_grad():
                next_action, next_log_prob = self.reparameterize_policy(next_state)
                next_observation = torch.cat([next_state, next_action], 1)
                q1_target = self.q1_target_net(next_observation)
                q2_target = self.q2_target_net(next_observation)
                q_target = reward + done * self.configs.gamma * (torch.min(q1_target, q2_target) - self.alpha * next_log_prob)
            observation = torch.cat([state, action], 1)
            current_q1 = self.q1_net(observation)
            current_q2 = self.q2_net(observation)
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
            action_, log_prob = self.reparameterize_policy(state)
            observation_ = torch.cat([state, action_], 1)
            q1_value = self.q1_net(observation_)
            q2_value = self.q2_net(observation_)
            policy_loss = (self.alpha * log_prob - torch.min(q1_value, q2_value)).mean()

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Soft update target networks
            self._soft_update(self.q1_target_net, self.q1_net)
            self._soft_update(self.q2_target_net, self.q2_net)
