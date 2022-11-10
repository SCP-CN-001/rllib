from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np

from rllib.algorithms.base.config import ConfigBase
from rllib.algorithms.base.agent import AgentBase
from rllib.utils.replay_buffer.replay_buffer import ReplayBuffer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PPOActor(nn.Module):
    def __init__(self):
        super().__init__()


class PPOCritic(nn.Module):
    def __init__(self):
        super().__init__()


class PPOConfig(ConfigBase):
    def __init__(self, configs):
        super().__init__()

        # model
        ## hyper-parameters

        ## actor net

        ## critic net
        

        self.lr_actor = self.lr
        self.lr_critic = self.lr
        self.adam_epsilon = 1e-8
        self.dist_type = "beta"
        self.hidden_size = 256

        self.clip_epsilon = 0.2
        self.lambda_ = 0.95
        self.var_max = 1

        # tricks
        self.adv_norm = True
        self.state_norm = True
        self.reward_norm = False
        self.reward_scaling = False
        self.gradient_clip = False
        self.policy_entropy = False
        self.entropy_coef = 0.01

        self.merge_configs(configs)

        if hasattr(self, "activation"):
            if self.activation == "tanh":
                activation = nn.Tanh()
            elif self.activation == "relu":
                activation = nn.ReLU()
            else:
                raise NotImplementedError
        else:
            activation = nn.Tanh()

        product_state = np.prod(np.array(self.state_dim))
        product_action = np.prod(np.array(self.action_dim))
        if not hasattr(self, "actor_layers"):
            if self.discrete or self.dist_type == "gaussian":
                self.actor_layers = [
                    ('flatten',nn.Flatten()),
                    ("linear1", nn.Linear(product_state, self.hidden_size)),
                    ("activate1", activation),
                    ("linear2", nn.Linear(self.hidden_size, self.hidden_size)),
                    ("activate2", activation),
                    ("linear3", nn.Linear(self.hidden_size, product_action))
                ]
            elif self.dist_type == "beta":
                self.actor_layers = [
                    ('flatten',nn.Flatten()),
                    ("linear1", nn.Linear(product_state, self.hidden_size)),
                    ("activate1", activation),
                    ("linear2", nn.Linear(self.hidden_size, self.hidden_size)),
                    ("activate2", activation),
                    ("linear3", nn.Linear(self.hidden_size, 2*product_action))
                ]
            else:
                raise NotImplementedError

        if not hasattr(self, "critic_layers"):
            self.critic_layers = [
                ('flatten',nn.Flatten()),
                ("linear1", nn.Linear(product_state, self.hidden_size)),
                ("activate1", activation),
                ("linear2", nn.Linear(self.hidden_size, self.hidden_size)),
                ("activate2", activation),
                ("linear3", nn.Linear(self.hidden_size, 1))
            ]


class PPO(AgentBase):
    """_summary_

    Args:
        AgentBase (_type_): _description_
    """
    def __init__(
        self, configs: dict, log_path: str, discrete: bool = False
    ) -> None:

        super().__init__(PPOConfig, configs)
        self.discrete = discrete

        # the networks
        self.actor_net = \
            Network(self.configs.actor_layers, self.configs.orthogonal_init).to(self.device)
        self.actor_optimizer = \
            torch.optim.Adam(
                self.actor_net.parameters(), 
                self.configs.lr_actor, 
                eps=self.configs.lr_actor
            )
        if self.configs.dist_type == "gaussian":
            self.log_std = \
                nn.Parameter(
                    torch.zeros(1, self.configs.action_dim), requires_grad=True
                ).to(self.device)

        self.critic_net = \
            Network(self.configs.critic_layers, self.configs.orthogonal_init).to(self.device)
        self.critic_optimizer = \
            torch.optim.Adam(
                self.critic_net.parameters(), 
                self.configs.lr_critic,
                eps=self.configs.adam_epsilon
            )
        self.critic_target = deepcopy(self.critic_net)

        # As a on-policy RL algorithm, PPO does not have memory, the self.memory represents
        # the buffer
        self.memory = ReplayMemory(self.configs.batch_size, ["log_prob"])

        # tricks
        if self.configs.state_norm:
            self.n_state = 0
            self.state_mean = np.zeros(self.configs.state_dim, dtype=np.float32)
            self.S = np.zeros(self.configs.state_dim, dtype=np.float32)
            self.state_std = np.sqrt(self.S)

        # save and load
        self.check_list = [
            ("configs", self.configs, 0),
            ("actor_net", self.actor_net, 1),
            ("actor_optimizer", self.actor_optimizer, 1),
            ("critic_net", self.critic_net, 1),
            ("critic_optimizer", self.critic_optimizer, 1),
            ("critic_target", self.critic_target, 1)
        ]
        if self.configs.dist_type == "gaussian":
            self.check_list.append(("log_std", self.log_std, 0))
        
    def state_norm(self, observation: np.ndarray):
        if self.n_state == 0:
            self.state_mean = observation
            self.state_std = observation
        else:
            old_mean = self.mean.copy()
            self.state_mean = old_mean + (observation - old_mean) / self.n_state
            self.S = self.S + (observation - old_mean) * (observation - self.state_mean)
            self.state_std = np.sqrt(self.S / self.n_state)
        observation = (observation - self.state_mean) / (self.state_std + 1e-8)
        observation = observation.transpose(2,0,1)
        return observation

    def get_action(self, observation: np.ndarray):
        '''Take action based on one observation. 

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            action: If self.discrete, the action is an (int) index. 
                If the action space is continuous, the action is an (np.ndarray).
            log_prob(np.ndarray): the log probability of taken action.
        '''
        observation = torch.FloatTensor(observation).to(self.device)
        if len(observation.shape) == len(self.configs.state_dim):
            observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            policy_dist = self.actor_net(observation)
            if len(policy_dist.shape) > 1 and policy_dist.shape[0] > 1:
                raise NotImplementedError
            if self.discrete:
                dist = Categorical(F.softmax(policy_dist, dim=1))
            elif self.configs.dist_type == "beta":
                alpha, beta = torch.chunk(policy_dist, 2, dim=-1)
                alpha = F.softplus(alpha) + 1.0
                beta = F.softplus(beta) + 1.0
                dist = Beta(alpha, beta)
            elif self.configs.dist_type == "gaussian":
                mean = torch.tanh(policy_dist)
                log_std = self.log_std.expand_as(mean)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
            else:
                raise NotImplementedError
                
        action = dist.sample()
        if not self.discrete and self.configs.dist_type == "gaussian":
                action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action)
        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return action, log_prob

    def _reward_norm(self, reward):
        return (reward - reward.mean()) / (reward.std() + 1e-8)

    def update(self):
        # convert batches to tensors
        batches = self.memory.shuffle()
        state_batch = torch.FloatTensor(batches["state"]).to(self.device)
        if self.discrete:
            state_batch = torch.IntTensor(state_batch)
        action_batch = torch.FloatTensor(batches["action"]).to(self.device)
        rewards = self._reward_norm(batches["reward"]) \
            if self.configs.reward_norm else batches["reward"] 
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        done_batch = torch.FloatTensor(batches["done"]).to(self.device)
        old_log_prob_batch = torch.FloatTensor(batches["log_prob"]).to(self.device)
        next_state_batch = torch.FloatTensor(batches["next_state"]).to(self.device)
        self.memory.clear()

        # GAE
        gae = 0
        adv = []

        with torch.no_grad():
            value = self.critic_net(state_batch)
            next_value = self.critic_net(next_state_batch)
            deltas = reward_batch + self.configs.gamma * (1 - done_batch) * next_value - value
            for delta, done in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done_batch.cpu().flatten().numpy())):
                gae = delta + self.configs.gamma * self.configs.lambda_ * gae * (1.0 - done)
                adv.append(gae)
            adv.reverse()
            adv = torch.FloatTensor(adv).view(-1, 1).to(self.device)
            v_target = adv + value
            if self.configs.adv_norm: # advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        
        for i in range(self.configs.batch_size):
            state = state_batch[i].unsqueeze(0)
            if self.discrete:
                dist = Categorical(self.actor_net(state))
                dist_entropy = dist.entropy().view(-1, 1)
                log_prob= dist.log_prob(action_batch[i].squeeze()).view(-1, 1)
            elif self.configs.dist_type == "beta":
                policy_dist = self.actor_net(state)
                alpha, beta = torch.chunk(policy_dist, 2, dim=-1)
                alpha = F.softplus(alpha) + 1.0
                beta = F.softplus(beta) + 1.0
                dist = Beta(alpha, beta)
                dist_entropy = dist.entropy().sum(1, keepdim=True)
                log_prob = dist.log_prob(action_batch[i])
            elif self.configs.dist_type == "gaussian":
                policy_dist = self.actor_net(state)
                mean = torch.clamp(policy_dist, -1, 1)
                log_std = self.log_std.expand_as(mean)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                dist_entropy = dist.entropy().sum(1, keepdim=True)
                log_prob = torch.sum(dist.log_prob(action_batch[i]))
            prob_ratio = (log_prob - old_log_prob_batch[i]).exp()

            loss1 = prob_ratio * adv[i]
            loss2 = torch.clamp(prob_ratio, 1 - self.configs.clip_epsilon, 1 + self.configs.clip_epsilon) * adv[i]

            actor_loss = - torch.min(loss1, loss2)
            if self.configs.policy_entropy:
                actor_loss += - self.configs.entropy_coef * dist_entropy
            critic_loss = F.mse_loss(v_target[i], self.critic_net(state))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.mean().backward()
            critic_loss.mean().backward()
            if self.configs.gradient_clip: # gradient clip
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                nn.utils.clip_grad_norm(self.actor_net.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step() 

        self._soft_update(self.critic_target, self.critic_net)

        if self.configs.lr_decay: # learning rate decay
            self.actor_optimizer.param_groups["lr"] = self.lr_decay(self.configs.lr_actor)
            self.critic_optimizer.param_groups["lr"] = self.lr_decay(self.configs.lr_critic)

        # for debug
        a = actor_loss.detach().cpu().numpy()[0][0]
        b = critic_loss.item()
        return a, b