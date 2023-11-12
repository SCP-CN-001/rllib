#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: config_base.py
# @Description: This script defines a base config interface for all RL algorithms.
# @Time: 2023/10/17
# @Author: Yueyuan Li

import gymnasium as gym


class ConfigBase:
    """This class provide a base config for all RL algorithms."""

    def __init__(self):
        # runtime
        self.n_epoch = None
        self.n_initial_exploration_steps = None
        self.debug = False

        # environment
        self.state_space = None
        self.action_space = None
        self.state_dim = None
        self.action_dim = None

        # model
        self.gamma: float = 0.99  # reward discount factor
        self.batch_size: int = 128  # batch size
        self.lr: float = 1e-3  # learning rate

        # explore
        self.explore: bool = True
        self.explore_config: dict = {"type": "StochasticSampling"}

        # evaluation
        self.evaluation_interval = None
        self.evaluation_duration = 10
        self.evaluation_duration_unit = "episodes"

        # debug
        self.log_level = "verbose"
        self.seed = None

    def set_env(self, configs):
        """Set the environment related configs"""

        for key in ["state_space", "action_space"]:
            if key in configs:
                setattr(self, key, configs[key])
            else:
                raise AttributeError("[%s] is not defined!" % key)

        if "state_dim" not in configs.keys():
            if isinstance(self.state_space, gym.spaces.Box):
                self.state_dim = self.state_space.shape[0]
            elif isinstance(self.state_space, gym.spaces.Discrete):
                self.state_dim = self.state_space.n
        else:
            self.state_dim = configs["state_dim"]

        if "action_dim" not in configs.keys():
            if isinstance(self.action_space, gym.spaces.Box):
                self.action_dim = self.action_space.shape[0]
            elif isinstance(self.action_space, gym.spaces.Discrete):
                self.action_dim = self.action_space.n
        else:
            self.action_dim = configs["action_dim"]

    def merge_configs(self, configs: dict):
        """Merge the custom configs for a specific algorithm

        Args:
            configs (dict): the custom configs
        """
        for key, value in configs.items():
            setattr(self, key, value)
