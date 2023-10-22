#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: observation_wrapper.py
# @Description: This script implements some common observation wrappers.
# @Time: 2023/10/17
# @Author: Yueyuan Li

try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python package is required for this functionality. Run `pip install opencv-python` to install."
    )

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class GrayscaleObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], 4), dtype=np.uint8
        )

    def observation(self, observation):
        grayscale_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        grayscale_observation = np.expand_dims(grayscale_observation, -1)
        observation = np.dstack([observation, grayscale_observation])
        observation = np.transpose(observation, (2, 0, 1))
        return observation


class ScaleObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32) / 255.0
        return observation
