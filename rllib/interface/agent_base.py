#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: agent_base.py
# @Description: This script defines the base agent interface for all RL algorithms.
# @Time: 2023/10/17
# @Author: Yueyuan Li

from abc import ABC, abstractmethod


class AgentBase(ABC):
    """This abstract class provide a base agent for all RL algorithms."""

    def __init__(self, configs, device):
        """Initialize the model structure here"""
        self.configs = configs
        self.device = device
        self.buffer = None

    @abstractmethod
    def get_action(self, state):
        """Return an action based on the input state"""

    def push(self, transition: tuple):
        """Add a new memory transition to the buffer"""
        self.buffer.push(transition)

    @abstractmethod
    def train(self):
        """Update the network's parameters."""

    @abstractmethod
    def save(self, path: str):
        """Save the networks to the given path."""

    @abstractmethod
    def load(self, path: str):
        """Load the networks from the given path."""
