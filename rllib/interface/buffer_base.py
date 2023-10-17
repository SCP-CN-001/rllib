#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: buffer_base.py
# @Description: This script defines the base buffer interface for all RL algorithms.
# @Time: 2023/10/17
# @Author: Yueyuan Li

from abc import ABC, abstractmethod


class BufferBase(ABC):
    """This abstract class provide a base buffer for all RL algorithms."""

    def __init__(self, buffer_size: int):
        """Initialize the buffer structure here"""
        self.buffer_size = buffer_size

    @abstractmethod
    def push(self, transition: tuple):
        """Add a new memory transition to the buffer"""

    @abstractmethod
    def sample(self, batch_size: int):
        """Sample a batch of memory transitions from the buffer"""

    @abstractmethod
    def clear(self):
        """Clear the buffer"""
