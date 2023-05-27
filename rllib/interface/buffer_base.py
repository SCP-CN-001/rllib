from abc import ABC, abstractmethod

import numpy as np


class BufferBase(ABC):
    """This abstract class provide a base buffer for all RL algorithms."""

    def __init__(self, buffer_size: int):
        """Initialize the buffer structure here"""
        self.buffer_size = buffer_size

    @abstractmethod
    def push(self, memory: tuple):
        """Add a new memory item to the buffer"""

    @abstractmethod
    def get(self, idx: np.ndarray):
        """Get an arbitrary batch of memory item from the buffer"""

    @abstractmethod
    def sample(self, batch_size: int):
        """Sample a batch of memory items from the buffer"""
