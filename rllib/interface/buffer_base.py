from abc import ABC, abstractmethod

class BufferBase(ABC):
    """This abstract class provide a base buffer for all RL algorithms."""
    def __init__(self, buffer_size: int):
        """Initialize the buffer structure here"""
        self.buffer_size = buffer_size

    @abstractmethod
    def push(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer"""

    @abstractmethod
    def sample(self):
        """Return a batch of transitions from the buffer"""