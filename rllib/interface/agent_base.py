from abc import ABC, abstractmethod


class AgentBase(ABC):
    """This abstract class provide a base agent for all RL algorithms."""

    def __init__(self, configs, device):
        """Initialize the model structure here"""
        self.configs = configs
        self.device = device

    @abstractmethod
    def get_action(self, state):
        """Return an action based on the input state"""

    @abstractmethod
    def train(self):
        """Update the network's parameters."""

    @abstractmethod
    def save(self, path: str):
        """Save the networks to the given path."""

    @abstractmethod
    def load(self, path: str):
        """Load the networks from the given path."""
