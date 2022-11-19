from abc import ABC, abstractmethod


from rllib.algorithms.base.config import ConfigBase


class AgentBase(ABC):
    def __init__(self, config_type: ConfigBase, configs:dict):
        """Initialize the model structure here
        """
        self.configs = config_type(configs)

    @abstractmethod
    def get_action(self, state):
        """Return an action based on the input state
        """

    @abstractmethod
    def train(self):
        """Update the network's parameters.
        """

    @abstractmethod
    def save(self, checkpoint_dir: str, param_only: bool = True):
        """Store the model structure and corresponding parameters to a file.
        """

    @abstractmethod
    def load(self, checkpoint_dir: str, param_only: bool = True) -> None:
        """Load the model structure and corresponding parameters from a file.
        """