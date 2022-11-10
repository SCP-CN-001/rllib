


from rllib.algorithms.base.config import ConfigBase
from rllib.algorithms.base.agent import AgentBase


class DDPGConfig(ConfigBase):
    def __init__(self, configs: dict):
        super().__init__()


class DDPG(AgentBase):
    def __init__(self, configs: dict):
        super().__init__(DDPGConfig, configs)
        