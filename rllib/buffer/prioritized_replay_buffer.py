from rllib.interface import BufferBase
from rllib.utils.sum_tree import SumTree


class PrioritizedReplayBuffer(BufferBase):
    def __init__(self, buffer_size: int, extra_items: list = ...):
        super().__init__(buffer_size)
