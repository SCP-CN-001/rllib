from rllib.replay_buffer.replay_buffer import ReplayBuffer
from rllib.utils.sum_tree import SumTree


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size: int, extra_items: list = ...):
        super().__init__(buffer_size, extra_items)
