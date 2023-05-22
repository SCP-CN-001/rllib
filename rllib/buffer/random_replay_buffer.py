from collections import deque
import numpy as np

from rllib.interface.buffer_base import BufferBase


class RandomReplayBuffer(BufferBase):
    def __init__(self, buffer_size: int, extra_items: list = []):
        self.buffer_size = buffer_size
        self.cnt = 0
        self.items = ["state", "action", "reward", "done"] + extra_items
        self.buffer = {}
        for item in self.items:
            self.buffer[item] = deque([], maxlen=buffer_size)
        
    def __len__(self):
        return min(self.cnt, self.buffer_size)

    def push(self, observations: tuple):
        """Save a transition"""
        for i, item in enumerate(self.items):
            self.buffer[item].append(observations[i])

        self.cnt += 1

    def get(self, idx_list: np.ndarray):
        batches = {}
        for name in self.items:
            batches[name] = []
        batches["next_state"] = []

        for idx in idx_list:
            for name in self.items:
                batches[name].append(self.buffer[name][idx])
            batches["next_state"].append(self.buffer["state"][idx + 1])

        for name in batches.keys():
            batches[name] = np.array(batches[name])

        return batches

    def sample(self, batch_size: int):
        idx_list = np.random.randint(self.__len__() - 1, size=batch_size)
        return self.get(idx_list)

    def shuffle(self, idx_range: int = None):
        idx_range = self.__len__() if idx_range is None else idx_range
        idx_list = np.arange(idx_range)
        np.random.shuffle(idx_list)
        return self.get(idx_list)

    def all(self):
        batches = {}
        for key, value in self.buffer.items():
            batches[key] = np.array(list(value))
        return batches

    def clear(self):
        for item in self.items:
            self.buffer[item].clear()
