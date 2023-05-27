from collections import deque
import numpy as np

from rllib.interface.buffer_base import BufferBase


class RandomReplayBuffer(BufferBase):
    """The random replay buffer.

    In the implementation, to save memory, we don't save the next state.
    """

    def __init__(self, buffer_size: int, extra_items: list = []):
        self.buffer_size = buffer_size
        self.items = ["state", "action", "reward", "done"] + extra_items
        self.buffer = {}
        for item in self.items:
            self.buffer[item] = deque([], maxlen=buffer_size)
        self.next_state = None

    def __len__(self):
        return len(self.buffer["state"])

    def push(self, observations: tuple, next_state):
        for i, item in enumerate(self.items):
            self.buffer[item].append(observations[i])

        self.next_state = next_state

    def get(self, idx_list: np.ndarray):
        batches = {}
        for name in self.items:
            batches[name] = []
        batches["next_state"] = []

        for idx in idx_list:
            for name in self.items:
                batches[name].append(self.buffer[name][idx])

            if idx + 1 == self.buffer_size:
                batches["next_state"].append(self.next_state)
            else:
                batches["next_state"].append(self.buffer["state"][idx + 1])

        for name in batches.keys():
            batches[name] = np.array(batches[name], dtype=np.float32)

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
            batches[key] = np.array(list(value), dtype=np.float32)

        batches["next_state"] = list(self.buffer["state"])[1:]
        batches["next_state"].append(self.next_state)
        batches["next_state"] = np.array(batches["next_state"], dtype=np.float32)

        return batches

    def clear(self):
        for item in self.items:
            self.buffer[item].clear()
