import numpy as np

from rllib.interface.buffer_base import BufferBase


class RandomReplayBuffer(BufferBase):
    """The random replay buffer.
    """

    def __init__(self, buffer_size: int, extra_items: list = []):
        super().__init__(buffer_size)

        # The items to store are not initialized here, but in the push method.
        self.items = ["state", "action", "next_state", "reward", "done"] + extra_items
        for item in self.items:
            setattr(self, item, None)

        self.cnt = 0
        self.init = False

    def __len__(self):
        return min(self.cnt, self.buffer_size)

    def push(self, observations: tuple):
        # initialize the buffer
        if not self.init and self.cnt == 0:
            for i, item in enumerate(self.items):
                if hasattr(observations[i], "shape"):
                    setattr(self, item, np.empty((self.buffer_size, *observations[i].shape), dtype=np.float32))
                else:
                    setattr(self, item, np.empty((self.buffer_size, 1), dtype=np.float32))

            self.init = True

        # push the transition
        for i, item in enumerate(self.items):
            idx = self.cnt % self.buffer_size
            getattr(self, item)[idx] = observations[i]

        self.cnt += 1

    def get(self, idx_list: np.ndarray):
        batches = {}
        for item in self.items:
            batches[item] = getattr(self, item)[idx_list]

        return batches

    def sample(self, batch_size: int):
        idx_list = np.random.randint(self.__len__(), size=batch_size)
        return self.get(idx_list)

    def shuffle(self, idx_range: int = None):
        idx_range = self.__len__() if idx_range is None else idx_range
        idx_list = np.arange(idx_range)
        np.random.shuffle(idx_list)
        return self.get(idx_list)

    def all(self):
        idx_list = np.arange(self.__len__())
        return self.get(idx_list)

    def clear(self):
        for item in self.items:
            setattr(self, item, np.empty_like(getattr(self, item)))

        self.cnt = 0
