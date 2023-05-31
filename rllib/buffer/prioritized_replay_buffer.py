import numpy as np

from rllib.interface import BufferBase


class PrioritizedReplayBuffer(BufferBase):
    """This class implements the prioritized replay buffer.

    Attributes:
        exponent: The exponent used to calculate the priority.
        beta: The beta used to calculate the importance sampling weight. The value of beta will slowly increase to 1.
        beta_increment: The increment of beta.
        tree: The sum tree used to store the priorities.
        data_pointer: The pointer to the current position in the buffer.
        items: The items to store in the buffer.
        cnt: record the true length of the buffer.
    """

    def __init__(
        self,
        buffer_size: int,
        extra_items: list = [],
        exponent: float = 0.5,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
    ):
        super().__init__(buffer_size)

        self.exponent = exponent
        self.beta = beta
        self.beta_increment = beta_increment

        self.tree = np.zeros(2 * self.buffer_size - 1)
        self.data_pointer = 0
        # The items to store are not initialized here, but in the push method.
        self.items = ["state", "action", "next_state", "reward", "done"] + extra_items
        for item in self.items:
            setattr(self, item, None)

        self.cnt = 0
        self.init = False

    def __len__(self):
        return min(self.cnt, self.buffer_size)

    @property
    def total_priority(self):
        return self.tree[0]

    def _update_priority(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def push(self, transition: tuple, priority: float):
        # initialize the buffer
        if not self.init and self.cnt == 0:
            for i, item in enumerate(self.items):
                if hasattr(transition[i], "shape"):
                    setattr(
                        self,
                        item,
                        np.empty((self.buffer_size, *transition[i].shape), dtype=np.float32),
                    )
                else:
                    setattr(self, item, np.empty((self.buffer_size, 1), dtype=np.float32))

            self.init = True

        # push the transition
        tree_idx = self.data_pointer + self.buffer_size - 1
        self._update_priority(tree_idx, priority)
        for i, item in enumerate(self.items):
            getattr(self, item)[self.data_pointer] = transition[i]

        self.data_pointer += 1
        if self.data_pointer >= self.buffer_size:
            self.data_pointer = 0

        self.cnt += 1

    def _get_idx(self, value: float):
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_idx]:
                    parent_idx = left_idx
                else:
                    value -= self.tree[left_idx]
                    parent_idx = right_idx

        idx = leaf_idx - self.buffer_size + 1
        return idx

    def sample(self, batch_size: int):
        self.beta = np.min([1, self.beta + self.beta_increment])
        priority_segment = self.total_priority / batch_size
        batch_idx = np.empty((batch_size,), dtype=np.int32)
        batch = {}

        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            idx = self._get_idx(value)
            batch_idx[i] = idx

        for item in self.items:
            batch[item] = getattr(self, item)[batch_idx]

        return batch

    def clear(self):
        self.tree = np.zeros(2 * self.buffer_size - 1)
        self.data_pointer = 0

        for item in self.items:
            setattr(self, item, np.empty_like(getattr(self, item)))

        self.cnt = 0
