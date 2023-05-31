import numpy as np

from rllib.interface import BufferBase


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data_pointer = 0

    @property
    def sum(self):
        return self.tree[0]
    
    @property
    def max_leaf(self):
        return np.max(self.tree[self.capacity - 1 :])

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_idx(self, value: float):
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

        idx = leaf_idx - self.capacity + 1
        return idx


class PrioritizedReplayBuffer(BufferBase):
    """This class implements the prioritized replay buffer.

    Attributes:
        alpha: The alpha used to calculate how much prioritization is used.
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
        alpha: float = 0.5,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
    ):
        super().__init__(buffer_size)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.initial_priority = 1.0
        self.priority = SumTree(self.buffer_size)
        self.priority_exponent = SumTree(self.buffer_size)

        self.data_pointer = 0
        # The items to store are not initialized here, but in the push method.
        self.items = ["state", "action", "next_state", "reward", "done"] + extra_items
        for item in self.items:
            setattr(self, item, None)

        self.cnt = 0
        self.init = False

    def __len__(self):
        return min(self.cnt, self.buffer_size)

    def push(self, transition: tuple):
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
        max_priority = np.max(self.priority.max_leaf, self.initial_priority)
        tree_idx = self.data_pointer + self.buffer_size - 1
        self.priority.update(tree_idx, max_priority)
        self.priority_exponent.update(tree_idx, max_priority ** self.alpha)

        for i, item in enumerate(self.items):
            getattr(self, item)[self.data_pointer] = transition[i]

        self.data_pointer += 1
        if self.data_pointer >= self.buffer_size:
            self.data_pointer = 0

        self.cnt += 1

    def sample(self, batch_size: int):
        # update beta
        self.beta = np.min([1, self.beta + self.beta_increment])

        # sample the batch
        priority_segment = self.priority_exponent.sum / batch_size
        batch_idx = np.empty((batch_size,), dtype=np.int32)
        batch = {}
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            idx = self.priority_exponent.get_idx(value)
            batch_idx[i] = idx

        for item in self.items:
            batch[item] = getattr(self, item)[batch_idx]

        # calculate the importance sampling weight
        probability = self.priority_exponent[self.batch_idx + self.buffer_size - 1] / self.priority.sum
        weight = np.power(self.buffer_size * probability, -self.beta)
        batch["weight"] = weight / np.max(weight)

        return batch, batch_idx
    
    def update_priority(self, batch_idx: np.ndarray, priority: np.ndarray):
        priority = priority + 1e-6

        for idx, p in zip(batch_idx, priority):
            self.priority.update(idx, p)
            self.priority_exponent.update(idx, p ** self.alpha)

    def clear(self):
        self.tree = np.zeros(2 * self.buffer_size - 1)
        self.data_pointer = 0

        for item in self.items:
            setattr(self, item, np.empty_like(getattr(self, item)))

        self.cnt = 0
