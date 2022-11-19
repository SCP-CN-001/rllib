import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(
        self, mu: float, theta: float, sigma: float, dt: float,
        x0: np.ndarray = None
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.x_prev.shape)
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros.like(self.mu)
    