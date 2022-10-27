class ConfigBase:
    def __init__(self):
        # runtime
        self.n_epoch = None
        self.n_initial_exploration_steps = None

        # environment
        self.observation_space = None
        self.action_space = None
        self.state_dim: tuple = None
        self.action_dim: tuple = None

        # model
        self.gamma: float = 0.99
        self.batch_size:int = 128
        self.lr: float = 1e-3
        self.tau: float = 1e-3 # when tau=0, the update becomes hard update
        self.max_train_steps = 1e6

        # explore
        self.explore: bool = True
        self.explore_config: dict = {
            "type": "StochasticSampling",
        }

        # tricks
        self.orthogonal_init = True
        self.lr_decay = False

        # evaluation
        self.evaluation_interval = None

        # save and load
        self.check_list = []
    
    def merge_configs(self, configs: dict):
        for key, value in configs.items():
            setattr(self, key, value)