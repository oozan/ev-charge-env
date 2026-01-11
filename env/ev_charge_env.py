import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EVChargeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # State: [current_charge, grid_load, energy_price]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Action: charge rate (0.0 - 1.0)
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.charge = 0.0
        self.grid_load = np.random.random()
        self.price = np.random.random()
        obs = np.array([self.charge, self.grid_load, self.price], dtype=np.float32)
        return obs, {}

    def step(self, action):
        a = float(action[0])

        # Update charge level
        self.charge += a * 0.1
        self.charge = min(self.charge, 1.0)

        # Random fluctuations
        self.grid_load = np.random.random()
        self.price = np.random.random()

        # Reward: low price = good, high charge = good
        reward = (self.charge * 1.0) - (self.price * a)

        terminated = self.charge >= 1.0
        truncated = False

        obs = np.array([self.charge, self.grid_load, self.price], dtype=np.float32)
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Charge: {self.charge}, Load: {self.grid_load}, Price: {self.price}")
