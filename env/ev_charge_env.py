import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EVChargeEnv(gym.Env):
    """
    EV charging environment.

    Goal:
      - Reach full battery (charge = 1.0)
      - Minimize cost
      - Avoid stressing the grid

    State (obs):
      [charge_level, price, grid_load, time_step_norm]

    Action:
      continuous charging rate in [0.0, 1.0]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 48, scenario: str = "medium"):
        super().__init__()

        # Scenario difficulty
        assert scenario in ["easy", "medium", "hard"]
        self.scenario = scenario

        # Observation: charge, price, load, time
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: charge rate between 0 and 1
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.max_steps = max_steps
        self.step_count = 0

        # Internal state
        self.charge = 0.0
        self.price = 0.0
        self.grid_load = 0.0

        # Scenario parameters (set in reset)
        self.base_price = 0.3
        self.base_load = 0.5
        self.load_threshold = 0.8  # above this → overload penalty
        self.charge_rate_scale = 0.08  # how fast battery fills

    def _set_scenario_params(self):
        """Set parameters based on difficulty scenario."""
        if self.scenario == "easy":
            self.base_price = 0.25
            self.base_load = 0.4
            self.load_threshold = 0.9
            self.charge_rate_scale = 0.10
        elif self.scenario == "medium":
            self.base_price = 0.30
            self.base_load = 0.5
            self.load_threshold = 0.85
            self.charge_rate_scale = 0.08
        else:  # hard
            self.base_price = 0.35
            self.base_load = 0.6
            self.load_threshold = 0.8
            self.charge_rate_scale = 0.06

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._set_scenario_params()

        self.step_count = 0
        # Random initial charge, slightly low
        self.charge = np.random.uniform(0.1, 0.4)
        # Start price/load around base with small noise
        self.price = np.clip(self.base_price + np.random.normal(0, 0.05), 0.0, 1.0)
        self.grid_load = np.clip(self.base_load + np.random.normal(0, 0.05), 0.0, 1.0)

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        time_step_norm = self.step_count / max(1, self.max_steps - 1)
        return np.array(
            [self.charge, self.price, self.grid_load, time_step_norm],
            dtype=np.float32,
        )

    def step(self, action):
        self.step_count += 1

        # Clamp action into valid range
        a = float(np.clip(action[0], 0.0, 1.0))

        # --- Dynamics ---
        # Battery charging
        self.charge += a * self.charge_rate_scale
        self.charge = float(np.clip(self.charge, 0.0, 1.0))

        # Price & load as noisy processes around base values
        self.price = float(
            np.clip(
                self.price * 0.7
                + self.base_price * 0.3
                + np.random.normal(0, 0.05),
                0.0,
                1.0,
            )
        )
        self.grid_load = float(
            np.clip(
                self.grid_load * 0.6
                + self.base_load * 0.4
                + np.random.normal(0, 0.07),
                0.0,
                1.0,
            )
        )

        # --- Reward ---
        # Progress reward
        progress = a * self.charge_rate_scale
        progress_reward = progress * 5.0  # scaled up

        # Cost penalty (higher price * more charging = worse)
        cost_penalty = self.price * a * 4.0

        # Grid overload penalty if we charge too much when load is high
        effective_load = self.grid_load + a * 0.2
        overload = max(0.0, effective_load - self.load_threshold)
        overload_penalty = overload * 6.0

        # Small time penalty to encourage faster completion
        time_penalty = 0.01

        reward = progress_reward - cost_penalty - overload_penalty - time_penalty

        # Episode done?
        terminated = self.charge >= 0.999
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "progress_reward": progress_reward,
            "cost_penalty": cost_penalty,
            "overload_penalty": overload_penalty,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"step={self.step_count} charge={self.charge:.3f} "
            f"price={self.price:.3f} load={self.grid_load:.3f}"
        )
