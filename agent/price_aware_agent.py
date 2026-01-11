import numpy as np


class PriceAwareAgent:
    """
    Heuristic agent for EVChargeEnv.

    - Charges more when price is low and grid load is safe.
    - Charges less when price is high or grid load is high.
    """

    def __init__(self,
                 low_price_threshold: float = 0.4,
                 high_price_threshold: float = 0.7,
                 high_load_threshold: float = 0.8):
        self.low_price_threshold = low_price_threshold
        self.high_price_threshold = high_price_threshold
        self.high_load_threshold = high_load_threshold

    def select_action(self, observation):
        """
        observation = [charge, price, load, time_step_norm]
        returns: np.array([action]) in [0, 1]
        """
        charge, price, load, t = observation

        # If almost full, stop charging.
        if charge >= 0.98:
            return np.array([0.0], dtype=np.float32)

        # If grid is very stressed, back off.
        if load >= self.high_load_threshold:
            return np.array([0.1], dtype=np.float32)

        # If price is low, charge aggressively.
        if price <= self.low_price_threshold:
            return np.array([0.9], dtype=np.float32)

        # If price is very high, charge slowly, just enough to make progress.
        if price >= self.high_price_threshold:
            return np.array([0.2], dtype=np.float32)

        # Medium case: moderate charging.
        return np.array([0.5], dtype=np.float32)
