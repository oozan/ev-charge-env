import numpy as np

class BaselineAgent:
    def select_action(self, observation):
        return np.array([np.random.random()], dtype=np.float32)
