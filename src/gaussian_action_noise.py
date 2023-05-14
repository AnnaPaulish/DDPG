import numpy as np
import random

random.seed(10)

class GaussianActionNoise():
    def __init__(self, std_deviation=1):
        self.std_deviation = std_deviation

    def get_noisy_action(self, action):
        noise = random.gauss(0, self.std_deviation)
        noisy_action = action + noise
        noisy_action = np.clip(noisy_action, -1., 1.)

        return noisy_action