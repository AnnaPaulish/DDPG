import numpy as np
import random

class OUActionNoise():
    def __init__(self, std_deviation=1):
        self.std_deviation = std_deviation
        self.noise = 0


    def get_noisy_action(self, action, theta = 0.5):
        
        noise = (1-theta)*self.noise + random.gauss(0, self.std_deviation)
        noisy_action = action + noise
        noisy_action = np.clip(noisy_action, -1., 1.)
        
        #update self.noise to contain most recent noise
        self.evolve_state(noise)
        return noisy_action
    
    def evolve_state(self, noise):
        self.noise = noise