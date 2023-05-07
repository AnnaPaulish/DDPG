
import random

class DDPGAgent:
    def __init__(self, policy_network, noisy_action):
        self.policy_network = policy_network
        self.noisy_action = noisy_action

    def compute_action(self, state, deterministic=True):
        action = self.policy_network(state)  # Process state with the policy network to get an action

        if not deterministic:
            action = self.action_noise.add_noise(action)

        return action
    
# action_noise = GaussianActionNoise(0.5)
 # agent = DDPGAgent(policy_network, action_noise)
