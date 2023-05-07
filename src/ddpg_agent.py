
import random
import torch

class DDPGAgent:
    def __init__(self, policy_network, noisy_action):
        self.policy_network = policy_network
        self.noisy_action = noisy_action

    def compute_action(self, state, deterministic=True):
        state  = torch.FloatTensor(state)
        action = self.policy_network(state)  # Process state with the policy network to get an action

        if not deterministic:
            action = self.noisy_action.add_noise(action)

        return action.detach().cpu().numpy()[0]
    
# action_noise = GaussianActionNoise(0.5)
 # agent = DDPGAgent(policy_network, action_noise)
