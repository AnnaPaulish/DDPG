
import random
import torch

class DDPGAgent:
    def __init__(self, policy_network, noisy_action):
        self.policy_network = policy_network
        self.noisy_action = noisy_action

    def compute_action(self, state, device, deterministic=True):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.policy_network(state)

        if not deterministic:
            out = action.detach().cpu().numpy()[0]
            out = self.noisy_action.get_noisy_action(out)
        else:
            out = action[0]

        return out

        # if not torch.is_tensor(state):
        #     state  = torch.FloatTensor(state).to(device)
        # with torch.no_grad():
        #     action = self.policy_network(state)  # Process state with the policy network to get an action
        # #action = action.detach().cpu().numpy()

        # if not deterministic:
        #     # action = action.detach().cpu().numpy()
                        
        #         action = self.noisy_action.get_noisy_action(action[0])
        # return action
    
# action_noise = GaussianActionNoise(0.5)
 # agent = DDPGAgent(policy_network, action_noise)
