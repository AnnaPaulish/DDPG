
import random
import torch

class DDPGAgent:
    def __init__(self, policy_network, noisy_action, target_policy_network):
        self.policy_network = policy_network
        self.target_policy_network = target_policy_network
        self.noisy_action = noisy_action

    def compute_action(self, state, device, deterministic=True, target = False):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)

        #network selection for action choice. 
        if target == False:
            action = self.policy_network(state)
        elif target == True:
            action = self.target_policy_network(state)


        if not deterministic:
            out = action.detach().cpu().numpy()[0]
            out = self.noisy_action.get_noisy_action(out)
        else:
            out = action[0]

        return out

    def update_target_params(self, network, target_network, Tau):
        for p_target, p in zip(target_network.parameters(), network.parameters()):
            p_target.data = Tau*p.data + (1-Tau)*p_target.data



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
