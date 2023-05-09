import numpy as np
import torch

class HeuristicPendulumAgent:
  def __init__(self, env, fixed_torque = 1):
    self.state_size = env.observation_space.shape[0]
    self.action_size = env.action_space.shape[0]
    self.fixed_torque = fixed_torque
  def compute_action(self, state):
    x, y, theta = state
    if torch.is_tensor(state):

      action = np.zeros(len(x))
      for i in range(len(x)):
        if x[i] < 0:
          action[i] = np.sign(theta[i])*self.fixed_torque
        else:
          action[i] = -np.sign(theta[i])*self.fixed_torque
    else:
      if x < 0:
        action = np.sign(theta)*self.fixed_torque
      else:
        action = -np.sign(theta)*self.fixed_torque

    return action