import numpy as np

class HeuristicPendulumAgent:
  def __init__(self, env, fixed_torque = 1):
    self.state_size = env.observation_space.shape[0]
    self.action_size = env.action_space.shape[0]
    self.fixed_torque = fixed_torque
  def compute_action(self, state):
    x, y, theta = state
    if x < 0:
      action = np.sign(theta)*self.fixed_torque
    else:
      action = -np.sign(theta)*self.fixed_torque
    return action