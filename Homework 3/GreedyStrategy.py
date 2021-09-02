import torch
import numpy as np
# Compute the greedy strategy
# There is a task here!

class GreedyStrategy():

    def __init__(self, bounds):
      """
      Initialize class

      bounds = upper and lower bounds for action
      low = lower bound for action
      high = upper bound for action
      ratio_noise_injected = noise in deterministic greedy policy for exploration
      """
      self.low, self.high = bounds
      self.ratio_noise_injected = 0

    def select_action(self, model, state):
      """
      Select greedy action

      model = policy model
      state = current state

      greedy_action = compute action from the policy
      action = action after clippping
      """
      with torch.no_grad():
          greedy_action = model(state).cpu().detach().data.numpy().squeeze()

      #TODO use the np.clip function to clip the greedy action between the values 'low' and 'high'
      action = np.clip(greedy_action, self.low, self.high)# To complete. You may find more information about the clip function in https://numpy.org/doc/stable/reference/generated/numpy.clip.html
      return np.reshape(action, self.high.shape)