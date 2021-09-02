import torch
import numpy as np
# Noise decay strategy for exploation
# There is a task here!
class NormalNoiseDecayStrategy():
    def __init__(self, bounds, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=10000):
      """
      Initialize class

      bounds = max-min bounds for noise
      init_noise_ratio = initial noise ratio
      min_noise_ratio = minimum noise ratio
      decay_steps=10000 = noise decay steps
      """
      self.t = 0
      self.low, self.high = bounds
      self.noise_ratio = init_noise_ratio
      self.init_noise_ratio = init_noise_ratio
      self.min_noise_ratio = min_noise_ratio
      self.decay_steps = decay_steps
      self.ratio_noise_injected = 0

    def _noise_ratio_update(self):
      """
      Update noise ratio
      """
      noise_ratio = 1 - self.t / self.decay_steps
      noise_ratio = (self.init_noise_ratio - self.min_noise_ratio) * noise_ratio + self.min_noise_ratio
      noise_ratio = np.clip(noise_ratio, self.min_noise_ratio, self.init_noise_ratio)
      self.t += 1
      return noise_ratio

    def select_action(self, model, state, max_exploration=False):
      """
      Select noisy action for exploration

      state = state in the environment
      max_exploration = noise scale selection

      """
      if max_exploration:
          noise_scale = self.high
      else:
          noise_scale = self.noise_ratio * self.high

      with torch.no_grad():
          greedy_action = model(state).cpu().detach().data.numpy().squeeze()

      noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))

      #TODO: compute the TD3 action with an exploration strategy. Use the values greedy_action and noise
      noisy_action = greedy_action.detach()*noise #To complete

      #TODO use the np.clip function to clip the TD3 action between the values 'self.low' and 'self.high'
      action = np.clip(noisy_action, self.low, self.high)# To complete. You may find more information about the clip function in https://numpy.org/doc/stable/reference/generated/numpy.clip.html

      self.noise_ratio = self._noise_ratio_update()
      self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
      return action