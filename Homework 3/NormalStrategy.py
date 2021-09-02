import numpy as np
import torch
# Compute noise for exploration
# There is a task here!

class NormalNoiseStrategy():
    def __init__(self, bounds, exploration_noise_ratio=0.1):
        """
        class initialization

        bounds = upper and lower bounds for noise
        exploration_noise_ratio = noise exploration ration
        """
        self.low, self.high = bounds
        self.exploration_noise_ratio = exploration_noise_ratio
        self.ratio_noise_injected = 0

    def select_action(self, model, state, max_exploration=False):
        """
        select DDPG policy

        model = policy model
        state = current state
        max_exploration = exploration strategy options

        noise_scale = standard deviation for normal distribution
        noise = noise for exploration
        noisy_action = DDPG action with exploration noise
        action = clipped DDPG action
        """
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.exploration_noise_ratio * self.high

        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))

        # TODO: compute the DDPG action with an exploration strategy. Use the values greedy_action and noise
        noisy_action =  greedy_action*noise# To complete, incomplete

        # TODO use the np.clip function to clip the DDPG action between the values 'self.low' and 'self.high'
        action = np.clip(noisy_action, self.low, self.high) # To complete. You may find more information about the clip function in https://numpy.org/doc/stable/reference/generated/numpy.clip.html

        self.ratio_noise_injected = np.mean(abs((greedy_action - action) / (self.high - self.low)))
        return action