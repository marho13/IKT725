import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
#End states = [5, 7, 11, 12]
env = gym.make("FrozenLake-v0")
env.render()
action_size = env.action_space.n
state_size = env.observation_space.n
print('Action space size:', action_size)
print('State space size:', state_size)

total_episodes = 10000       # Total episodes
learning_rate = 0.95          # Learning rate
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005            # Exponential decay rate for exploration probability

lrs = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.30, 0.25, 0.20]

class QlearningGymenv:
    def __init__(self, total_episodes, learning_rate, max_steps, gamma, epsilon,
                 max_epsilon, min_epsilon, decay_rate, env, action_size, state_size,
                 static, boltzman, Temp):
        random.seed(9001)
        env.seed(9001)
        np.random.seed(9001)
        self.static = static
        self.total = total_episodes
        self.lr = learning_rate
        self.steps = max_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.max = max_epsilon
        self.min = min_epsilon
        self.decay = decay_rate
        self.env = env
        self.action_size = action_size
        self.state_size = state_size
        self.boltzman = boltzman
        self.temp = Temp
        self.hole = [5, 7, 11, 12]
        self.rewards, self.qtable = self.getRewQ()


    def getRewQ(self):
        rewards = []
        qtable = np.zeros((state_size, action_size))
        return rewards, qtable

    def performEpisodes(self):
        print("Working with {} Learning rate".format(self.static))
        # 2 For life or until learning is stopped
        for episode in range(total_episodes):
            if type(lrs) == type([]):
                learning_rate = lrs[episode//1000]
            else:
                learning_rate = 0.95
            self.runEpisode(episode, learning_rate)


    def runEpisode(self, episode, learning_rate):
        # Reset the environment
        state = env.reset()
        total_rewards = 0
        for step in range(self.steps):
            action = self.performAction(state)
            # print(state)
            # 4. Take action A, observe R,S'
            new_state, reward, done, info = self.env.step(action)

            # 5. Update Q(S,A):= Q(S,A) + lr * [R + gamma * max Q(S',a) - Q(S,A)]
            self.qtable[state, action] = self.qtable[state, action] + learning_rate * (
                        reward + gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])

            total_rewards += reward

            # New current state
            state = new_state

            # If done (if we're dead) : finish episode
            if done == True:
                # print("Ended in state:", state)
                break

        # Reduce epsilon (less exploration)
        self.epsilon = self.min + (self.max - self.min) * np.exp(-decay_rate * episode)
        self.rewards.append(total_rewards)

    def performAction(self, state):
        if self.boltzman:
            probabilities = [0.0,0.0,0.0,0.0]
            for p in range(len(probabilities)):
                expectedRew = self.qtable[state, p]/self.temp
                sumy = 0.0
                for a in range(self.action_size):
                    sumy += (self.qtable[state, a]/self.temp)
                probabilities[p] = expectedRew/sumy

            action = np.argmax(probabilities)
            for x in range(self.action_size):
                if action != x:
                    self.qtable[state][x] += 0.0001
            # print(probabilities)
        else:
            exp_exp_tradeoff = np.random.uniform(0, 1)

            ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > self.epsilon:
                action = np.argmax(self.qtable[state, :])

            # Else doing a random choice --> exploration
            else:
                action = self.env.action_space.sample()
        return action


    def printy(self):
        print('Score over time: ' + str(sum(self.rewards) / self.total))
        for a in range(len(self.qtable)):
            print(np.argmax(self.qtable[a]), end=", ")
        print()

    def plot(self):
        rew = []

        for x in range(len(self.rewards)//100):
            tempSum = sum(self.rewards[x*100:(x+1)*100])
            rew.append(tempSum/100)

        fig, ax = plt.subplots()
        ax.plot(rew)
        ax.set_xlabel('Episodes/100')
        ax.set_ylabel('Reward')

        # fig.tight_layout()
        plt.show()

qstaticboltz = QlearningGymenv(total_episodes, learning_rate, max_steps, gamma, epsilon,
                 max_epsilon, min_epsilon, decay_rate, env, action_size, state_size,
                          "Static", True, 0.8)
qstatic = QlearningGymenv(total_episodes, learning_rate, max_steps, gamma, epsilon,
                 max_epsilon, min_epsilon, decay_rate, env, action_size, state_size,
                          "Static", False, 0.8)
qVarying = QlearningGymenv(total_episodes, lrs, max_steps, gamma, epsilon,
                 max_epsilon, min_epsilon, decay_rate, env, action_size, state_size,
                           "Varying", False, 0.8)
qVaryingboltz = QlearningGymenv(total_episodes, lrs, max_steps, gamma, epsilon,
                 max_epsilon, min_epsilon, decay_rate, env, action_size, state_size,
                           "Varying", True, 0.8)

qstatic.performEpisodes()
qstatic.printy()
qstatic.plot()

qVarying.performEpisodes()
qVarying.printy()
qVarying.plot()

qstaticboltz.performEpisodes()
qstaticboltz.printy()
qstaticboltz.plot()

qVaryingboltz.performEpisodes()
qVaryingboltz.printy()
qVaryingboltz.plot()
