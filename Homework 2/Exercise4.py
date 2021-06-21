import gym
import numpy as np
import random
import torch
import torch.hub
import torch.autograd.variable as Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from collections import Counter
from prioritisedExperienceReplay import Memory

device = 'cpu'
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        current_rate = self.start ** (current_step//self.decay)
        if current_rate < self.end:
            return self.end
        else:
            return current_rate

class policyNetworks(nn.Module):
    def __init__(self, stateDim, outputSize, n_latent_var):
        super().__init__()
        self.strategy = EpsilonGreedyStrategy(0.99, 0.05, 3000)
        # self.device = device
        self.randPolicy = {"Rand":0, "Policy":0}
        self.current_step = 0
        self.num_actions = outputSize
        self.fc1 = nn.Linear(in_features=stateDim, out_features=n_latent_var).float()
        self.fc2 = nn.Linear(in_features=n_latent_var, out_features=n_latent_var).float()
        self.out = nn.Linear(in_features=n_latent_var, out_features=outputSize).float()


    def forward(self, t):
        t = t.float()
        t = self.fc1(t).float()
        t = torch.tanh(t).float()
        t = self.fc2(t).float()
        t = torch.tanh(t).float()
        t = self.out(t).float()
        return torch.tanh(t)

    def act(self, state):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            self.randPolicy["Rand"] += 1
            action = random.randrange(self.num_actions)
            output = []
            for a in range(self.num_actions):
                if a != (action-1):
                    output.append(0.0)
                else:
                    output.append(1.0)
            return torch.tensor(output).to('cpu').detach().numpy() # explore
        else:
            self.randPolicy["Policy"] += 1
            state = torch.tensor(state, dtype=torch.float32)
            # state = torch.from_numpy(state)
            with torch.no_grad():
                return self.forward(state).to('cpu').detach().numpy() # exploit



class DQN():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        # torch.cuda.set_device(0)
        self.lr = lr
        self.betas = betas
        self.gamma = torch.tensor(gamma)

        self.policy_net = policyNetworks(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, betas=betas)
        self.target_net = policyNetworks(state_dim, action_dim, n_latent_var).to(device)
        self.policy_net = self.policy_net.float()
        self.target_net = self.target_net.float()

        self.MseLoss = nn.MSELoss()


    def update(self, memory, BATCH_SIZE):
            batch, idx, weight = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*batch))

            state_batch = torch.stack(batch.state, dim=0)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            next_state = torch.stack(batch.next_state, dim=0)

            state_action_values = self.policy_net(state_batch).gather(-1, action_batch.unsqueeze(-1)).squeeze(-1)
            next_state_values = torch.max(self.target_net(next_state), 1)

            # Compute the expected Q values
            expected_state_action_values = (next_state_values[0] * self.gamma) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

def append_sample(model, state, action, reward, next_state, done, memory, target_model):
    global discount
    target = model(Variable(torch.FloatTensor(state))).data
    old_val = target[action]
    target_val = target_model(Variable(torch.FloatTensor(next_state))).data
    if done == True:
        target[action] = reward
    else:
        target[action] = reward + 0.99 * torch.max(target_val)

    error = abs(old_val - target[action]) + 0.00001
    #actionTensor = torch.nn.functional.one_hot(torch.tensor([action]), env.action_space.n)
    memory.add(error, (torch.from_numpy(state), torch.LongTensor([action]), torch.from_numpy(next_state), torch.tensor([float(reward)])))

env = gym.make("LunarLander-v2")
env.seed(42)
model = DQN(env.observation_space.shape[0], env.action_space.n, env.observation_space.shape[0], 0.1, [0.99, 0.999], 0.99)
memory = Memory(capacity=1000)
state = env.reset()
maxSteps = 1000

print(env)

def train(model, env, memory):
    # state = env.reset()
    for episode in range(1000):
        reward = 0.0
        state = env.reset()
        for s in range(maxSteps):
            action = model.policy_net.act(torch.from_numpy(state))

            new_state, rew, done, info = env.step(np.argmax(action))
            reward += rew
            append_sample(model.policy_net, state, np.argmax(action), rew, new_state, done, memory, model.target_net)
            if done:
                break

        model.update(memory, 32)
        if episode%50 == 0:
            print("Episode {} gave a reward of {}".format(episode, reward))


train(model, env, memory)
