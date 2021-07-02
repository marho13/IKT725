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
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        current_rate = self.start ** (current_step // self.decay)
        if current_rate < self.end:
            return self.end
        else:
            return current_rate


class policyNetworks(nn.Module):
    def __init__(self, stateDim, outputSize, n_latent_var):
        super().__init__()
        self.strategy = EpsilonGreedyStrategy(0.99, 0.05, 3000)
        # self.device = device
        self.randPolicy = {"Rand": 0, "Policy": 0}
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
                if a != (action - 1):
                    output.append(0.0)
                else:
                    output.append(1.0)
            return torch.tensor(output).to('cpu').detach().numpy()  # explore
        else:
            self.randPolicy["Policy"] += 1
            state = torch.tensor(state, dtype=torch.float32)
            # state = torch.from_numpy(state)
            with torch.no_grad():
                return self.forward(state).to('cpu').detach().numpy()  # exploit


class DQN():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, optim):
        # torch.cuda.set_device(0)
        self.lr = lr
        self.betas = betas
        self.gamma = torch.tensor(gamma)

        self.policy_net = policyNetworks(state_dim, action_dim, n_latent_var).to(device)
        self.target_net = policyNetworks(state_dim, action_dim, n_latent_var).to(device)
        self.policy_net = self.policy_net.float()
        self.target_net = self.target_net.float()

        self.optimizer = optim(self.policy_net.parameters(), lr=lr)

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
    memory.add(error, (torch.from_numpy(state), torch.LongTensor([action]), torch.from_numpy(next_state),
                       torch.tensor([float(reward)])))


def hyperparameters():
    rewards = []
    files = []
    lr = [0.1, 0.001, 0.0001]
    unitsFully = [8, 32, 128]
    batchSize = [32, 128]
    optimizers = [torch.optim.Adam, torch.optim.RMSprop, torch.optim.Adagrad, torch.optim.SGD]
    for l in lr:
        for unit in unitsFully:
            for optim in optimizers:
                for b in batchSize:
                    env = gym.make("LunarLander-v2")

                    env.seed(42)
                    torch.manual_seed(42)
                    random.seed(42)
                    np.random.seed(42)

                    model = DQN(env.observation_space.shape[0], env.action_space.n, unit, l,
                                [0.99, 0.999], 0.99, optim)
                    memory = Memory(capacity=1000)
                    maxSteps = 1000

                    res = train(model, env, memory, maxSteps, b)

                    print("Given Learning Rate: {} and hidden units: {} optimiser: {} and batch size: {}, "
                          "we got a reward of {}".format(l, unit, optim, b, sum(res)))
                    rewards.append(res)
                    files.append("lr{}unit{}optim{}batch{}.png".format(str(l), str(unit), str(optim), str(b)))
    plot(rewards, files)

def train(model, env, memory, maxSteps, batch):
    rewards = []
    for episode in range(1000):
        reward = 0.0
        state = env.reset()
        for s in range(maxSteps):
            action = model.policy_net.act(torch.from_numpy(state))

            new_state, rew, done, info = env.step(np.argmax(action))
            reward += rew
            append_sample(model.policy_net, state, np.argmax(action), rew, new_state, done, memory, model.target_net)
            if done:
                rewards.append(reward)
                break

        model.update(memory, batch)
        # if episode % 50 == 0:
        #     print("Episode {} gave a reward of {}".format(episode, reward))

    return rewards


def plot(values, files):
    for a in range(len(values)):
        import matplotlib.pyplot as plt
        rew = []

        for x in range(len(values[a]) // 100):
            tempSum = sum(values[a][x * 100:(x + 1) * 100])
            rew.append(tempSum / 100)

        fig, ax = plt.subplots()
        ax.plot(rew)
        ax.set_xlabel('Episodes/100')
        ax.set_ylabel('Reward')

        # fig.tight_layout()
        plt.savefig(files[a])

hyperparameters()
