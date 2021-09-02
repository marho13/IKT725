# Libraries and setup
# ---DO NOT TOUCH---
import warnings ; warnings.filterwarnings('ignore')
import os

import TD3.TD3 as TD3
import DDPG.DDPG as DDPG
import FCQV.FCQV as FCQV
import FCDP.FCDP as FCDP
import FCTQV.FCTQV as FCTQV

import NormalNoiseStrategy.NormalNoiseStrategy as NormalNoiseStrategy
import GreedyStrategy.GreedyStrategy as GreedyStrategy
import ReplayBuffer.ReplayBuffer as ReplayBuffer
import NormalNoiseDecayStrategy.NormalNoiseDecayStrategy as NormalNoiseDecayStrategy

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'

import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import os.path
import tempfile
import gym
import os

from gym import wrappers

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)
# Setup plotting parameters
# ---DO NOT TOUCH---
plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
np.set_printoptions(suppress=True)

torch.cuda.is_available()

# Get gym environment function
# ---DO NOT TOUCH---
def get_make_env_fn(**kargs):
    def make_env_fn(env_name, seed=None, render=None, record=False,
                    unwrapped=False, monitor_mode=None,
                    inner_wrappers=None, outer_wrappers=None):
        mdir = tempfile.mkdtemp()
        env = None
        if render:
            try:
                env = gym.make(env_name, render=render)
            except:
                pass
        if env is None:
            env = gym.make(env_name)
        if seed is not None: env.seed(seed)
        env = env.unwrapped if unwrapped else env
        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)
        env = wrappers.Monitor(
            env, mdir, force=True,
            mode=monitor_mode,
            video_callable=lambda e_idx: record) if monitor_mode else env
        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)
        return env
    return make_env_fn, kargs


# DDPG training/evaluation routine
# There is a task here

ddpg_results = []
best_agent, best_eval_score = None, float('-inf')

# Prepare environment
for seed in SEEDS:
    environment_settings = {
        'env_name': 'Pendulum-v0',
        'gamma': 0.99,
        'max_minutes': 5,
        'max_episodes': 500,
        'goal_mean_100_reward': -150
    }

    # Setup DDPG agent
    # ---Policy neural network---
    policy_model_fn = lambda nS, bounds: FCDP(nS, bounds)
    policy_max_grad_norm = float('inf')

    # TODO: select an optimization algorithm for the policy neural network. For further information: https://pytorch.org/docs/stable/optim.html - Algorithms
    policy_optimizer_fn = lambda net, lr: policy_optimizer_lr # To complete -> Follow the format optim.Optimization_Algorithm(net.parameters(), lr=lr)

    # TODO: select a suitable learning rate for the optimization algorithm
    policy_optimizer_lr = 0.001 # To complete

    # ---Value function neural network---
    value_model_fn = lambda nS, nA: FCQV(nS, nA)
    value_max_grad_norm = float('inf')

    # TODO: select an optimization algorithm for the value neural network. For further information: https://pytorch.org/docs/stable/optim.html - Algorithms
    value_optimizer_fn = lambda net, lr: value_optimizer_lr # To complete -> Follow the format optim.Optimization_Algorithm(net.parameters(), lr=lr)

    # TODO: select a suitable learning rate for the optimization algorithm
    value_optimizer_lr = 0.01 # To complete

    # Training/evaluation strategy
    training_strategy_fn = lambda bounds: NormalNoiseStrategy(bounds, exploration_noise_ratio=0.1)
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)

    replay_buffer_fn = lambda: ReplayBuffer(max_size=100000, batch_size=256)
    n_warmup_batches = 5
    update_target_every_steps = 1
    tau = 0.005

    env_name, gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()

    # Update agent
    agent = DDPG(replay_buffer_fn,
                 policy_model_fn,
                 policy_max_grad_norm,
                 policy_optimizer_fn,
                 policy_optimizer_lr,
                 value_model_fn,
                 value_max_grad_norm,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau)

    # Train/evaluate agent
    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)

    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)

    # Save results
    ddpg_results.append(result)

    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent

ddpg_results = np.array(ddpg_results)
_ = BEEP()


# Save results from DDPG agent
# ---DO NOT TOUCH---
ddpg_max_t, ddpg_max_r, ddpg_max_s, \
ddpg_max_sec, ddpg_max_rt = np.max(ddpg_results, axis=0).T
ddpg_min_t, ddpg_min_r, ddpg_min_s, \
ddpg_min_sec, ddpg_min_rt = np.min(ddpg_results, axis=0).T
ddpg_mean_t, ddpg_mean_r, ddpg_mean_s, \
ddpg_mean_sec, ddpg_mean_rt = np.mean(ddpg_results, axis=0).T
ddpg_x = np.arange(len(ddpg_mean_s))


# Plot results
# ---DO NOT TOUCH---
fig, axs = plt.subplots(2, 1, figsize=(15,10), sharey=False, sharex=True)


# DDPG
axs[0].plot(ddpg_max_r, 'r', linewidth=1)
axs[0].plot(ddpg_min_r, 'r', linewidth=1)
axs[0].plot(ddpg_mean_r, 'r:', label='DDPG', linewidth=2)
axs[0].fill_between(
    ddpg_x, ddpg_min_r, ddpg_max_r, facecolor='r', alpha=0.3)

axs[1].plot(ddpg_max_s, 'r', linewidth=1)
axs[1].plot(ddpg_min_s, 'r', linewidth=1)
axs[1].plot(ddpg_mean_s, 'r:', label='DDPG', linewidth=2)
axs[1].fill_between(
    ddpg_x, ddpg_min_s, ddpg_max_s, facecolor='r', alpha=0.3)


# ALL
axs[0].set_title('Moving Avg Reward (Training)')
axs[1].set_title('Moving Avg Reward (Evaluation)')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.show()


# Plot results
# ---DO NOT TOUCH---
fig, axs = plt.subplots(3, 1, figsize=(15,15), sharey=False, sharex=True)


# DDPG
axs[0].plot(ddpg_max_t, 'r', linewidth=1)
axs[0].plot(ddpg_min_t, 'r', linewidth=1)
axs[0].plot(ddpg_mean_t, 'r:', label='DDPG', linewidth=2)
axs[0].fill_between(
    ddpg_x, ddpg_min_t, ddpg_max_t, facecolor='r', alpha=0.3)

axs[1].plot(ddpg_max_sec, 'r', linewidth=1)
axs[1].plot(ddpg_min_sec, 'r', linewidth=1)
axs[1].plot(ddpg_mean_sec, 'r:', label='DDPG', linewidth=2)
axs[1].fill_between(
    ddpg_x, ddpg_min_sec, ddpg_max_sec, facecolor='r', alpha=0.3)

axs[2].plot(ddpg_max_rt, 'r', linewidth=1)
axs[2].plot(ddpg_min_rt, 'r', linewidth=1)
axs[2].plot(ddpg_mean_rt, 'r:', label='DDPG', linewidth=2)
axs[2].fill_between(
    ddpg_x, ddpg_min_rt, ddpg_max_rt, facecolor='r', alpha=0.3)


# ALL
axs[0].set_title('Total Steps')
axs[1].set_title('Training Time')
axs[2].set_title('Wall-clock Time')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.show()


# Plot noise decay function
# --- DO NOT TOUCH ---
s = NormalNoiseDecayStrategy(([-2],[2]))
plt.plot([s._noise_ratio_update() for _ in range(50000)])
plt.title('Normal Noise Linear ratio')
plt.xticks(rotation=45)
plt.show()

# TD3 training/evaluation routine
# There is a task here
td3_results = []
best_agent, best_eval_score = None, float('-inf')

# Prepare environment
for seed in SEEDS:
    environment_settings = {
        'env_name': 'Pendulum-v0',
        'gamma': 0.99,
        'max_minutes': 5,
        'max_episodes': 500,
        'goal_mean_100_reward': -150
    }

    # Setup TD3 agent
    # ---Policy neural network---
    policy_model_fn = lambda nS, bounds: FCDP(nS, bounds)
    policy_max_grad_norm = float('inf')

    # TODO: select an optimization algorithm for the policy neural network. For further information: https://pytorch.org/docs/stable/optim.html - Algorithms
    policy_optimizer_fn = lambda net, lr: policy_optimizer_lr # To complete -> Follow the format optim.Optimization_Algorithm(net.parameters(), lr=lr)

    # TODO: select a suitable learning rate for the optimization algorithm
    policy_optimizer_lr = 0.001 # To complete

    # ---Value function neural network---
    value_model_fn = lambda nS, nA: FCTQV(nS, nA)
    value_max_grad_norm = float('inf')

    # TODO: select an optimization algorithm for the value neural network. For further information: https://pytorch.org/docs/stable/optim.html - Algorithms
    value_optimizer_fn = lambda net, lr: value_optimizer_lr # To complete -> Follow the format optim.Optimization_Algorithm(net.parameters(), lr=lr)

    # TODO: select a suitable learning rate for the optimization algorithm
    value_optimizer_lr = 0.01 # To complete

    # Training/evaluation strategy
    training_strategy_fn = lambda bounds: NormalNoiseDecayStrategy(bounds,
                                                                   init_noise_ratio=0.5,
                                                                   min_noise_ratio=0.1,
                                                                   decay_steps=200000)
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)

    replay_buffer_fn = lambda: ReplayBuffer(max_size=1000000, batch_size=256)
    n_warmup_batches = 5
    update_value_target_every_steps = 2
    update_policy_target_every_steps = 2
    train_policy_every_steps = 2
    policy_noise_ratio = 0.1
    policy_noise_clip_ratio = 0.5
    tau = 0.005

    env_name, gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()

    # Update agent
    agent = TD3(replay_buffer_fn,
                policy_model_fn,
                policy_max_grad_norm,
                policy_optimizer_fn,
                policy_optimizer_lr,
                value_model_fn,
                value_max_grad_norm,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                evaluation_strategy_fn,
                n_warmup_batches,
                update_value_target_every_steps,
                update_policy_target_every_steps,
                train_policy_every_steps,
                tau,
                policy_noise_ratio,
                policy_noise_clip_ratio)
    # Train/evaluate agent
    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)

    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)

    # Save results
    td3_results.append(result)

    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent

td3_results = np.array(td3_results)
_ = BEEP()


# Save results from TD3 agent
# ---DO NOT TOUCH---
td3_max_t, td3_max_r, td3_max_s, \
td3_max_sec, td3_max_rt = np.max(td3_results, axis=0).T
td3_min_t, td3_min_r, td3_min_s, \
td3_min_sec, td3_min_rt = np.min(td3_results, axis=0).T
td3_mean_t, td3_mean_r, td3_mean_s, \
td3_mean_sec, td3_mean_rt = np.mean(td3_results, axis=0).T
td3_x = np.arange(len(td3_mean_s))


# Plot results
# ---DO NOT TOUCH---
fig, axs = plt.subplots(2, 1, figsize=(15,10), sharey=False, sharex=True)

# TD3
axs[0].plot(td3_max_r, 'b', linewidth=1)
axs[0].plot(td3_min_r, 'b', linewidth=1)
axs[0].plot(td3_mean_r, 'b:', label='TD3', linewidth=2)
axs[0].fill_between(
    td3_x, td3_min_r, td3_max_r, facecolor='b', alpha=0.3)

axs[1].plot(td3_max_s, 'b', linewidth=1)
axs[1].plot(td3_min_s, 'b', linewidth=1)
axs[1].plot(td3_mean_s, 'b:', label='TD3', linewidth=2)
axs[1].fill_between(
    td3_x, td3_min_s, td3_max_s, facecolor='b', alpha=0.3)

# ALL
axs[0].set_title('Moving Avg Reward (Training)')
axs[1].set_title('Moving Avg Reward (Evaluation)')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.show()


# Plot results
# ---DO NOT TOUCH---
fig, axs = plt.subplots(3, 1, figsize=(15,15), sharey=False, sharex=True)

# TD3
axs[0].plot(td3_max_t, 'b', linewidth=1)
axs[0].plot(td3_min_t, 'b', linewidth=1)
axs[0].plot(td3_mean_t, 'b:', label='TD3', linewidth=2)
axs[0].fill_between(
    td3_x, td3_min_t, td3_max_t, facecolor='b', alpha=0.3)

axs[1].plot(td3_max_sec, 'b', linewidth=1)
axs[1].plot(td3_min_sec, 'b', linewidth=1)
axs[1].plot(td3_mean_sec, 'b:', label='TD3', linewidth=2)
axs[1].fill_between(
    td3_x, td3_min_sec, td3_max_sec, facecolor='b', alpha=0.3)

axs[2].plot(td3_max_rt, 'b', linewidth=1)
axs[2].plot(td3_min_rt, 'b', linewidth=1)
axs[2].plot(td3_mean_rt, 'b:', label='TD3', linewidth=2)
axs[2].fill_between(
    td3_x, td3_min_rt, td3_max_rt, facecolor='b', alpha=0.3)

# ALL
axs[0].set_title('Total Steps')
axs[1].set_title('Training Time')
axs[2].set_title('Wall-clock Time')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.show()