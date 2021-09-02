import os
import sys
import tempfile
import torch
import time
import numpy as np
import random
import gc
from itertools import cycle, count
import glob
# DDPG agent class
# There is a task here!

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')

class DDPG():
    def __init__(self,
                 replay_buffer_fn,
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
                 tau):
        """
        Class initialization

        replay_buffer_fn = replay buffer function

        policy_model_fn = policy neural network architecture
        policy_max_grad_norm = maximum gradient norm for policy model
        policy_optimizer_fn = optimizer for policy neural network
        policy_optimizer_lr = learning rate for policy neural network

        value_model_fn = value function neural network architecture
        value_max_grad_norm = maximum gradient norm for Q-value model
        value_optimizer_fn = optimizer for value function neural network
        value_optimizer_lr = learning rate for value function neural network

        training_strategy_fn = exploration strategy - Normal Noise Strategy
        evaluation_strategy_fn = evaluation strategy - Greedy Strategy
        n_warmup_batches = warm up batches for training
        update_target_every_steps = updating rate
        tau = Polyak averaging factor
        """
        self.replay_buffer_fn = replay_buffer_fn

        self.policy_model_fn = policy_model_fn
        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

        self.value_model_fn = value_model_fn
        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr

        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn

        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

    def optimize_model(self, experiences):
        """
        Optimize and update parameters in neural network models (Q value and policy models)

        experiences= experience buffer replay - Database

        argmax_a_q_sp = greedy policy for next state
        max_a_q_sp = max Q value function for next state
        target_q_sa = target Q value function
        q_sa = current Q value function
        td_error = TD error
        value_loss = loss function for Q value function neural network model

        argmax_a_q_s = greedy action for current state
        max_a_q_s = Q value from greedy action in current state
        policy_loss = loss function for policy neural network model
        """

        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        # Compute greedy policy for next state
        argmax_a_q_sp = self.target_policy_model(next_states)

        # Compute max Q value function for next state
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)

        # Update target Q value function
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)

        # Compute current Q value function
        q_sa = self.online_value_model(states, actions)

        # TODO: compute TD error with target_q_sa and q_sa
        td_error =  target_q_sa.detach()# To complete ---> When using the target_q_sa variable, you may want to work with target_q_sa.detach()

        # TODO: compute the value loss function to train the neural network (check out slide 35 from lecture 13)
        value_loss =  q_sa.detach()# To complete ---> Recall that we are dealing with td error samples, don't forget to compute the expectation via Monte Carlo

        # Backpropagation for value function neural network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer.step()

        # Compute greedy action with policy model for current state
        argmax_a_q_s = self.online_policy_model(states)

        # Get Q value from greedy action and current state
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)

        # Compute loss for policy model
        policy_loss = -max_a_q_s.mean()

        # Backpropagation for policy neural network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(),
                                       self.policy_max_grad_norm)
        self.policy_optimizer.step()

    def interaction_step(self, state, env):
        """
        agent interacts with the environment by applying action

        min_samples = database from the buffer replay - Train neural networks
        action = action from the policy model
        new_state = new state in the environment
        reward = reward from the current interaction
        is_failure = flag for terminal state
        experience = new sample tuple (state, action, reward, new_state, is_failure) for buffer replay
        """
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
        action = self.training_strategy.select_action(self.online_policy_model,
                                                      state,
                                                      len(self.replay_buffer) < min_samples)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += self.training_strategy.ratio_noise_injected
        return new_state, is_terminal

    def update_networks(self, tau=None):
        """
        Update neural network models parameters (policy and Q-value function) via Polyak Averaging
        * We use this technique to avoid aggressive model parameters updates

        tau = Polyak averaging factor
        target_ratio = averaging ratio
        mixed_weights = new mixed weights
        """
        tau = self.tau if tau is None else tau

        for target, online in zip(self.target_value_model.parameters(),
                                  self.online_value_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_policy_model.parameters(),
                                  self.online_policy_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def train(self, make_env_fn, make_env_kargs, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):
        """
        Training function and computing stats

        make_env_fn = make environment function
        make_env_kargs = arguments for make environment function
        seed = seed for random numbers
        gamma = discount factor
        max_minutes = maximum training time
        max_episodes = maximum training episodes
        goal_mean_100_reward = target reward goal
        """

        # Setup environment
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Initialize actor-critic agent
        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        # Setup target and online Q-value functions
        self.target_value_model = self.value_model_fn(nS, nA)
        self.online_value_model = self.value_model_fn(nS, nA)

        # Setup target and online policy model
        self.target_policy_model = self.policy_model_fn(nS, action_bounds)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds)

        # Setup optimize and update parameters functions
        self.update_networks(tau=1.0)
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model,
                                                       self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model,
                                                         self.policy_optimizer_lr)

        # Setup replay buffer, and training/evaluation strategies
        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn(action_bounds)
        self.evaluation_strategy = self.evaluation_strategy_fn(action_bounds)

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0

        # Episodic interaction agent-environment
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_value_model.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_networks()

                if is_terminal:
                    gc.collect()
                    break

            # save stats
            # ---DO NOT TOUCH---
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_policy_model, env)
            self.save_checkpoint(episode - 1, self.online_policy_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = total_step, mean_100_reward, \
                                  mean_100_eval_score, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:07}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode - 1, total_step, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break

        # End training and save results
        final_eval_score, score_std = self.evaluate(self.online_policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
            final_eval_score, score_std, training_time, wallclock_time))
        env.close();
        del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        """
        evaluate trained policy

        eval_policy_model = policy model to evaluate
        eval_env = environment to evaluate
        n_episodes = number of episodes to evaluate
        a = action
        s = current state
        r = reward
        d = next state
        """
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=4):
        """
        clean database for saving
        """
        try:
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]): path for path in paths}
        last_ep = max(paths_dic.keys())
        checkpoint_idxs = np.linspace(1, last_ep + 1, n_checkpoints, endpoint=True, dtype=np.int) - 1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def save_checkpoint(self, episode_idx, model):
        """
        Save model
        """
        torch.save(model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))