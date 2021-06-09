### MDP Value Iteration and Policy Iteration
import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, value, gamma=0.9, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	# -1 is reward -2 is nextstate, 0 or -3 is probability
	while True:
		newValueFunction = np.copy(value)
		for s in range((nS)):
			for a in range((nA)):
				prob, nextState, r, terminal = P[s][a][0]
				newValueFunction[s] = r + gamma * prob * value[nextState]
		valueChange = np.sum(np.abs(value - newValueFunction))
		value = newValueFunction
		if valueChange < tol:
			break

	return value


def policy_improvement(P, nS, nA, valuefromPolicy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

	# new_policy = np.zeros(nS, dtype='int')
	newPolicy = np.copy(policy)  # Important

	############################
	# YOUR IMPLEMENTATION HERE #
	for s in range(nS):
		actionReward = []
		for a in range(nA):
			prob, nextState, r, terminal = P[s][a][0]
			actionReward.append(r + gamma * prob * valuefromPolicy[nextState])
		newPolicy[s] = np.argmax(actionReward)
	############################

	return newPolicy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""
	# global value_function, policy

	value = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	timestep = 0

	############################
	# YOUR IMPLEMENTATION HERE #
	while True:
		timestep += 1
		# Policy evaluation
		value = policy_evaluation(P, nS, nA, value, gamma, tol)
		# Policy improvement
		newPolicy = policy_improvement(P, nS, nA, value, policy, gamma)
		policyChange = (newPolicy != policy).sum()
		print('policy changed in %d states' % (policyChange))
		policy = newPolicy
		if policyChange == 0:
			break
	############################
	print("It took {} timesteps".format(timestep))
	return value, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""
	value = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	timestep = 0
	while True:
		timestep += 1
		newValueFunction = np.copy(value)
		for s in range(nS):
			rewards = []
			for a in range(nA):
				prob, nextState, r, terminal = P[s][a][0]
				rewards.append(r + gamma * prob * newValueFunction[nextState])
			newValueFunction[s] = np.max(rewards)
		value_change = np.sum(np.abs(value - newValueFunction))
		value = newValueFunction
		if value_change < tol:
			break

	# Get best policy
	for s in range(nS):
		rewards = []
		for a in range(nA):
			prob, nextState, r, terminal = P[s][a][0]
			rewards.append(r + gamma * prob * newValueFunction[nextState])
		policy[s] = np.argmax(rewards)
	############################
	print("It took {} timesteps".format(timestep))
	return value, policy

def render_single(env, policy, max_steps=100):
	"""
	This function does not need to be modified
	Renders policy once on environment. Watch your agent play!

	Parameters
	----------
	env: gym.core.Environment
	  Environment to play on. Must have nS, nA, and P as
	  attributes.
	Policy: np.array of shape [env.nS]
	  The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(max_steps):
		env.render()
		time.sleep(0.25)
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
		env.render()
		if not done:
			print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
		else:
			print("Episode reward: %f" % episode_reward)


def runStochastic():
	# comment/uncomment these lines to switch between deterministic/stochastic environments
	# env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	# print(env.P)
	print("\n" + "-" * 25 + "\nBeginning Stochastic Policy Iteration\n" + "-" * 25)
	# value_function = np.zeros(env.nS)
	# policy = np.zeros(env.nS, dtype=int)
	# R = np.zeros((env.nS, env.nA), dtype=float)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	# render_single(env, p_pi, 100)
	print("\n Optimal State_Values: \n", V_pi.reshape(4, 4))
	print("\n Optimal Policy: \n", p_pi.reshape(4, 4))

	print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	# render_single(env, p_vi, 100)
	print("\n Optimal State_Values: \n", V_vi.reshape(4, 4))
	print("\n Optimal Policy: \n", p_vi.reshape(4, 4))

def runDeterministic():
	# comment/uncomment these lines to switch between deterministic/stochastic environments
	# env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	# print(env.P)
	print("\n" + "-" * 25 + "\nBeginning Deterministic Policy Iteration\n" + "-" * 25)
	# value_function = np.zeros(env.nS)
	# policy = np.zeros(env.nS, dtype=int)
	# R = np.zeros((env.nS, env.nA), dtype=float)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	# render_single(env, p_pi, 100)
	print("\n Optimal State_Values: \n", V_pi.reshape(4, 4))
	print("\n Optimal Policy: \n", p_pi.reshape(4, 4))

	print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	# render_single(env, p_vi, 100)
	print("\n Optimal State_Values: \n", V_vi.reshape(4, 4))
	print("\n Optimal Policy: \n", p_vi.reshape(4, 4))

# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
	runStochastic()
	runDeterministic()


