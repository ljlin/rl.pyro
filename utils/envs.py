import gym
import numpy as np
import random
from copy import deepcopy
import torch

# github.com/google-research/google-research/blob/master/algae_dice/wrappers/normalize_action_wrapper.py
class NormalizeBoxActionWrapper(gym.ActionWrapper):
  """Rescale the action space of the environment."""

  def __init__(self, env):
    if not isinstance(env.action_space, gym.spaces.Box):
      raise ValueError('env %s does not use spaces.Box.' % str(env))
    super(NormalizeBoxActionWrapper, self).__init__(env)
    
  def action(self, action):
    # rescale the action
    low, high = self.env.action_space.low, self.env.action_space.high
    scaled_action = low + (action + 1.0) * (high - low) / 2.0
    scaled_action = np.clip(scaled_action, low, high)

    return scaled_action

  def reverse_action(self, scaled_action):
    low, high = self.env.action_space.low, self.env.action_space.high
    action = (scaled_action - low) * 2.0 / (high - low) - 1.0
    return action

# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render = False):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    if render: env.render()
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, info = env.step(action)
        if render: env.render()
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

def play_episode_tensor(env, policy, t, render = False):
    states, actions, rewards = play_episode(env,policy,render)
    return t.f(states), t.l(actions), t.f(rewards)

# Play an episode according to a given policy and add 
# to a replay buffer
# env: environment
# policy: function(env, state)
def play_episode_rb(env, policy, buf):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, info = env.step(action)
        buf.add(states[-1], action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

def play_episode_rb_with_steps(env, policy, buf):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    step = 0
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, info = env.step(action)
        buf.add(states[-1], action, reward, obs, done, step)
        step += 1
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards