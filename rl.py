import torch
import numpy as np
from torch import Tensor
from build import np_interop as cpp
import gymnasium as gym

pol = cpp.Policy()
replay_buf = cpp.ReplayBuffer()

pol.train()

# Initialise the environment
# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
num_sims: int = 10
total_returns: list = []

observation, info = env.reset(seed=42)
for i in range(num_sims):
    this_return: torch.Tensor = Tensor([0.0])

    iteration: int = 0
    while True:
        # this is where you would insert your policy
        # action = env.action_space.sample()
        # action = pol.forward(torch.Tensor(observation)[None, :]).data
        old_obs = observation
        action = torch.argmax(pol.forward(torch.Tensor(observation)[None, :])).item()


        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        replay_buf.append(cpp.MDPTransition(old_obs, action, reward))
        this_return += torch.Tensor([reward])

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            total_returns.append(this_return)
            env.reset()
            break
env.close()

print(f"Total returns: {total_returns}")
