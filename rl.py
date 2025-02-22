import torch
import numpy as np
from torch import Tensor
from build import np_interop as cpp
from build.np_interop import Policy, ReplayBuffer, MDPTransition
import gymnasium as gym

pol = Policy()
replay_buf = ReplayBuffer()


pol.train()

# Initialise the environment
# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
num_sims: int = 2
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
        cur_transition = MDPTransition(old_obs, action, reward)
        # print(cur_transition.get_state(), cur_transition.get_action(), cur_transition.get_reward())
        replay_buf.append(cur_transition)
        this_return += torch.Tensor([reward])

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            total_returns.append(this_return)
            env.reset()
            break
env.close()

print(f"Total returns: {total_returns}")
# rbp_elem_tp = [(e.get_state(), e.get_action(), e.get_reward()) for e in replay_buf.as_list()]
rbp_elem_tp = [e.get() for e in replay_buf.as_list()]
print(rbp_elem_tp)
