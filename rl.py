import torch
from torch import Tensor

from build.np_interop import Policy, ReplayBuffer, MDPTransition, train, transfer_state_dict
import gymnasium as gym

# pol = Policy()
# replay_buf = ReplayBuffer()
#
#
# pol.train()
#
# # Initialise the environment
# # env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("LunarLander-v3", render_mode="human")
#
# # Reset the environment to generate the first observation
# observation, info = env.reset(seed=42)
# num_sims: int = 100
# total_returns: list = []
#
# observation, info = env.reset(seed=42)
# for i in range(num_sims):
#     print(f"Iteration: {i}")
#     this_return: torch.Tensor = Tensor([0.0])
#
#     iteration: int = 0
#     while True:
#         # this is where you would insert your policy
#         # action = env.action_space.sample()
#         # action = pol.forward(torch.Tensor(observation)[None, :]).data
#         old_obs = observation
#         action = torch.argmax(pol.forward(torch.Tensor(observation)[None, :])).item()
#
#         # step (transition) through the environment with the action
#         # receiving the next observation, reward and if the episode has terminated or truncated
#         observation, reward, terminated, truncated, info = env.step(action)
#         cur_transition = MDPTransition(old_obs, action, reward, observation)
#         # print(cur_transition.get_state(), cur_transition.get_action(), cur_transition.get_reward())
#         replay_buf.append(cur_transition)
#         this_return += torch.Tensor([reward])
#
#         # If the episode has ended then we can reset to start a new episode
#         if terminated or truncated:
#             total_returns.append(this_return)
#             env.reset()
#             break
# env.close()
#
# print(f"Total returns: {total_returns}")
# # rbp_elem_tp = [(e.get_state(), e.get_action(), e.get_reward()) for e in replay_buf.as_list()]
# rbp_elem_tp = [e.get() for e in replay_buf.as_list()]
# print(rbp_elem_tp)
#


def collect_data(policy: Policy, replay_buffer: ReplayBuffer, env: gym.core.Env):

    observation, _ = env.reset(seed=42)
    for _ in range(1000):
        cur_obs = observation
        action = torch.argmax(policy.forward(torch.from_numpy(cur_obs)))
        observation, reward, terminated, truncated, _ = env.step(action.item())
        cur_trans: MDPTransition = MDPTransition(cur_obs, int(action), float(reward), observation)

        # MDPTransition(Eigen::VectorXd s, int a, double r, Eigen::VectorXd s_prime) {
        # replay_buffer.add((cur_obs.astype(float), action, reward, observation.astype(float)))
        replay_buffer.append(cur_trans)
        if terminated or truncated:
            observation, _ = env.reset()
    return None


def meta_train_loop(env: gym.core.Env, policy: Policy, critic: Policy, replay_buffer: ReplayBuffer):

    UPDATE_CRITIC: int = 3  # C in paper
    NUM_EPOCHS: int = 5
    COLLECT_DATA: int = 3
    CLEAR_REPLAY_BUF: int = 5

    for i in range(100):
        print(f"Meta loop: {i}")
        # meta loop
        # collect data using policy
        if i % COLLECT_DATA == 0:
            collect_data(policy, replay_buffer, env)

        # create dataset and dataloader
        # train 3 epochs using data
        train(replay_buffer, policy, critic, NUM_EPOCHS, 4)
        # update critic
        if i % UPDATE_CRITIC == 0:
            transfer_state_dict(policy, critic)
            # critic.load_state_dict(policy.state_dict())

        if i % CLEAR_REPLAY_BUF == 0:
            replay_buffer.clear()

    env.close()
    return None


def main():

    policy = Policy()
    critic = Policy()
    replay_buffer = ReplayBuffer()
    env = gym.make("LunarLander-v3", render_mode="human")

    meta_train_loop(env, policy, critic, replay_buffer)

    # observation, info = env.reset(seed=42)
    # for _ in range(10):
    #     cur_obs = observation
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #
    #     replay_buffer.add((cur_obs, action, reward, observation))
    #
    #     if terminated or truncated:
    #         observation, info = env.reset()
    # env.close()
    #
    # print(replay_buffer.buffer)
    return None


if __name__ == "__main__":
    main()
