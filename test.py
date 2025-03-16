import unittest
import torch
import numpy as np
from build import np_interop
from build.np_interop import MDPTransition, Policy, ReplayBuffer, train, transfer_state_dict


class TestCppModule(unittest.TestCase):

    def test_train_bs1(self):
        print("test_train_bs1")
        state = np.random.rand(8)
        next_state = np.random.rand(8)
        action = 2
        reward = 1.0

        trans = MDPTransition(state, action, reward, next_state)
        rb = ReplayBuffer()
        rb.append(trans)
        policy = Policy()
        critic = Policy()
        train(rb, policy, critic, 3, 8)
        self.assertTrue(True)

    def test_train_4(self):
        print("test_train_4")
        states = [np.random.rand(8) for _ in range(4)]
        next_states = [np.random.rand(8) for _ in range(4)]
        action = 2
        reward = 1.0
        rb = ReplayBuffer()
        for _ in range(4):
            rb.append(MDPTransition(states[_], action, reward, next_states[_]))
        policy = Policy()
        critic = Policy()

        train(rb, policy, critic, 3, 2)
        self.assertTrue(True)

    def test_train_8(self):
        print("test_train_8")
        states = [np.random.rand(8) for _ in range(8)]
        next_states = [np.random.rand(8) for _ in range(8)]
        action = 2
        reward = 1.0
        rb = ReplayBuffer()
        for _ in range(8):
            rb.append(MDPTransition(states[_], action, reward, next_states[_]))
        policy = Policy()
        critic = Policy()

        train(rb, policy, critic, 3, 4)
        self.assertTrue(True)

    def test_transfer_state_dict(self):
        print("Transfer state dict")
        policy = Policy()
        critic = Policy()
        transfer_state_dict(policy, critic)


if __name__ == '__main__':
    unittest.main()
