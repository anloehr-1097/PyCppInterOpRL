import unittest
import torch
import numpy as np
import np_interop
from np_interop import MDPTransition, Policy, ReplayBuffer, train


class TestCppModule(unittest.TestCase):

    def test_train_bs1(self):
        state = np.random.rand(5)
        next_state = np.random.rand(5)
        action = 2
        reward = 1.0

        trans = MDPTransition(state, action, reward, next_state)
        rb = ReplayBuffer()
        rb.append(trans)
        policy = Policy()
        train(rb, policy, 3)
        self.assertTrue(True)

    def test_train_4(self):
        state = np.random.rand(5)
        next_state = np.random.rand(5)
        action = 2
        reward = 1.0

        trans = MDPTransition(state, action, reward, next_state)
        rb = ReplayBuffer()
        for _ in range(4):
            rb.append(trans)
        policy = Policy()
        train(rb, policy, 3)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
