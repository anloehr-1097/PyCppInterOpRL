import unittest
from test import TestCppModule



def run():
    suite = unittest.TestSuite()
    suite.addTest(TestCppModule("test_train_4"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    return None

if __name__ == "__main__":
    unittest.main(test)
