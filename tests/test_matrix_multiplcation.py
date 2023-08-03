import unittest
import numpy as np
import tracemalloc
from src.matrix_multiplication import matrix_multiplication_cuda

class TestMatrixMultiplication(unittest.TestCase):
    def setUp(self):
        # Enable tracemalloc before each test
        tracemalloc.start()

    def tearDown(self):
        # Stop tracemalloc after each test
        tracemalloc.stop()
    def test_matrix_multiplication(self):
        A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        B = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        expected_result = np.array([4.0, 10.0, 18.0], dtype=np.float32)

        result = matrix_multiplication_cuda(A, B)

        assert np.array_equal(result, expected_result), f"Test failed: Expected {expected_result}, Got {result}"

        print("Test passed.")
