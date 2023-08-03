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
        A = np.array([1, 2, 3, 4], dtype=np.float32)
        B = np.array([5, 6, 7, 8], dtype=np.float32)

        A = A.reshape(1, -1)  # Reshape A to a row vector (1 x 4)
        B = B.reshape(-1, 1)  # Reshape B to a column vector (4 x 1)

        result = matrix_multiplication_cuda(A, B)
        print(result)

        assert np.array_equal(result, np.array([[70]])), f"Test failed: Expected {np.array([[70]])}, Got {result}"

        print("Test passed.")
