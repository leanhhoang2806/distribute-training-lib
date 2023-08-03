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
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        B = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)

        result = matrix_multiplication_cuda(A, B)

        expected_result = np.matmul(A, B)
        np.testing.assert_array_equal(result, expected_result)
