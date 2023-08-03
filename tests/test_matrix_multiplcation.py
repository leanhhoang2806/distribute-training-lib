import unittest
import numpy as np
import tracemalloc
from src.matrix_multiplication import matrix_multiplication_cuda, matrix_multiplication_using_numpy

class TestMatrixMultiplication(unittest.TestCase):
    def setUp(self):
        # Enable tracemalloc before each test
        tracemalloc.start()

    def tearDown(self):
        # Stop tracemalloc after each test
        tracemalloc.stop()
    def test_matrix_multiplication(self):
        # Test case: 2x4 matrix multiplication
        matrix1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        matrix2 = np.array([[8, 7], [6, 5], [4, 3], [2, 1]], dtype=np.float32)

        # Perform matrix multiplication using NumPy
        result_numpy = matrix_multiplication_using_numpy(matrix1, matrix2)

        # Perform matrix multiplication using CUDA
        result_cuda = matrix_multiplication_cuda(matrix1, matrix2)

        # Check if the results from NumPy and CUDA are the same
        assert np.allclose(result_numpy, result_cuda)
        print("Test passed.")
