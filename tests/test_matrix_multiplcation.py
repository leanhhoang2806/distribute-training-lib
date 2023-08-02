import unittest
from src.matrix_multiplication import matrix_multiplication, matrix_multiplication_cuda

class TestMatrixMultiplication(unittest.TestCase):
    def test_matrix_multiplication(self):
        matrix1 = [[1, 2], [3, 4]]  # 2x2 matrix
        matrix2 = [[5, 6], [7, 8]]  # 2x2 matrix
        expected_result = [[19, 22], [43, 50]]

        result = matrix_multiplication(matrix1, matrix2)
        cuda_result = matrix_multiplication_cuda(matrix1, matrix2)

        self.assertEqual(result, expected_result)
        self.assertEqual(cuda_result, expected_result)
