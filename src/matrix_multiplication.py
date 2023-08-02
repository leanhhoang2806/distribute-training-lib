import cupy as cp
import numpy as np
from src.gpu_availability import count_gpus


def matrix_multiplication(matrix1, matrix2):
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])

    if cols1 != rows2:
        raise ValueError("Number of columns in matrix1 must be equal to the number of rows in matrix2")

    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

def matrix_multiplication_cuda(matrix1, matrix2):
    if count_gpus() == 0:
        raise Exception("No GPUs found on this computer.")
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    if cols1 != rows2:
        raise ValueError("Number of columns in matrix1 must be equal to the number of rows in matrix2")

    # Move matrices to GPU memory
    matrix1_gpu = cp.asarray(matrix1, dtype=np.float32)
    matrix2_gpu = cp.asarray(matrix2, dtype=np.float32)

    # Allocate memory for the result on GPU
    result_gpu = cp.empty((rows1, cols2), dtype=np.float32)

    # Define block and grid sizes for CUDA kernel
    block_size = (32, 32)
    grid_size = ((cols2 - 1) // block_size[0] + 1, (rows1 - 1) // block_size[1] + 1)

    # Load and compile the CUDA kernel
    with open('matrix_mult.cu', 'r') as f:
        kernel_code = f.read()

    matrix_mult_kernel = cp.RawKernel(kernel_code, 'matrix_multiplication')

    # Launch the CUDA kernel
    matrix_mult_kernel(grid_size, block_size, (matrix1_gpu, matrix2_gpu, result_gpu, rows1, cols1, cols2))

    # Move the result back to CPU memory
    result_cpu = cp.asnumpy(result_gpu)

    return result_cpu
