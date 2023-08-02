
from numba import cuda
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


@cuda.jit
def matrix_multiplication_kernel(matrix1, matrix2, result, rows1, cols1, cols2):
    i, j = cuda.grid(2)
    if i < rows1 and j < cols2:
        tmp = 0.0
        for k in range(cols1):
            tmp += matrix1[i, k] * matrix2[k, j]
        result[i, j] = tmp

def matrix_multiplication_cuda(matrix1, matrix2):
    if count_gpus() == 0:
        raise Exception("No GPUs found on this computer.")
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    if cols1 != rows2:
        raise ValueError("Number of columns in matrix1 must be equal to the number of rows in matrix2")

    # Move matrices to GPU memory
    matrix1_gpu = cuda.to_device(np.array(matrix1, dtype=np.float32))
    matrix2_gpu = cuda.to_device(np.array(matrix2, dtype=np.float32))

    # Allocate memory for the result on GPU
    result_gpu = cuda.device_array((rows1, cols2), dtype=np.float32)

    # Define block and grid sizes for CUDA kernel
    block_size = (32, 32)
    grid_size = ((cols2 - 1) // block_size[0] + 1, (rows1 - 1) // block_size[1] + 1)

    # Launch the CUDA kernel
    matrix_multiplication_kernel[grid_size, block_size](matrix1_gpu, matrix2_gpu, result_gpu, rows1, cols1, cols2)

    # Move the result back to CPU memory
    result_cpu = result_gpu.copy_to_host()

    return result_cpu.tolist()  # Convert NumPy array back to a Python list

