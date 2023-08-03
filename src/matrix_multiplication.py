
import numpy as np
from src.gpu_availability import count_gpus
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os



def matrix_multiplication_cuda(A, B):
    if count_gpus() == 0:
        raise EnvironmentError("No GPUs found on your computer.")
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions are not compatible for multiplication.")

    # Allocate GPU memory
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc((rows_A * cols_B * np.dtype(np.float32).itemsize))

    # Transfer data to GPU
    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)

    # Define block and grid sizes
    block_size = (16, 16, 1)
    grid_size = ((cols_B - 1) // block_size[0] + 1, (rows_A - 1) // block_size[1] + 1)
    print("Block size is " + str(block_size))
    print("Grid size is " + str(grid_size))

    # Compile CUDA kernel
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(current_dir, "matrix_mult.cu")
    with open(cuda_file, 'r') as f:
        kernel_code = f.read()
    mod = SourceModule(kernel_code)
    matrix_multiply = mod.get_function("matrix_multiply")

    # Call the CUDA kernel
    matrix_multiply(A_gpu, B_gpu, C_gpu, np.int32(rows_A), np.int32(cols_A), np.int32(cols_B), block=block_size, grid=grid_size)

    # Allocate memory for the result on the host and transfer the result from the GPU
    C = np.empty((rows_A, cols_B), dtype=np.float32)
    cuda.memcpy_dtoh(C, C_gpu)

    return C


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

