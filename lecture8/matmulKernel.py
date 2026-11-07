from numba import cuda, float32
import numpy as np
import cupy as cp

"""
load from matA -> Coalesced access
load from matB -> Strided access

"""
ceil_div = lambda a, b: (a+b-1)//b

@cuda.jit
def matmulKernel(A, B, C, m, n, k):
    out_row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    out_col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    if out_col >= n or out_row >= m:
        return

    dot_product = float32(0.0)
    for i in range(k):
        dot_product += A[out_row, i] * B[i, out_col]
    C[out_row, out_col] = dot_product

m = 1000
n = 800
k = 1200

A = cp.ones(shape=(m, k), dtype=cp.float32) * 2.0
B = cp.ones(shape=(k, n), dtype=cp.float32) * 3.0
C = cp.zeros(shape=(m, n), dtype=cp.float32)

threads_per_block = (32, 32)
blocks_per_grid = (
    ceil_div(n, threads_per_block[0]),
    ceil_div(m, threads_per_block[1])
)

matmulKernel[blocks_per_grid, threads_per_block](A, B, C, m, n, k)
cuda.synchronize()
matA = cp.asnumpy(A)
matB = cp.asnumpy(B)
matC = cp.asnumpy(C)

np.testing.assert_allclose(matC, np.matmul(matA, matB), rtol=1e-5, atol=1e-5)