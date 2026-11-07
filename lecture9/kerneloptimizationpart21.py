from numba import cuda, float32
import numpy as np
import cupy as cp

"""
2D BLOCK TILLING (MORE REGISTERS)
"""

ceil_div = lambda a, b: (a+b-1)//b

#Block tile dimension (BM x BW) 
BM = 64
BN = 64
BK = 16

#Thread tile dimension (TM x TN)
TM = 8
TN = 8

NT = (BM // TM) * (BN // TN)

    
@cuda.jit
def matmulKernel(A, B, C, m, n, k):
    tileA = cuda.shared.array((BM, BK), dtype=float32)
    tileB = cuda.shared.array((BK, BN), dtype=float32)



m = 1024
n = 1024
k = 1024

A = cp.ones(shape=(m, k), dtype=cp.float32) * 2.0
B = cp.ones(shape=(k, n), dtype=cp.float32) * 3.0
C = cp.zeros(shape=(m, n), dtype=cp.float32)

threads_per_block = (NT, 1)
blocks_per_grid = (
    ceil_div(n, BN),
    ceil_div(m, BM)
)

matmulKernel[blocks_per_grid, threads_per_block](A, B, C, m, n, k)
cuda.synchronize()
matA = cp.asnumpy(A)
matB = cp.asnumpy(B)
matC = cp.asnumpy(C)

np.testing.assert_allclose(matC, np.matmul(matA, matB), rtol=1e-5, atol=1e-5)