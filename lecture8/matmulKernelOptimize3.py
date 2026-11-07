from numba import cuda, float32
import numpy as np
import cupy as cp


ceil_div = lambda a, b: (a+b-1)//b
tile_dim = 32

BM = 32
BN = 32
BK = 32
TM = 8               # no. of output elements processeed by a single thread

NT = (BM * BN) // TM # no. of threads required within a block 

@cuda.jit
def matmulKernel(A, B, C, m, n, k):
    tileA = cuda.shared.array((BM, BK), dtype=float32)
    tileB = cuda.shared.array((BK, BN), dtype=float32)

    tid = cuda.threadIdx.x
    col = tid % BN
    row = (tid // BN) * TM

    block_row = cuda.blockIdx.y * BM
    block_col = cuda.blockIdx.x * BN

    # register allocation and initialization of the accumulator
    acc = cuda.local.array(TM, dtype=float32)
    for t in range(TM):
        acc[t] = float32(0.0)

    for k0 in range(0, k, BK):
        # phase 1:
        # threads within the block
        # will read elements from A and B (gmem)
        # and store them in tileA and tileB (smem)
        
        # phase 1: load values from A into tileA
        for i in range(tid, BM * BK, NT):
            r = i // BK
            c = i % BK

            global_mem_row = block_row + r
            global_mem_col = k0 + c
            if global_mem_row < m and global_mem_col < k:
                tileA[r, c] = A[global_mem_row, global_mem_col]
            else:
                tileA[r, c] = float32(0.0)

        # phase 1: load values from B into tileB
        for i in range(tid, BK * BN, NT):
            r = i // BN
            c = i % BN

            global_mem_row = k0 + r
            global_mem_col = block_col + c
            if global_mem_row < k and global_mem_col < n:
                tileB[r, c] = B[global_mem_row, global_mem_col]
            else:
                tileB[r, c] = float32(0.0)

        cuda.syncthreads()

        # phase2:
        # threads within the block
        # will read elements from tileA and tileB (smem)

        # for kk in range(BK):
        #     b = tileB[kk, col]
        #     for t in range(TM):
        #         product[t] += tileA[row + t, kk] * b

        for i in range(TM):
            a = tileA[row + 0, i]
            b = tileB[i, col]
            acc[0] += a * b

            a = tileA[row + 1, i]
            acc[1] += a * b

            a = tileA[row + 2, i]
            acc[2] += a * b

            a = tileA[row + 3, i]
            acc[3] += a * b

            a = tileA[row + 4, i]
            acc[4] += a * b

            a = tileA[row + 5, i]
            acc[5] += a * b

            a = tileA[row + 6, i]
            acc[6] += a * b

            a = tileA[row + 7, i]
            acc[7] += a * b

        cuda.syncthreads()

        for t in range(TM):
            global_mem_row = block_row + row + t
            global_mem_col = block_col + col
            if global_mem_row < m and global_mem_col < n:
                C[global_mem_row, global_mem_col] = acc[t]





m = 1000
n = 800
k = 1200

A = cp.ones(shape=(m, k), dtype=cp.float32) * 2.0
B = cp.ones(shape=(k, n), dtype=cp.float32) * 3.0
C = cp.zeros(shape=(m, n), dtype=cp.float32)

threads_per_block = (
    NT, 1
)
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