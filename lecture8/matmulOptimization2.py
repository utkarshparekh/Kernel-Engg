from numba import cuda, float32
import numpy as np
import cupy as cp


ceil_div = lambda a, b: (a+b-1)//b
tile_dim = 32

@cuda.jit
def matmulKernel(A, B, C, m, n, k):
    tileA = cuda.shared.array((tile_dim, tile_dim), dtype=float32)
    tileB = cuda.shared.array((tile_dim, tile_dim), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    out_col = tx + cuda.blockDim.x * cuda.blockIdx.x
    out_row = ty + cuda.blockDim.y * cuda.blockIdx.y

    num_tiles = (k + tile_dim - 1) // tile_dim
    product = float32(0.0)

    for t in range(num_tiles):
        # phase 1:
        # threads within the block
        # will read elements from A and B (gmem)
        # and store them in tileA and tileB (smem)
        tile_offset = t * tile_dim
        a_col = tile_offset + tx
        b_row = tile_offset + ty

        # load valuesinto shared memory from global memory
        if out_row < m and a_col < k:
            tileA[ty, tx] = A[out_row, a_col]
        else:
            tileA[ty, tx] = float32(0.0)

        if b_row < k and out_col < n:
            tileB[ty, tx] = B[b_row, out_col]
        else:
            tileB[ty, tx] = float32(0.0)

        cuda.syncthreads()

        # phase2:
        # threads within the block
        # will read elements from tileA and tileB (smem)
        
        for i in range(tile_dim):
            product += tileA[ty, i] * tileB[i, tx]

        cuda.syncthreads()

    if out_row < m and out_col < n:
        C[out_row, out_col] = product

m = 1000
n = 800
k = 1200

A = cp.ones(shape=(m, k), dtype=cp.float32) * 2.0
B = cp.ones(shape=(k, n), dtype=cp.float32) * 3.0
C = cp.zeros(shape=(m, n), dtype=cp.float32)

threads_per_block = (tile_dim, tile_dim)
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