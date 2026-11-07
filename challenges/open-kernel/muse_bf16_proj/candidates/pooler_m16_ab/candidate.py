"""Isolated exact-BF16 pooler experiment; not an official submission.

One compute binary accepts either raw or losslessly packed weight records.
The timed path reads only prepared weights and the current activations.
"""

from dataclasses import dataclass
import hashlib

M, N, K = 16, 4096, 8192
FORMAT_VERSION = 1
_extension = None


def weight_coordinate(unit, cta, warp, lane, value):
    """Original W[N,K] location of one register-fragment value."""
    return (cta * 32 + warp * 8 + (lane >> 2),
            unit * 64 + (value // 4) * 16 + ((value // 2) % 2) * 8
            + (lane & 3) * 2 + (value & 1))


def shared_offset(row, col):
    """Byte offset in one 16x512 activation buffer, in 16-byte chunks."""
    return row * 1024 + ((col // 8) ^ (row & 7)) * 16 + (col & 7) * 2


def pack_record_cpu(words, compress=True):
    """Independent scalar format oracle: 32 lanes, 16 BF16 bit patterns/lane."""
    assert len(words) == 512 and all(0 <= w < 65536 for w in words)
    exponents = [(w >> 8) & 127 for w in words]
    base = min(exponents)
    raw = not compress or max(exponents) - base > 7
    if raw:
        planes = []
        for half in range(2):
            for lane in range(32):
                for v in range(half * 8, half * 8 + 8):
                    planes.extend(words[lane * 16 + v].to_bytes(2, "little"))
        return bytes(planes), 0x80000000 | (base << 24)
    low = bytes(w & 255 for w in words)
    codes = [((w >> 15) << 3) | (((w >> 8) & 127) - base) for w in words]
    nibbles = bytearray()
    for lane in range(32):
        for half in range(2):
            start = lane * 16 + half * 8
            nibbles.extend(codes[start + i] | (codes[start + i + 4] << 4)
                           for i in range(4))
    return low + bytes(nibbles), base << 24


def unpack_record_cpu(payload, header):
    raw, base = bool(header & 0x80000000), (header >> 24) & 127
    words = []
    for lane in range(32):
        for v in range(16):
            if raw:
                offset = (v // 8) * 512 + lane * 16 + (v % 8) * 2
                words.append(int.from_bytes(payload[offset:offset + 2], "little"))
            else:
                code_byte = payload[512 + lane * 8 + (v // 8) * 4 + v % 4]
                code = (code_byte >> (4 * ((v % 8) // 4))) & 15
                words.append(payload[lane * 16 + v] | ((base + (code & 7)) << 8)
                             | ((code >> 3) << 15))
    return words


CPP_SOURCE = r"""
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
void launch_pooler(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t);
void launch_unpack(uintptr_t, uintptr_t, uintptr_t, int, int, uintptr_t);
std::vector<int64_t> kernel_resources();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch_pooler);
  m.def("unpack", &launch_unpack);
  m.def("resources", &kernel_resources);
}
"""

CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>

static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ int sx(int row, int col) {
  return row * 1024 + ((col / 8) ^ (row & 7)) * 16 + (col & 7) * 2;
}
struct Loaded { uint4 lo, hi; unsigned header; };
struct Fragment { unsigned b0,b1,b2,b3,b4,b5,b6,b7; };

__device__ __forceinline__ Loaded load_record(
    const unsigned char* __restrict__ payload, unsigned h, int lane) {
  const unsigned char* p = payload + ((h & 0x00ffffffu) * 16u);
  Loaded r;
  r.header = h;
  r.lo = *reinterpret_cast<const uint4*>(p + lane * 16);
  if (h & 0x80000000u) {
    r.hi = *reinterpret_cast<const uint4*>(p + 512 + lane * 16);
  } else {
    uint2 codes = *reinterpret_cast<const uint2*>(p + 512 + lane * 8);
    r.hi = make_uint4(codes.x, codes.y, 0u, 0u);
  }
  return r;
}

__device__ __forceinline__ void decode8(
    unsigned l0, unsigned l1, unsigned codes, unsigned base,
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  unsigned b = base * 0x01010101u;
  unsigned h0 = ((codes & 0x07070707u) + b) | ((codes << 4) & 0x80808080u);
  unsigned h1 = (((codes >> 4) & 0x07070707u) + b) | (codes & 0x80808080u);
  r0 = __byte_perm(l0, h0, 0x5140);
  r1 = __byte_perm(l0, h0, 0x7362);
  r2 = __byte_perm(l1, h1, 0x5140);
  r3 = __byte_perm(l1, h1, 0x7362);
}
__device__ __forceinline__ Fragment decode(const Loaded& r) {
  Fragment f;
  if (r.header & 0x80000000u) {
    f = {r.lo.x,r.lo.y,r.lo.z,r.lo.w,r.hi.x,r.hi.y,r.hi.z,r.hi.w};
  } else {
    unsigned base = (r.header >> 24) & 127u;
    decode8(r.lo.x,r.lo.y,r.hi.x,base,f.b0,f.b1,f.b2,f.b3);
    decode8(r.lo.z,r.lo.w,r.hi.y,base,f.b4,f.b5,f.b6,f.b7);
  }
  return f;
}

__device__ __forceinline__ void copy_x(
    unsigned char* dst, const __nv_bfloat16* x, int group, int tid) {
  // 16 rows x 64 chunks/row, 16 bytes/chunk: 16 KiB per buffer.
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    int chunk = tid + i * 128, row = chunk / 64, col = (chunk % 64) * 8;
    unsigned dest = smem_addr(dst + sx(row, col));
    const void* src = x + row * 8192 + group * 512 + col;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dest), "l"(src));
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void step(
    const unsigned char* xbuf, int col, int lane, unsigned b0, unsigned b1,
    float& c0, float& c1, float& c2, float& c3) {
  int row = lane & 15, xcol = col + (lane / 16) * 8;
  unsigned address = smem_addr(xbuf + sx(row, xcol));
  unsigned a0,a1,a2,a3;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(address) : "memory");
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__global__ __launch_bounds__(128) void pooler16(
    const __nv_bfloat16* __restrict__ x,
    const unsigned char* __restrict__ payload,
    const unsigned* __restrict__ descriptors, __nv_bfloat16* __restrict__ out) {
  __shared__ __align__(16) unsigned char xbuf[2][16384];
  __shared__ unsigned headers[128 * 4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    headers[tid + i * 128] = descriptors[blockIdx.x * 512 + tid + i * 128];
  __syncthreads();
  copy_x(xbuf[0], x, 0, tid);
  float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
  for (int group = 0; group < 16; ++group) {
    // Fixed-size, unrolled register prefetch. Resource report rejects spilling.
    Loaded weights[8];
    #pragma unroll
    for (int q = 0; q < 8; ++q)
      weights[q] = load_record(payload, headers[(group * 8 + q) * 4 + warp], lane);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // Waits for all readers before a later iteration overwrites the old buffer.
    __syncthreads();
    if (group + 1 < 16) copy_x(xbuf[(group + 1) & 1], x, group + 1, tid);
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
      Fragment f = decode(weights[q]);
      const unsigned char* current = xbuf[group & 1];
      step(current,q*64,   lane,f.b0,f.b1,c0,c1,c2,c3);
      step(current,q*64+16,lane,f.b2,f.b3,c0,c1,c2,c3);
      step(current,q*64+32,lane,f.b4,f.b5,c0,c1,c2,c3);
      step(current,q*64+48,lane,f.b6,f.b7,c0,c1,c2,c3);
    }
  }
  int row = lane / 4, col = blockIdx.x * 32 + warp * 8 + (lane & 3) * 2;
  *reinterpret_cast<__nv_bfloat162*>(out + row * 4096 + col) = __floats2bfloat162_rn(c0,c1);
  *reinterpret_cast<__nv_bfloat162*>(out + (row + 8) * 4096 + col) = __floats2bfloat162_rn(c2,c3);
}

__global__ void unpack_records(const unsigned char* payload, const unsigned* desc,
    unsigned short* out, int n, int k) {
  int ctas=n/32, units=k/64, lane=threadIdx.x;
  int unit=blockIdx.x/(ctas*4), rem=blockIdx.x%(ctas*4), cta=rem/4, warp=rem%4;
  Fragment f=decode(load_record(payload,desc[(cta*units+unit)*4+warp],lane));
  unsigned vals[8]={f.b0,f.b1,f.b2,f.b3,f.b4,f.b5,f.b6,f.b7};
  #pragma unroll
  for(int v=0;v<16;++v) {
    int row=cta*32+warp*8+lane/4;
    int col=unit*64+(v/4)*16+((v/2)%2)*8+(lane&3)*2+(v&1);
    out[row*k+col]=static_cast<unsigned short>(vals[v/2]>>((v&1)*16));
  }
}

void launch_pooler(uintptr_t x, uintptr_t p, uintptr_t d, uintptr_t o, uintptr_t s) {
  pooler16<<<128,128,0,reinterpret_cast<cudaStream_t>(s)>>>(
      reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const unsigned char*>(p),
      reinterpret_cast<const unsigned*>(d), reinterpret_cast<__nv_bfloat16*>(o));
  check(cudaGetLastError());
}
void launch_unpack(uintptr_t p, uintptr_t d, uintptr_t o, int n, int k, uintptr_t s) {
  unpack_records<<<(n/8)*(k/64),32,0,reinterpret_cast<cudaStream_t>(s)>>>(
      reinterpret_cast<const unsigned char*>(p), reinterpret_cast<const unsigned*>(d),
      reinterpret_cast<unsigned short*>(o),n,k);
  check(cudaGetLastError());
}
std::vector<int64_t> kernel_resources() {
  cudaFuncAttributes a; check(cudaFuncGetAttributes(&a,pooler16));
  int blocks=0; check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,pooler16,128,0));
  return {a.numRegs,static_cast<int64_t>(a.sharedSizeBytes),
          static_cast<int64_t>(a.localSizeBytes),blocks,a.binaryVersion,a.ptxVersion};
}
"""


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load_inline
        digest = hashlib.sha256((CPP_SOURCE + CUDA_SOURCE).encode()).hexdigest()[:12]
        _extension = load_inline(
            name=f"pooler16_ab_{digest}", cpp_sources=CPP_SOURCE, cuda_sources=CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v",
                              "-gencode=arch=compute_90,code=sm_90"],
            with_cuda=True, verbose=True,
        )
    return _extension


@dataclass
class Packed:
    payload: object
    descriptors: object
    n: int
    k: int
    stats: dict


def prepare(weight, *, compress):
    import torch
    if (not weight.is_cuda or weight.dtype != torch.bfloat16 or weight.ndim != 2
            or not weight.is_contiguous()):
        raise ValueError("expected contiguous CUDA BF16 W[N,K]")
    n, k = weight.shape
    if n % 32 or k % 64 or n * k * 2 >= (1 << 28):
        raise ValueError("format requires N%32=K%64=0 and payload <256 MiB")
    units, ctas = k // 64, n // 32
    device = weight.device
    # Build lane fragments directly from the documented mma.m16n8k16 mapping.
    u = torch.arange(units, device=device)[:, None, None, None, None]
    c = torch.arange(ctas, device=device)[None, :, None, None, None]
    w = torch.arange(4, device=device)[None, None, :, None, None]
    lane = torch.arange(32, device=device)[None, None, None, :, None]
    v = torch.arange(16, device=device)[None, None, None, None, :]
    rows = c * 32 + w * 8 + lane // 4
    cols = u * 64 + v // 4 * 16 + (v // 2 % 2) * 8 + (lane % 4) * 2 + v % 2
    bits = weight.view(torch.int16).to(torch.int32) & 65535
    fragment = bits[rows, cols].reshape(-1, 32, 16)
    exp = (fragment >> 8) & 127
    base = exp.amin(dim=(1, 2))
    raw = (exp.amax(dim=(1, 2)) - base > 7) if compress else torch.ones_like(base, dtype=torch.bool)
    sizes = torch.where(raw, 1024, 768).to(torch.int64)
    offsets = sizes.cumsum(0) - sizes
    payload = torch.empty(int(sizes.sum().item()), device=device, dtype=torch.uint8)
    raw_ids = raw.nonzero().flatten()
    if raw_ids.numel():
        raw_data = (fragment[raw_ids].to(torch.int16).view(torch.uint8)
                    .reshape(-1, 32, 2, 16).permute(0, 2, 1, 3).reshape(-1, 1024))
        payload[offsets[raw_ids, None] + torch.arange(1024, device=device)] = raw_data
    packed_ids = (~raw).nonzero().flatten()
    if packed_ids.numel():
        f = fragment[packed_ids]
        low = (f & 255).to(torch.uint8).reshape(-1, 512)
        code = (((f >> 15) << 3) | (((f >> 8) & 127) - base[packed_ids, None, None]))
        code = code.reshape(-1, 32, 2, 8)
        code = (code[..., :4] | (code[..., 4:] << 4)).to(torch.uint8).reshape(-1, 256)
        packed_data = torch.cat((low, code), dim=1)
        payload[offsets[packed_ids, None] + torch.arange(768, device=device)] = packed_data
    descriptor = ((offsets >> 4) | (base.to(torch.int64) << 24)
                  | (raw.to(torch.int64) << 31)).to(torch.int32)
    descriptor = descriptor.reshape(units, ctas, 4).permute(1, 0, 2).contiguous()
    nr = int(raw.sum().item())
    stats = {"format_version": FORMAT_VERSION, "compressed_enabled": compress,
             "records": int(raw.numel()), "raw_records": nr,
             "compressed_records": int(raw.numel()) - nr,
             "raw_weight_bytes": n * k * 2, "payload_bytes": payload.numel(),
             "descriptor_bytes": descriptor.numel() * descriptor.element_size(),
             "correction_table_bytes": 0, "temporary_output_bytes": 0}
    stats["prepared_bytes"] = stats["payload_bytes"] + stats["descriptor_bytes"]
    return Packed(payload, descriptor, n, k, stats)


def run(x, state):
    import torch
    if (x.shape != (M, K) or x.dtype != torch.bfloat16 or not x.is_cuda
            or not x.is_contiguous() or (state.n, state.k) != (N, K)
            or state.payload.device != x.device):
        raise ValueError("compute experiment is fixed to contiguous BF16 [16,8192] x [4096,8192].T")
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    extension().launch(x.data_ptr(), state.payload.data_ptr(), state.descriptors.data_ptr(),
                       out.data_ptr(), torch.cuda.current_stream(x.device).cuda_stream)
    return out


def unpack(state):
    import torch
    out = torch.empty((state.n, state.k), device=state.payload.device, dtype=torch.bfloat16)
    extension().unpack(state.payload.data_ptr(), state.descriptors.data_ptr(), out.data_ptr(),
                       state.n, state.k, torch.cuda.current_stream(out.device).cuda_stream)
    return out


def resources():
    keys = ("registers_per_thread", "static_shared_bytes", "local_bytes_per_thread",
            "max_resident_blocks_per_sm", "binary_version", "ptx_version")
    return dict(zip(keys, extension().resources()))
