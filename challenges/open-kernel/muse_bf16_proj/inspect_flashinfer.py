import dataclasses
import inspect
import os
from pathlib import Path

import torch
from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as kernel


SHAPES = (
    ("qkv_gate", 1, 2304, 8192),
    ("qkv_gate", 8, 2304, 8192),
    ("qkv_gate", 16, 2304, 8192),
    ("qkv_gate", 32, 2304, 8192),
    ("o", 1, 8192, 1024),
    ("o", 8, 8192, 1024),
    ("o", 16, 8192, 1024),
    ("o", 32, 8192, 1024),
    ("pooler", 1, 4096, 8192),
    ("pooler", 8, 4096, 8192),
    ("pooler", 16, 4096, 8192),
    ("pooler", 32, 4096, 8192),
    ("unpooler-slice", 1, 1024, 4096),
    ("unpooler-slice", 8, 1024, 4096),
    ("unpooler", 16, 8192, 4096),
    ("unpooler", 32, 8192, 4096),
)


def describe(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "__dict__"):
        return vars(value)
    return repr(value)


print("GPU:", torch.cuda.get_device_name())
properties = torch.cuda.get_device_properties(0)
print("SM count:", properties.multi_processor_count)
print("Shared memory per SM:", properties.shared_memory_per_multiprocessor)
print("Registers per SM:", properties.regs_per_multiprocessor)
print("Warp size:", properties.warp_size)

print("Module:", inspect.getfile(kernel))
print("default_tactic:", inspect.signature(kernel.default_tactic))
print("run_warp_splitk_dense:", inspect.signature(kernel.run_warp_splitk_dense))
print("WarpSplitKTactic:", inspect.signature(kernel.WarpSplitKTactic))
print("_SUPPORTED_OUTPUT_TILES:", kernel._SUPPORTED_OUTPUT_TILES)
print("_SM_SMEM_BYTES:", kernel._SM_SMEM_BYTES)
print("_CTA_RESERVED_SMEM:", kernel._CTA_RESERVED_SMEM)

source = Path(inspect.getfile(kernel))
results = Path(os.environ["GPU_RESULTS_DIR"])
(results / "dense_bf16_gemm_warp_splitk.py").write_text(source.read_text())

for layer, m, n, k in SHAPES:
    tactic = kernel.default_tactic(m, n, k)
    print(layer, m, n, k, describe(tactic))

for name in (
    "default_tactic",
    "run_warp_splitk_dense",
    "_smem_bytes",
    "_legal_stages",
    "validate_tactic",
    "_compile",
):
    value = getattr(kernel, name)
    print(f"\n===== {name} =====")
    try:
        print(inspect.getsource(value))
    except (OSError, TypeError) as error:
        print("Source unavailable:", error)
