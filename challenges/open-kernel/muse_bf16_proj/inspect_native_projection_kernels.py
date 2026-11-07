"""Inventory projection-kernel capabilities in the official challenge image.

This script performs no compilation, kernel launch, or benchmark.  It records
installed versions, module availability, callable signatures, source paths,
and relevant packaged files so later experiments use only capabilities that
actually exist in the judge environment.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any


PACKAGES = (
    "torch",
    "triton",
    "flashinfer-python",
    "sglang",
    "nvidia-cutlass-dsl",
    "cutlass",
)

MODULES = (
    "flashinfer",
    "flashinfer.gemm",
    "flashinfer.gemm.gemm_base",
    "flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk",
    "flashinfer.experimental.dense_projection_gemm",
    "sglang",
    "sglang.kernels.ops.gemm.hopper_bf16_gemv",
    "sglang.kernels.ops.gemm.tiny_gemm",
    "sglang.kernels.ops.gemm.small_gemm_bf16",
    "sglang.srt.layers.quantization.unquant",
    "cutlass",
    "cute",
    "nvidia_cutlass_dsl",
)

CALLABLES = (
    ("flashinfer.gemm", "mm_bf16"),
    (
        "flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk",
        "run_warp_splitk_dense",
    ),
    (
        "flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk",
        "default_tactic",
    ),
    ("sglang.kernels.ops.gemm.hopper_bf16_gemv", "hopper_bf16_gemv"),
    ("sglang.kernels.ops.gemm.tiny_gemm", "tiny_gemm_bf16"),
    ("sglang.kernels.ops.gemm.small_gemm_bf16", "n128k512_gemm_bf16"),
    ("sglang.kernels.ops.gemm.small_gemm_bf16", "n32k5120_gemm_bf16"),
    (
        "sglang.srt.layers.quantization.unquant",
        "_bf16_gemm_dispatch_impl",
    ),
    (
        "sglang.srt.layers.quantization.unquant",
        "initialize_bf16_gemm_config",
    ),
)

FILE_PATTERNS = (
    "*dense*projection*gemm*",
    "*hopper*bf16*gemv*",
    "*tiny*gemm*",
    "*cutlass*",
    "*cute*dsl*",
    "*persistent*matmul*",
)


def distribution_version(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
        return {
            "installed": True,
            "version": distribution.version,
            "root": str(distribution.locate_file("")),
        }
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}


def module_info(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as error:
        return {
            "available": False,
            "find_error": f"{type(error).__name__}: {error}",
        }
    if spec is None:
        return {"available": False}
    return {
        "available": True,
        "origin": spec.origin,
        "search_locations": [str(path) for path in spec.submodule_search_locations or ()],
    }


def callable_info(module_name: str, attribute: str) -> dict[str, Any]:
    key = f"{module_name}:{attribute}"
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError) as error:
            signature = f"unavailable: {type(error).__name__}: {error}"
        try:
            source_file = inspect.getsourcefile(value)
        except (TypeError, OSError):
            source_file = None
        try:
            source = inspect.getsource(value)
        except (TypeError, OSError):
            source = None
        return {
            "key": key,
            "available": True,
            "signature": signature,
            "source_file": source_file,
            "source_excerpt": source[:8000] if source else None,
        }
    except Exception as error:
        return {
            "key": key,
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }


def discover_relevant_files(roots: set[Path]) -> list[str]:
    matches: set[str] = set()
    for root in sorted(roots):
        if not root.exists() or not root.is_dir():
            continue
        for pattern in FILE_PATTERNS:
            try:
                for path in root.rglob(pattern):
                    matches.add(str(path))
                    if len(matches) >= 500:
                        return sorted(matches)
            except (OSError, PermissionError):
                continue
    return sorted(matches)


def main() -> None:
    package_info = {name: distribution_version(name) for name in PACKAGES}
    modules = {name: module_info(name) for name in MODULES}
    callables = [callable_info(module, attribute) for module, attribute in CALLABLES]

    roots: set[Path] = set()
    for package in package_info.values():
        if package.get("installed") and package.get("root"):
            roots.add(Path(package["root"]))
    for module in modules.values():
        if module.get("origin"):
            roots.add(Path(module["origin"]).parent)
        roots.update(Path(path) for path in module.get("search_locations", ()))

    hardware: dict[str, Any]
    try:
        import torch

        hardware = {
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_runtime": torch.version.cuda,
        }
    except Exception as error:
        hardware = {"error": f"{type(error).__name__}: {error}"}

    inventory = {
        "hardware": hardware,
        "packages": package_info,
        "modules": modules,
        "callables": callables,
        "relevant_files": discover_relevant_files(roots),
    }

    print(json.dumps(inventory, indent=2))
    results_directory = os.environ.get("GPU_RESULTS_DIR")
    if results_directory:
        output_path = Path(results_directory) / "native_projection_inventory.json"
        output_path.write_text(json.dumps(inventory, indent=2) + "\n")
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
