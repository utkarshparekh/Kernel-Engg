"""Inspect whether the official challenge image ships QuACK's SM90 GEMM.

This script performs module/file discovery only. It does not import the GEMM
implementation, initialize CUDA, compile code, or launch a kernel.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


MODULES = (
    "quack",
    "quack.gemm",
    "quack.gemm_sm90",
    "quack.gemm_interface",
    "quack.gemm_runtime",
)


def module_info(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    if spec is None:
        return {"available": False}
    return {
        "available": True,
        "origin": spec.origin,
        "search_locations": [
            str(path) for path in spec.submodule_search_locations or ()
        ],
    }


def distribution_info() -> dict[str, Any]:
    for name in ("quack", "quack-kernels", "dao-ai-lab-quack"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        return {
            "name": distribution.metadata.get("Name", name),
            "version": distribution.version,
            "root": str(distribution.locate_file("")),
        }
    return {"installed_distribution": False}


def inspect_package_files(quack_info: dict[str, Any]) -> dict[str, Any]:
    roots = {
        Path(path) for path in quack_info.get("search_locations", ())
    }
    origin = quack_info.get("origin")
    if origin:
        roots.add(Path(origin).parent)

    matches: set[str] = set()
    gemm_sm90_excerpt = None
    for root in sorted(roots):
        if not root.is_dir():
            continue
        for pattern in ("gemm*.py", "*sm90*.py", "*hopper*.py"):
            for path in root.rglob(pattern):
                matches.add(str(path))
        candidate = root / "gemm_sm90.py"
        if candidate.is_file():
            source = candidate.read_text(errors="replace")
            gemm_sm90_excerpt = source[:16000]

    return {
        "matching_files": sorted(matches),
        "gemm_sm90_source_excerpt": gemm_sm90_excerpt,
    }


def main() -> None:
    modules = {name: module_info(name) for name in MODULES}
    result = {
        "distribution": distribution_info(),
        "modules": modules,
        "package_files": inspect_package_files(modules["quack"]),
    }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    results_directory = os.environ.get("GPU_RESULTS_DIR")
    if results_directory:
        output_path = Path(results_directory) / "quack_sm90_inventory.json"
        output_path.write_text(rendered + "\n")
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
