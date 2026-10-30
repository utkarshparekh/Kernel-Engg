"""Small H200 correctness smoke test for submission.py."""

import torch

from submission import kernel_fn, prepare


def run_case(layer, m, n, k, weight_dtype, output_dtype, tolerance):
    generator = torch.Generator(device="cuda").manual_seed(2026 + m)
    x = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), generator=generator, device="cuda", dtype=weight_dtype)
    state = prepare({layer: weight})
    data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}

    actual = kernel_fn(data, state)
    expected = (x.float() @ weight.float().T).to(output_dtype)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    print(f"PASS {layer:8s} shape=({m},{n},{k}) dtype={actual.dtype}")


run_case("qkv_gate", 7, 2304, 8192, torch.bfloat16, torch.bfloat16, 1e-2)
run_case("router", 8, 256, 8192, torch.float32, torch.float32, 1e-4)
print("Challenge smoke test passed")
