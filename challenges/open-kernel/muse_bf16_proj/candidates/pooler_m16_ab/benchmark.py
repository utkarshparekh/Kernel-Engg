"""Single-configuration A/B test. --cpu-test needs only the Python standard library."""

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import time
import traceback
import unittest

import candidate

HERE = Path(__file__).resolve().parent
SOURCE_BASELINE_SHA256 = "2f8f50f019238f3916b1f2cd3105840751f57cf359db89e9772c5ab7b689c3ae"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def byte_perm(a, b, selectors):
    data = a.to_bytes(4, "little") + b.to_bytes(4, "little")
    return sum(data[(selectors >> (4 * i)) & 7] << (8 * i) for i in range(4))


def decode8_cpu(low0, low1, codes, base):
    b = base * 0x01010101
    h0 = ((codes & 0x07070707) + b) | ((codes << 4) & 0x80808080)
    h1 = (((codes >> 4) & 0x07070707) + b) | (codes & 0x80808080)
    return [byte_perm(low0, h0, 0x5140), byte_perm(low0, h0, 0x7362),
            byte_perm(low1, h1, 0x5140), byte_perm(low1, h1, 0x7362)]


def paired_stats(samples, cycles):
    """Bootstrap whole balanced-order cycles, preserving within-cycle dependence."""
    result = {}
    rng = random.Random(971)
    for control in ("raw", "current"):
        blocks = [statistics.mean(math.log(s[control] / s["packed"]) for s in samples
                                  if s["cycle"] == i) for i in range(cycles)]
        boots = sorted(math.exp(statistics.mean(rng.choices(blocks, k=cycles)))
                       for _ in range(2000))
        point = math.exp(statistics.mean(blocks))
        result[control] = {
            "geometric_speedup": point,
            "bootstrap_ci95": [boots[49], boots[1949]],
            "speedup_ci95": [min(boots[49],point/1.015), max(boots[1949],point*1.015)],
            "minimum_relative_uncertainty": 0.015,
            "method": "cluster bootstrap of balanced six-order cycles; ratio control/packed",
        }
    return result


def traffic(stats):
    unique_x, out = 16 * 8192 * 2, 16 * 4096 * 2
    repeated_x = 128 * unique_x
    return {
        **stats,
        "unique_activation_bytes": unique_x,
        "logical_activation_global_read_bytes": repeated_x,
        "logical_activation_shared_write_bytes": repeated_x,
        "logical_activation_ldmatrix_read_bytes": 4 * repeated_x,
        "output_store_bytes": out,
        "compulsory_cold_bytes_lower_bound": stats["prepared_bytes"] + unique_x + out,
        "logical_global_requested_bytes": stats["prepared_bytes"] + repeated_x + out,
        "measured_dram_read_bytes": None, "measured_dram_write_bytes": None,
        "note": "Logical requests can hit caches. Actual HBM traffic requires hardware counters.",
    }


class CpuTests(unittest.TestCase):
    def test_all_bf16_bit_patterns_lossless(self):
        # Includes zeros, subnormals, infinities and every NaN payload, both signs.
        # Each record fits a seven-bit exponent window; the scalar codec is exact.
        words = [(sign << 15) | (high << 8) | low
                 for start in range(0, 128, 8) for sign in range(2)
                 for high in range(start, start + 8) for low in range(256)]
        for pos in range(0, len(words), 512):
            part = words[pos:pos + 512]
            payload, h = candidate.pack_record_cpu(part)
            self.assertEqual(len(payload), 768)
            self.assertEqual(candidate.unpack_record_cpu(payload, h), part)

    def test_raw_fallback_and_control(self):
        rng = random.Random(31)
        words = [rng.randrange(65536) for _ in range(512)]
        for compress in (True, False):
            payload, h = candidate.pack_record_cpu(words, compress)
            self.assertEqual(len(payload), 1024)
            self.assertTrue(h & 0x80000000)
            self.assertEqual(candidate.unpack_record_cpu(payload, h), words)

    def test_vector_decoder_matches_scalar_codec(self):
        rng = random.Random(91)
        for base in (0, 1, 57, 120):
            words = [(rng.randrange(2) << 15) | ((base + rng.randrange(8)) << 8)
                     | rng.randrange(256) for _ in range(512)]
            words[0], words[1] = base << 8, (base + 7) << 8
            payload, h = candidate.pack_record_cpu(words)
            for lane in range(32):
                lo = struct.unpack_from("<4I", payload, lane * 16)
                codes = struct.unpack_from("<2I", payload, 512 + lane * 8)
                decoded = decode8_cpu(lo[0], lo[1], codes[0], base)
                decoded += decode8_cpu(lo[2], lo[3], codes[1], base)
                restored = [v for pair in decoded for v in (pair & 65535, pair >> 16)]
                self.assertEqual(restored, words[lane * 16:lane * 16 + 16])

    def test_weight_fragments_cover_output_tile(self):
        points = [candidate.weight_coordinate(0, 0, w, l, v)
                  for w in range(4) for l in range(32) for v in range(16)]
        self.assertEqual(set(points), set(itertools.product(range(32), range(64))))
        self.assertEqual(len(points), len(set(points)))
        for lane in range(32):
            row = lane // 4
            for v in range(4):
                self.assertEqual(candidate.weight_coordinate(127, 127, 3, lane, v),
                                 (4096 - 8 + row, 8128 + (lane % 4) * 2
                                  + (v & 1) + (v // 2) * 8))

    def test_activation_swizzle_is_bijective_and_aligned(self):
        offsets = [candidate.shared_offset(r, c) for r in range(16) for c in range(512)]
        self.assertEqual(set(offsets), set(range(0, 16384, 2)))
        for r in range(16):
            for c in range(0, 512, 8):
                self.assertEqual(candidate.shared_offset(r, c) % 16, 0)
                self.assertEqual([candidate.shared_offset(r, c+i) for i in range(8)],
                                 list(range(candidate.shared_offset(r, c),
                                            candidate.shared_offset(r, c)+16, 2)))

    def test_ldmatrix_addresses_match_mma_a_fragments(self):
        # ldmatrix.x4: eight row addresses/matrix, four output pairs per row.
        for q in range(32):
            for lane in range(32):
                for reg in range(4):
                    address_lane = reg * 8 + lane // 4
                    source_row = address_lane % 16
                    source_col = q * 16 + (address_lane // 16) * 8 + (lane % 4) * 2
                    expected_row = lane // 4 + (reg % 2) * 8
                    expected_col = q * 16 + (reg // 2) * 8 + (lane % 4) * 2
                    self.assertEqual((source_row, source_col), (expected_row, expected_col))

    def test_accumulator_store_covers_all_outputs_once(self):
        points = [(lane // 4 + (i // 2) * 8, warp * 8 + (lane % 4) * 2 + i % 2)
                  for warp in range(4) for lane in range(32) for i in range(4)]
        self.assertEqual(set(points), set(itertools.product(range(16), range(32))))
        self.assertEqual(len(points), len(set(points)))

    def test_mixed_payload_offsets_and_descriptors(self):
        payload = bytearray()
        rng = random.Random(67)
        descriptors = []
        inputs = []
        for raw in (False, True, False, False, True):
            words = ([rng.randrange(65536) for _ in range(512)] if raw else
                     [(63 << 8) | rng.randrange(256) for _ in range(512)])
            block, h = candidate.pack_record_cpu(words)
            self.assertEqual(len(payload) % 16, 0)
            descriptors.append(h | (len(payload) // 16))
            inputs.append(words)
            payload.extend(block)
        for h, words in zip(descriptors, inputs):
            offset = (h & 0xffffff) * 16
            size = 1024 if h & 0x80000000 else 768
            self.assertEqual(candidate.unpack_record_cpu(payload[offset:offset+size], h), words)

    def test_statistical_gate_has_expected_direction(self):
        samples = [{"cycle": c, "raw": 12.0, "current": 15.0, "packed": 10.0}
                   for c in range(4) for _ in range(6)]
        result = paired_stats(samples, 4)
        self.assertAlmostEqual(result["raw"]["geometric_speedup"], 1.2)
        self.assertGreater(result["current"]["speedup_ci95"][0], 1)

    def test_python_sources_parse(self):
        for path in HERE.glob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


def validate_codec_gpu(torch, weight, state):
    restored = candidate.unpack(state)
    torch.cuda.synchronize()
    if not torch.equal(restored.view(torch.int16), weight.view(torch.int16)):
        count = int((restored.view(torch.int16) != weight.view(torch.int16)).sum())
        raise AssertionError(f"codec reconstruction differs at {count} weights")


def validate_small_format_gpu(torch):
    # Deliberately mixed raw/packed records, exponent extremes and special bits.
    bits = ((torch.arange(32 * 512, device="cuda", dtype=torch.int32).reshape(32, 512)
             & 255) | (63 << 8))
    for row, col, value in ((0,0,0x7fc1),(8,64,0),(16,128,0x7f80),
                            (24,192,0xff80),(0,256,1),(8,320,0x8000)):
        bits[row,col] = value
    weight = bits.to(torch.int16).view(torch.bfloat16)
    state = candidate.prepare(weight, compress=True)
    if not (0 < state.stats["raw_records"] < state.stats["records"]):
        raise AssertionError("adversarial codec test did not exercise both branches")
    validate_codec_gpu(torch, weight, state)
    payload = bytes(state.payload.cpu().tolist())
    desc = state.descriptors.cpu().tolist()[0]
    bit_list = bits.cpu().tolist()
    for unit in range(8):
        for warp in range(4):
            words = [bit_list[r][c] for lane in range(32) for value in range(16)
                     for r,c in [candidate.weight_coordinate(unit,0,warp,lane,value)]]
            oracle, oracle_header = candidate.pack_record_cpu(words)
            h = desc[unit][warp] & 0xffffffff
            offset = (h & 0xffffff) * 16
            if (h & 0xff000000) != oracle_header or payload[offset:offset+len(oracle)] != oracle:
                raise AssertionError(f"GPU packer disagrees with scalar oracle: unit={unit},warp={warp}")
    return state.stats


def gpu_experiment(result, cycles):
    import torch
    import baseline_snapshot as baseline
    import reference
    from checks import compare

    def check_output(data, out, ref):
        ok, msg = compare(out, ref, reference.REL_L2_TOL)
        if not ok:
            raise AssertionError(msg)
        rows = reference.probe_rows(data["x"])
        if len(rows):
            err = ((out[rows].double() - ref[rows].double()).norm(dim=-1)
                   / ref[rows].double().norm(dim=-1).clamp(min=1e-30))
            if float(err.max()) > reference.PROBE_TOL:
                raise AssertionError(f"probe relative error {float(err.max())} > {reference.PROBE_TOL}")
        return msg + f"; {len(rows)} probe rows"

    p = torch.cuda.get_device_properties(0)
    if p.major != 9:
        raise RuntimeError("This experiment is compiled for SM90 (H100/H200)")
    result["device"] = {"name": p.name, "sm_count": p.multi_processor_count,
                        "capability": list(torch.cuda.get_device_capability()),
                        "torch": torch.__version__, "cuda": torch.version.cuda}
    result["resources"] = candidate.resources()
    if result["resources"]["local_bytes_per_thread"]:
        raise RuntimeError("Compute kernel uses local memory: stop before timing")

    result["small_codec"] = validate_small_format_gpu(torch)
    # Preserve the official seeded generator/order, then retain only pooler.
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    weight = weights["pooler"]
    weights = {"pooler": weight}
    states = {"packed": candidate.prepare(weight, compress=True),
              "raw": candidate.prepare(weight, compress=False)}
    result["traffic"] = {name: traffic(state.stats) for name,state in states.items()}
    saved = result["traffic"]["raw"]["compulsory_cold_bytes_lower_bound"] - result["traffic"]["packed"]["compulsory_cold_bytes_lower_bound"]
    result["bandwidth_math"] = {"total_byte_reduction": saved,
                                "ideal_saving_us_at_4p8TB_s": saved / 4.8e12 * 1e6,
                                "runtime_prediction": False}
    for state in states.values():
        validate_codec_gpu(torch, weight, state)
    current = baseline.prepare(weights)
    fns = {"packed": lambda data: candidate.run(data["x"], states["packed"]),
           "raw": lambda data: candidate.run(data["x"], states["raw"]),
           "current": lambda data: baseline.kernel_fn(data, current)}
    data_sets = {dist: reference.generate_input(16,"pooler",seed=741+i,dist=dist)
                 for i,dist in enumerate(("normal","outliers"))}
    forced = torch.zeros((16,8192),device="cuda",dtype=torch.bfloat16)
    cols = torch.tensor([0,1,7,8,15,16,31,32,63,64,127,128,511,512,8190,8191],device="cuda")
    forced[torch.arange(16,device="cuda"),cols] = torch.tensor(
        [8,-16,32,-64,128,-8,16,-32,64,-128,8,-16,32,-64,128,-8],device="cuda",dtype=torch.bfloat16)
    data_sets["forced_probe"] = {"x":forced,"config":{"m":16,"n":4096,"k":8192,"layer":"pooler"}}
    refs = {name: reference.ref_kernel(data,weights) for name,data in data_sets.items()}
    result["correctness"] = {}
    for dist,data in data_sets.items():
        for name,fn in fns.items():
            out = fn(data)
            result["correctness"][f"{name}/{dist}"] = check_output(data,out,refs[dist])

    # Check fresh output ownership: subsequent invocations must not alter retained results.
    retained = fns["packed"](data_sets["normal"])
    retained_copy = retained.clone()
    fns["packed"](data_sets["outliers"])
    if not torch.equal(retained,retained_copy):
        raise AssertionError("candidate overwrote retained output")

    static_x = data_sets["normal"]["x"].clone()
    static_data = {"x": static_x, "config": data_sets["normal"]["config"]}
    graphs, outputs = {}, {}
    for name,fn in fns.items():
        for _ in range(5):
            fn(static_data)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = fn(static_data)
        graphs[name] = graph
    for dist,data in data_sets.items():
        static_x.copy_(data["x"])
        for name,graph in graphs.items():
            graph.replay()
            result["correctness"][f"graph/{name}/{dist}"] = check_output(data,outputs[name],refs[dist])
    static_x.copy_(data_sets["normal"]["x"])
    torch.cuda.synchronize()

    # Untimed read sweep of 128 MiB before EACH replay, including the baseline.
    flush = torch.ones(32 * 1024 * 1024,device="cuda",dtype=torch.float32)
    sink = torch.empty((),device="cuda",dtype=torch.float32)
    permutations = list(itertools.permutations(fns))
    rng = random.Random(916)
    records = []
    for cycle in range(cycles):
        orderings = permutations.copy()
        rng.shuffle(orderings)
        for order in orderings:
            events = {}
            for name in order:
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.sum(flush,dim=0,out=sink)
                start.record()
                graphs[name].replay()
                end.record()
                events[name] = (start,end)
            records.append((cycle,order,events))
    torch.cuda.synchronize()
    samples = [{"cycle":cycle,"order":list(order),
                **{name: pair[0].elapsed_time(pair[1])*1000 for name,pair in events.items()}}
               for cycle,order,events in records]
    result["timing"] = {
        "unit":"microseconds", "mode":"raw CUDA graph event time; no judge overhead correction",
        "cache_policy":"128 MiB FP32 read reduction outside events before every replay",
        "cycles":cycles,"samples_per_arm":len(samples),
        "medians":{name:statistics.median(s[name] for s in samples) for name in fns},
        "paired":paired_stats(samples,cycles),"samples":samples,
    }
    comparisons = result["timing"]["paired"]
    result["gate"] = {
        "correctness":True,"bit_exact_weights":True,"no_local_memory":True,
        "beats_both_ci95":all(v["speedup_ci95"][0]>1 for v in comparisons.values()),
        "stretch_20pct_less_time_than_current":comparisons["current"]["geometric_speedup"]>=1.25,
        "automatic_integration":False,"aggregate_score_prediction":None,
    }
    result["status"] = "timed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test",action="store_true")
    parser.add_argument("--cycles",type=int,default=10)
    args = parser.parse_args()
    if args.cpu_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CpuTests)
        passed = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        raise SystemExit(0 if passed else 1)
    if args.cycles<4:
        parser.error("at least four balanced cycles required")
    if digest(HERE/"baseline_snapshot.py") != SOURCE_BASELINE_SHA256:
        raise RuntimeError("Baseline snapshot changed: review the comparator before GPU execution")
    result = {"status":"started","case":{"m":16,"n":4096,"k":8192,"layer":"pooler"},
              "source_baseline_sha256":SOURCE_BASELINE_SHA256,
              "source_hashes":{p.name:digest(p) for p in HERE.glob("*.py")},
              "official_check":False,"official_submission":False}
    started = time.monotonic()
    error = None
    try:
        gpu_experiment(result,args.cycles)
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        error = result["error"]
    finally:
        result["wall_seconds"] = time.monotonic()-started
        directory = Path(os.environ.get("GPU_RESULTS_DIR","/tmp/pooler_m16_ab_results"))
        directory.mkdir(parents=True,exist_ok=True)
        output = directory / "pooler_m16_ab.json"
        output.write_text(json.dumps(result,indent=2))
        compact = {k:v for k,v in result.items() if k not in ("source_hashes","correctness","timing")}
        if "timing" in result:
            compact["timing"] = {k:v for k,v in result["timing"].items() if k!="samples"}
        print(json.dumps(compact,indent=2),flush=True)
        print(f"RESULT_JSON={output}",flush=True)
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
