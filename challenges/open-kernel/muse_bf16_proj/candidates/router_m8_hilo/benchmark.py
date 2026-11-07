"""One fixed M=8 router candidate vs the current implementation. No sweep."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import traceback
import unittest

import plan

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CpuTests(unittest.TestCase):
    def test_uncompressed_budget_and_parallelism(self):
        g = plan.geometry()
        self.assertEqual(g["original_weight_bytes"], 8 * 2**20)
        self.assertEqual(g["execution_weight_bytes"], g["original_weight_bytes"])
        self.assertEqual(g["scratch_bytes"], 128 * 2**10)
        self.assertEqual(g["partial_programs"], 128)

    def test_partition_covers_both_weight_planes_once(self):
        visits = set()
        for split in range(plan.SPLITS):
            for offset in range(plan.PART_K):
                virtual = split * plan.PART_K + offset
                key = (virtual // plan.K, virtual % plan.K)
                self.assertNotIn(key, visits)
                visits.add(key)
        self.assertEqual(visits, {(half,k) for half in range(2) for k in range(plan.K)})
        self.assertEqual(plan.PART_K % plan.BK, 0)

    def test_partial_and_reduction_address_ownership(self):
        visited = set()
        for split in range(plan.SPLITS):
            for program_n in range(plan.N // plan.BN):
                for row in range(plan.M):
                    for col in range(program_n*plan.BN,(program_n+1)*plan.BN):
                        address = split*plan.M*plan.N + row*plan.N + col
                        self.assertNotIn(address, visited)
                        visited.add(address)
        self.assertEqual(visited, set(range(plan.SPLITS*plan.M*plan.N)))

    def test_hilo_retains_residual_required_by_precision(self):
        w = 1.00390625
        high,low = plan.hilo(w)
        self.assertGreater(abs(high-w)/abs(w), 1e-4)
        self.assertEqual(high+low,w)
        rng = random.Random(31)
        for _ in range(1000):
            w = plan.f32(rng.gauss(0,1/math.sqrt(plan.K)))
            high,low = plan.hilo(w)
            self.assertLessEqual(abs(high+low-w)/max(abs(w),1e-30), 8e-6)

    def test_virtual_k_dot_matches_original_fp32_weights(self):
        rng = random.Random(73)
        k,n,m,splits = 64,5,3,16
        weights = [[plan.f32(rng.gauss(0,0.1)) for _ in range(k)] for _ in range(n)]
        xs = [[plan.bf16(rng.gauss(0,1)) for _ in range(k)] for _ in range(m)]
        pairs = [[plan.hilo(v) for v in row] for row in weights]
        for x in xs:
            ref,actual = [],[]
            for col in range(n):
                parts = []
                for split in range(splits):
                    acc = 0.0
                    for virtual in range(split*(2*k//splits),(split+1)*(2*k//splits)):
                        acc = plan.f32(acc + x[virtual%k]*pairs[col][virtual%k][virtual//k])
                    parts.append(acc)
                actual.append(sum(parts))
                ref.append(sum(a*b for a,b in zip(x,weights[col])))
            relative = math.sqrt(sum((a-b)**2 for a,b in zip(actual,ref)) / sum(b*b for b in ref))
            self.assertLess(relative,1e-4)

    def test_paired_gate_direction_and_uncertainty(self):
        samples = [{"cycle":i,"current":10.0,"candidate":8.0}
                   for i in range(5) for _ in range(2)]
        r = plan.paired_stats(samples,5)
        self.assertAlmostEqual(r["geometric_raw_speedup"],1.25)
        self.assertLess(r["ci95_with_uncertainty_floor"][0],1.25)
        self.assertGreater(r["ci95_with_uncertainty_floor"][0],1)

    def test_sources_parse_and_baseline_is_pinned(self):
        for p in HERE.glob("*.py"):
            ast.parse(p.read_text(),filename=str(p))
        self.assertEqual(digest(HERE/"baseline_snapshot.py"),plan.BASELINE_SHA256)


def resources(kernel):
    return {"registers_per_thread":getattr(kernel,"n_regs",None),
            "spill_count":getattr(kernel,"n_spills",None),
            "shared_bytes":getattr(kernel.metadata,"shared",None)}


def gpu_run(result,cycles,directory):
    import torch
    import candidate
    import baseline_snapshot as baseline
    import reference

    device = torch.cuda.get_device_properties(0)
    if device.major != 9:
        raise RuntimeError("This experiment requires an SM90 GPU")
    result["device"] = {"name":device.name,"sms":device.multi_processor_count,
                        "torch":torch.__version__,"cuda":torch.version.cuda}
    result["correctness"] = {}

    def validate(name,data,out,weights):
        ok,message = reference.check_implementation(data,out,weights)
        result["correctness"][name] = {"pass":ok,"message":message}
        if not ok:
            raise AssertionError(f"{name}: {message}")

    def inputs(seed):
        cases = {dist:reference.generate_input(8,"router",seed+i,dist=dist)
                 for i,dist in enumerate(("normal","outliers"))}
        x = torch.zeros((plan.M,plan.K),device="cuda",dtype=torch.bfloat16)
        cols = torch.tensor([0,1,127,128,1023,4096,8190,8191],device="cuda")
        x[torch.arange(plan.M,device="cuda"),cols] = torch.tensor(
            [8,-16,32,-64,128,-8,16,-32],device="cuda",dtype=torch.bfloat16)
        cases["forced_probe"] = {"x":x,"config":{"m":8,"n":256,"k":8192,"layer":"router"}}
        return cases

    # Seed/order matches the official generator. No reference is imported by candidate.py.
    weights = {"router":reference.generate_weights(reference.WEIGHTS_SEED)["router"]}
    packed = candidate.prepare(weights["router"])
    if packed.numel()*packed.element_size() != weights["router"].numel()*4:
        raise AssertionError("execution weights changed byte budget")
    base_state = baseline.prepare(weights)
    cases = inputs(742)
    first,kernels = candidate.run(cases["normal"]["x"],packed,diagnostics=True)
    validate("primary/candidate/diagnostic",cases["normal"],first,weights)
    result["resources"] = {name:resources(kernel) for name,kernel in kernels.items()}
    for name,kernel in kernels.items():
        (directory/f"{name}.ptx").write_text(kernel.asm["ptx"])
        if result["resources"][name]["spill_count"] != 0:
            raise RuntimeError(f"{name}: spilling nonzero or unreported; stop before timing")
    ptx = kernels["partial"].asm["ptx"]
    result["tensor_core_instruction_verified"] = "mma.sync" in ptx or "wgmma.mma_async" in ptx
    if not result["tensor_core_instruction_verified"]:
        raise RuntimeError("No Tensor Core MMA instruction in compiled partial kernel")

    fns = {"current":lambda data:baseline.kernel_fn(data,base_state),
           "candidate":lambda data:candidate.run(data["x"],packed)}
    for dist,data in cases.items():
        for name,fn in fns.items():
            validate(f"primary/{name}/{dist}",data,fn(data),weights)

    # A second weight seed checks the same algorithm; no extra configuration is tested.
    held = {"router":reference.generate_weights(reference.WEIGHTS_SEED+1)["router"]}
    held_pair,held_base = candidate.prepare(held["router"]),baseline.prepare(held)
    for dist,data in inputs(1731).items():
        validate(f"heldout/candidate/{dist}",data,candidate.run(data["x"],held_pair),held)
        validate(f"heldout/current/{dist}",data,baseline.kernel_fn(data,held_base),held)
    del held,held_pair,held_base

    retained = fns["candidate"](cases["normal"])
    saved = retained.clone()
    fns["candidate"](cases["outliers"])
    if not torch.equal(retained,saved):
        raise AssertionError("candidate overwrote retained output")

    x = cases["normal"]["x"].clone()
    data = {"x":x,"config":cases["normal"]["config"]}
    graphs,outputs = {},{}
    for name,fn in fns.items():
        for _ in range(5):fn(data)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):outputs[name]=fn(data)
        graphs[name]=graph
    for dist,case in cases.items():
        x.copy_(case["x"])
        for name,graph in graphs.items():
            graph.replay()
            validate(f"graph/{name}/{dist}",case,outputs[name],weights)
    x.copy_(cases["normal"]["x"])
    torch.cuda.synchronize()

    flush = torch.ones(32*1024*1024,device="cuda",dtype=torch.float32)
    sink = torch.empty((),device="cuda",dtype=torch.float32)
    records=[]
    rng=random.Random(918)
    for cycle in range(cycles):
        orders=[("current","candidate"),("candidate","current")]
        rng.shuffle(orders)
        for order in orders:
            events={}
            for name in order:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.sum(flush,dim=0,out=sink)
                start.record();graphs[name].replay();end.record()
                events[name]=(start,end)
            records.append((cycle,order,events))
    torch.cuda.synchronize()
    samples=[{"cycle":c,"order":list(order),
              **{name:pair[0].elapsed_time(pair[1])*1000 for name,pair in events.items()}}
             for c,order,events in records]
    paired=plan.paired_stats(samples,cycles)
    result["timing"]={"unit":"microseconds","mode":"raw CUDA graph event time",
                      "cold_l2_flush_bytes":flush.numel()*4,
                      "samples_per_arm":len(samples),"paired":paired,
                      "medians":{name:statistics.median(s[name] for s in samples) for name in fns},
                      "samples":samples}
    result["gate"]={"correctness":True,"spills":False,
                    "significant_win":paired["ci95_with_uncertainty_floor"][0]>1,
                    "at_least_20pct_less_raw_time":paired["geometric_raw_speedup"]>=1.25,
                    "integration":False,"official_score_prediction":None}
    result["status"]="timed"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test",action="store_true")
    parser.add_argument("--cycles",type=int,default=30)
    args=parser.parse_args()
    if args.cpu_test:
        ok=unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(CpuTests)).wasSuccessful()
        raise SystemExit(0 if ok else 1)
    if args.cycles<10:parser.error("at least ten balanced cycles required")
    if digest(HERE/"baseline_snapshot.py")!=plan.BASELINE_SHA256:
        raise RuntimeError("baseline snapshot changed; review comparator")
    directory=Path(os.environ.get("GPU_RESULTS_DIR","/tmp/router_m8_hilo_results"))
    directory.mkdir(parents=True,exist_ok=True)
    result={"status":"started","plan":plan.geometry(),"official_check":False,
            "official_submission":False,"baseline_official_job":"6750c2608ef6433a",
            "baseline_official_speedup":1.1870644519612519,
            "source_hashes":{p.name:digest(p) for p in HERE.glob("*.py")}}
    started=time.monotonic()
    try:gpu_run(result,args.cycles,directory)
    except Exception:
        result["status"]="failed"
        result["error"]=traceback.format_exc()
    finally:
        result["wall_seconds"]=time.monotonic()-started
        output=directory/"router_m8_hilo.json"
        output.write_text(json.dumps(result,indent=2))
        brief={k:v for k,v in result.items() if k not in ("source_hashes","correctness","timing")}
        if "timing" in result:brief["timing"]={k:v for k,v in result["timing"].items() if k!="samples"}
        brief["correctness_checks"]=len(result.get("correctness",{}))
        print(json.dumps(brief,indent=2),flush=True)
        print(f"RESULT_JSON={output}",flush=True)
    if result["status"]!="timed":raise SystemExit(1)


if __name__=="__main__":main()
