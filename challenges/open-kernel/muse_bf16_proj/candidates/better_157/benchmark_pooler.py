"""Focused SS-WGMMA gate. Self-contained for just run; --cpu-test uses stdlib."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, math, os, random, statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
SHAPES=(128,256)
LAYERS={"qkv_gate":(2304,8192),"o":(8192,1024),"pooler":(4096,8192),"unpooler":(8192,4096),"router":(256,8192)}
TARGET={128:17.0,256:21.0}

def cpu_test():
    source=(HERE/"submission.py").read_text()
    ast.parse(source)
    def prmt(a,b,s):
        return sum((((a|(b<<32))>>(8*((s>>(4*j))&7)))&255)<<(8*j) for j in range(4))
    for base in (0,52,120):
        for bits in range(65536):
            high=bits>>8; exp=high&127
            code=((high>>7)<<3)|min(7,max(0,exp-base))
            decoded=(((((code>>3)<<7)|(base+(code&7)))<<8)|(bits&255))
            if not base<=exp<base+8:
                decoded=(((bits<<16)|4095)&0xFFFFFFFF)>>16
            assert decoded==bits
    rng=random.Random(5151)
    for _ in range(1000):
        base=rng.randrange(121)
        vals=[(rng.randrange(2)<<15)|(rng.randrange(base,base+8)<<8)|rng.randrange(256) for _ in range(8)]
        codes=[((v>>15)<<3)|(((v>>8)&127)-base) for v in vals]
        low=[sum((v&255)<<(8*j) for j,v in enumerate(vals[h:h+4])) for h in (0,4)]
        c=sum((codes[j]|(codes[j+4]<<4))<<(8*j) for j in range(4))
        h0=((c&0x07070707)+base*0x01010101)|((c<<4)&0x80808080)
        h1=(((c>>4)&0x07070707)+base*0x01010101)|(c&0x80808080)
        regs=[prmt(low[0],h0,0x5140),prmt(low[0],h0,0x7362),prmt(low[1],h1,0x5140),prmt(low[1],h1,0x7362)]
        assert [(r>>s)&65535 for r in regs for s in (0,16)]==vals
    assert {r*128+(((c//8)^(r&7))*16)+(c&7)*2 for r in range(64) for c in range(64)}==set(range(0,8192,2))
    for bn,q in ((128,5),(256,3)):
        st=8192+bn*128+6144
        frag=(q*st+24*q+127)//128*128
        smem=1024+frag+(bn//8)*128*16
        assert smem<=227*1024 and st%1024==0 and q>=2
        points=[]
        for rank in range(2):
            for tid in range(128):
                lane=tid&31; row=(tid>>5)*16+(lane>>2)
                for j in range(rank*bn//16,(rank+1)*bn//16):
                    tok=rank*bn//2+8*(j-rank*bn//16)+2*(lane&3)
                    points.extend(((tok,row),(tok,row+8),(tok+1,row),(tok+1,row+8)))
        assert len(points)==len(set(points))==bn*64
        assert set(points)=={(r,c) for r in range(bn) for c in range(64)}
        print(f"CPU M={bn} threads=288 cluster=2 stages={q} shared={smem} bytes")
    print("CPU PASS: codec mathematics, vector PRMT, swizzle and output coverage; CUDA execution remains untested")

def make_input(t,m,kind):
    g=t.Generator(device="cuda").manual_seed(20261030+m)
    x=t.randn((m,8192),generator=g,device="cuda")
    if kind=="outliers":
        x[:,t.randperm(8192,generator=g,device="cuda")[:81]]*=30
    rows=t.arange(m,device="cuda") if kind=="probes" else t.arange(0,m,8,device="cuda")
    cols=t.randint(8192,(len(rows),),generator=g,device="cuda")
    boundary=t.tensor([0,1,7,8,15,16,31,32,63,64,127,128,4095,4096,8191],device="cuda")
    cols[:len(boundary)]=boundary
    signs=t.randint(2,(len(rows),),generator=g,device="cuda")*2-1
    x[rows]=0
    x[rows,cols]=(signs*2.0**t.randint(3,8,(len(rows),),generator=g,device="cuda")).float()
    return {"x":x.bfloat16(),"config":{"m":m,"n":4096,"k":8192,"layer":"pooler"}}

def check(t,x,out,w):
    ref=(x.double()@w.double().T).bfloat16()
    if out.shape!=ref.shape or out.dtype!=ref.dtype:
        return {"ok":False,"reason":"shape/dtype"}
    d=out.double()-ref.double(); norm=ref.double().norm(dim=-1)
    l2=float(d.norm()/ref.double().norm().clamp(min=1e-12))
    row=float((d.norm(dim=-1)/norm.clamp(min=float(.1*norm.mean()))).max())
    probes=((x!=0).sum(-1)==1).nonzero().flatten()
    probe=float((d[probes].norm(dim=-1)/norm[probes].clamp(min=1e-30)).max())
    return {"ok":bool(t.isfinite(out).all()) and l2<=.01 and row<=.05 and probe<=.001,
            "l2":l2,"row":row,"probe":probe}

def capture(t,fn):
    fn(); t.cuda.synchronize()
    graph=t.cuda.CUDAGraph()
    with t.cuda.graph(graph):
        out=fn()
    graph.replay(); t.cuda.synchronize()
    return graph,out

def gpu_main(repeats):
    import torch as t
    spec=importlib.util.spec_from_file_location("b157_candidate",HERE/"submission.py")
    c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    print("GPU:",t.cuda.get_device_name(),flush=True)
    gen=t.Generator(device="cuda").manual_seed(5151)
    weights={name:(t.randn((n,k),generator=gen,device="cuda")/math.sqrt(k)).to(
        t.float32 if name=="router" else t.bfloat16) for name,(n,k) in LAYERS.items()}
    # Initialize only the established public pooler cuBLASLt plans.
    baseline=c.tt_lt_prepare_cublaslt(weights)
    if any((m,"pooler") not in baseline["configs"] for m in SHAPES):
        raise RuntimeError("Pinned pooler baseline unavailable: image/library version mismatch")
    plan=c.b157_prepare_pooler(weights["pooler"]); state={"b157_pooler":plan}
    print("Resources:",plan["resources"],flush=True)
    result={"gpu":t.cuda.get_device_name(),"sha256":hashlib.sha256((HERE/"submission.py").read_bytes()).hexdigest(),
            "resources":plan["resources"],"packed_bytes":plan["pack"]["dense_bytes"]+plan["pack"]["escape_bytes"],
            "shapes":{},"promote":False}
    dest=Path(os.environ.get("GPU_RESULTS_DIR",str(HERE)))/"better_157_pooler.json"
    flush=t.empty(128*1024*1024//4,device="cuda",dtype=t.float32)
    for m in SHAPES:
        tests={}
        for kind in ("normal","outliers","probes"):
            data=make_input(t,m,kind)
            out=c.kernel_fn(data,state); t.cuda.synchronize()
            tests[kind]=check(t,data["x"],out,weights["pooler"])
            print(f"M={m} {kind}: {tests[kind]}",flush=True)
        result["shapes"][m]={"correctness":tests}
        dest.write_text(json.dumps(result,indent=2)+"\n")
        if not all(v["ok"] for v in tests.values()):
            raise RuntimeError(f"M={m} correctness failed; timing rejected")
        data=make_input(t,m,"normal")
        bg,bo=capture(t,lambda:c.tt_lt_run_cublaslt(data["x"],"pooler",4096,baseline))
        cg,co=capture(t,lambda:c.kernel_fn(data,state))
        for label,out in (("baseline",bo),("candidate",co)):
            assert check(t,data["x"],out,weights["pooler"])["ok"],f"{label} graph failed"
        graphs=(bg,cg)
        for _ in range(5):
            for graph in graphs:
                flush.zero_(); graph.replay()
        t.cuda.synchronize()
        samples=[[],[]]; ratios=[]
        start,end=(t.cuda.Event(enable_timing=True) for _ in range(2))
        for rep in range(repeats):
            for idx in ((0,1) if rep%2==0 else (1,0)):
                flush.zero_(); start.record(); graphs[idx].replay(); end.record(); end.synchronize()
                samples[idx].append(start.elapsed_time(end)*1000)
            if rep%2:
                ratios.append(math.sqrt(samples[0][-2]/samples[1][-2]*samples[0][-1]/samples[1][-1]))
        bu,cu=(statistics.median(s) for s in samples)
        result["shapes"][m].update(baseline_us=bu,candidate_us=cu,paired_speedup=statistics.median(ratios),
                                  strong_target_pass=cu<=TARGET[m],samples_us=samples)
        print(f"M={m}: baseline={bu:.3f}us candidate={cu:.3f}us paired={statistics.median(ratios):.3f}x",flush=True)
        dest.write_text(json.dumps(result,indent=2)+"\n")
    # 18/22us initial gates project only ~1.5676x under the proposed hybrid.
    # Both strong gates are required; full integration/official timing follows.
    result["promote"]=all(result["shapes"][m]["strong_target_pass"] for m in SHAPES)
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print("PROMOTE:",result["promote"],"Saved:",dest,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-test",action="store_true")
    p.add_argument("--repeats",type=int,default=40)
    args=p.parse_args()
    if args.repeats<2 or args.repeats%2:
        p.error("--repeats must be even and >=2")
    cpu_test()
    if not args.cpu_test:
        gpu_main(args.repeats)
