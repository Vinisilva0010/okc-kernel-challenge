"""Anchored patch of proj_mk.py: final serving defaults (lean layout, router gemm, configurable weight set), temporaries freed before the memory is measured."""
P = "/home/vnspo/okc/muse_bf16_proj/proj_mk/proj_mk.py"
s = open(P).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, f"anchor count {s.count(old)}: {old[:70]!r}"
    s = s.replace(old, new)


rep("PROJ_MAX_M = 256\n",
    "PROJ_MAX_M = 256\n"
    "# Weights the serving plan binds. The unpooler is left out: stock keeps m <= 8 (fused all-gather GEMM) and m >= 33, it gains 3.5-4.7 us only at\n"
    "# m 9..32 and costs 48 MB per layer (~3 GB per GPU). Add \"unpooler\" here to bind it (the table has its m = 16 / 32 buckets).\n"
    "SERVING_WEIGHTS = (\"qkv_gate\", \"o\", \"pooler\", \"router\")\n")
rep("def bind_projection(name, weight, phase1=False, lean=False, router_big=\"rk\"):",
    "def bind_projection(name, weight, phase1=False, lean=True, router_big=\"gemm\"):")
rep("    def __init__(self, name, weight, phase1=False, lean=False, router_big=\"rk\"):",
    "    def __init__(self, name, weight, phase1=False, lean=True, router_big=\"gemm\"):")
rep("        shape = PROJ_SHAPES.get(name)\n",
    "        shape = PROJ_SHAPES.get(name)\n        if not phase1 and name not in SERVING_WEIGHTS:\n            return None\n")
rep("            hi = weight.to(torch.bfloat16)\n            lo = (weight - hi.float()).to(torch.bfloat16)\n            state[\"router_cat\"] = torch.cat([hi, lo], dim=1).contiguous()\n",
    "            hi = weight.to(torch.bfloat16)\n            lo = (weight - hi.float()).to(torch.bfloat16)\n            state[\"router_cat\"] = torch.cat([hi, lo], dim=1).contiguous()\n            del hi, lo\n")
# the rk route stays in the file (check variants); serving/phase-1 defaults never use it
assert all(len(l) <= 400 for l in s.split("\n")) and s.isascii()
open(P, "w").write(s)
print("patched")
