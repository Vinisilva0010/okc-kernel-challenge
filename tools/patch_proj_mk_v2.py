"""Anchored patch of muse_bf16_proj/proj_mk/proj_mk.py: lean sk2 layouts (one stream per weight) + router 'gemm' route + variant flags."""
import sys
P = "/home/vnspo/okc/muse_bf16_proj/proj_mk/proj_mk.py"
s = open(P).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, f"anchor count {s.count(old)}: {old[:70]!r}"
    s = s.replace(old, new)


# E1: the sk2 dispatcher is generated for the union of the full and the lean pick tables
rep("            if SK2_PICK.get((layer, tokr)) != name:\n                continue\n",
    "            if SK2_PICK.get((layer, tokr)) != name and SK2_PICK_LEAN.get((layer, tokr)) != name:\n                continue\n")
# E2: lean pick table (one sk2 layout per weight: the configs of the largest token bucket serve every bucket)
rep("SELFTEST_ROWS = (1, 8, 16, 32, 33, 128, 200, 256)\n",
    "SELFTEST_ROWS = (1, 8, 16, 32, 33, 128, 200, 256)\n"
    "SK2_PICK_LEAN = dict(SK2_PICK)   # one stream per weight: o / pooler / unpooler use their m = 32 layout for every token bucket\n"
    "for _tr in (8, 16, 32):\n"
    "    SK2_PICK_LEAN[(\"o\", _tr)] = SK2_PICK[(\"o\", 32)]\n"
    "    SK2_PICK_LEAN[(\"pooler\", _tr)] = SK2_PICK[(\"pooler\", 32)]\n"
    "SK2_PICK_LEAN[(\"unpooler\", 16)] = SK2_PICK[(\"unpooler\", 32)]\n"
    "SK2_PICK_LEAN[(\"unpooler\", 32)] = SK2_PICK[(\"unpooler\", 32)]\n")
# E3: handle constructor takes the variant
rep("    def __init__(self, name, weight, phase1=False):\n        self.name, self.phase1 = name, phase1\n",
    "    def __init__(self, name, weight, phase1=False, lean=False, router_big=\"rk\"):\n"
    "        self.name, self.phase1 = name, phase1\n        self.pick = SK2_PICK_LEAN if lean else SK2_PICK\n")
rep("        self.routes = dict((ROUTES_PHASE1 if phase1 else ROUTES_SERVING)[name])\n",
    "        self.routes = dict((ROUTES_PHASE1 if phase1 else ROUTES_SERVING)[name])\n"
    "        if name == \"router\":   # m 33..256: 'rk' (fast, not bitwise reproducible), 'gemm' (wgmma split-K) or None (stock)\n"
    "            for tb in (128, 256):\n"
    "                if router_big is None:\n                    self.routes.pop(tb, None)\n"
    "                else:\n                    self.routes[tb] = router_big\n")
rep("            picks = {SK2_PICK[(name, tr)] for tr in (8, 16, 32) if _tm(tr, False) in self.routes and self.routes[_tm(tr, False)] == \"sk2\"}\n"
    "            if name == \"unpooler\" and not phase1:\n                picks.discard(\"slice8_r192\")\n",
    "            picks = {self.pick[(name, tr)] for tr in (8, 16, 32) if self.routes.get(tr) == \"sk2\"}\n")
rep("        if \"rk\" in used:\n",
    "        if \"gemm\" in used:\n"
    "            floats = 0\n"
    "            for (ln, bn), (nwg, splits, _st) in GEMM_CONFIGS.items():\n"
    "                if ln == \"router\":\n                    floats = max(floats, (self.n // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))\n"
    "            state[\"mid_ws\"] = torch.empty(floats, dtype=torch.float32, device=dev)\n"
    "            state[\"mid_cnt\"] = torch.zeros(self.n // 16, dtype=torch.int32, device=dev)\n"
    "        if \"rk\" in used or \"gemm\" in used:\n")
# E4: route / run use the handle's pick table; gemm route
rep("        if r == \"sk2\" and _sk2_name(self.name, m) not in self.state[\"sk2\"]:\n",
    "        if r == \"sk2\" and self.pick[(self.name, _sk2_tokr(m))] not in self.state[\"sk2\"]:\n")
rep("        if r == \"sk2\":\n            return _run_sk2(x, name, m, n, self.k, st)\n",
    "        if r == \"sk2\":\n"
    "            ci, stream, tab, pmax, b4, _nr = st[\"sk2\"][self.pick[(name, _sk2_tokr(m))]]\n"
    "            out = torch.empty((m, n), dtype=torch.float32 if name == \"router\" else torch.bfloat16, device=x.device)\n"
    "            _get_ext().sk2(stream, x, out, tab, pmax, b4, m, n, self.k, ci, _sk2_tokr(m), 0, st[\"dummy\"])\n"
    "            return out\n"
    "        if r == \"gemm\":\n"
    "            bn = 64 if m <= 64 else 128 if m <= 128 else 256\n"
    "            nwg, splits, _st = GEMM_CONFIGS[(name, bn)]\n"
    "            out = torch.empty((m, n), dtype=torch.float32, device=x.device)\n"
    "            _get_ext().gemm90(st[\"router_cat\"], st[\"dummy\"], x, out, st[\"mid_ws\"], st[\"mid_cnt\"], m, n, st[\"router_cat\"].shape[1], self.k,\n"
    "                              splits, nwg, bn, 0)\n"
    "            return out\n")
rep("def bind_projection(name, weight, phase1=False):", "def bind_projection(name, weight, phase1=False, lean=False, router_big=\"rk\"):")
rep("        return ProjHandle(name, weight, phase1)\n", "        return ProjHandle(name, weight, phase1, lean, router_big)\n")
rep("print(f\"[proj_mk] {name}: not bound ({type(ex).__name__}: {str(ex)[:200]})\", file=sys.stderr)",
    "print(f\"[proj_mk] {name}: not bound ({type(ex).__name__}: {str(ex)[:600]})\", file=sys.stderr)")
assert all(len(l) <= 400 for l in s.split("\n")) and s.isascii()
open(P, "w").write(s)
print("patched")
