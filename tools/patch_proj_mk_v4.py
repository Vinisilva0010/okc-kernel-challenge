"""Anchored patch of proj_mk.py: serving safety (no work under torch.compile; every kernel instantiation launched at bind time, before any CUDA-graph capture)."""
P = "/home/vnspo/okc/muse_bf16_proj/proj_mk/proj_mk.py"
s = open(P).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, f"anchor count {s.count(old)}: {old[:70]!r}"
    s = s.replace(old, new)


# partial-tile instantiations (m < bucket token count) and every gq / gemm tile width are launched once at bind time
rep("SELFTEST_ROWS = (1, 8, 16, 32, 33, 128, 200, 256)\n", "SELFTEST_ROWS = (1, 3, 8, 9, 16, 24, 32, 33, 64, 100, 128, 129, 200, 256)\n")
rep('        """x: bf16 [m, k], contiguous. Returns [m, n] (bf16; fp32 for the router) or None for the stock path."""\n        if not self.enabled',
    '        """x: bf16 [m, k], contiguous. Returns [m, n] (bf16; fp32 for the router) or None for the stock path."""\n'
    '        if torch.compiler.is_compiling() or not self.enabled')
assert all(len(l) <= 400 for l in s.split("\n")) and s.isascii()
open(P, "w").write(s)
print("patched")
