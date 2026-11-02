"""One-time derivation of muse_bf16_proj/proj_mk/proj_mk.py from the public sub_pub1.py (vjeux/okc-2026, Apache-2.0) plus
proj_mk/proj_mk_tail.txt. After this runs, proj_mk.py is the single source of truth (edit it, not this script's output).
Every cut is anchored and asserted; the script exits non-zero if an anchor is missing or ambiguous."""
import sys

D = "/home/vnspo/okc/muse_bf16_proj/"
src = open(D + "sub_pub1.py").read()
tail = open(D + "proj_mk/proj_mk_tail.txt").read()


def once(text, s):
    assert text.count(s) == 1, f"anchor not unique ({text.count(s)}): {s[:60]!r}"
    return text.index(s)


# 1. header docstring -> new header (credit kept)
i0 = once(src, '"""Dense projections')
i1 = once(src, "import importlib.util\n")
header = '''# Based on vjeux/okc-2026 (https://github.com/vjeux/okc-2026, muse_bf16_proj.py), Copyright (c) 2026 vjeux, Apache License 2.0.
# Derived for serving by Vinisilva0010: kernel code unchanged; the prepare-time autotune, selftest and CUTLASS route are replaced by a
# fixed route table and a bitwise round-trip check of the compressed weights (see the proj_mk section at the end of this file).
"""Public bf16 projection kernels (lossless 12-bit compressed weights, swap-AB tensor-core GEMV/GEMM) behind a fixed route table.

``ProjHandle.run(x)`` returns the projection output for the shapes in ROUTES_*, or None, in which case the caller runs its own
stock path with its own BF16 weight (prefill, M > 256, buckets where cuBLAS won). See the proj_mk section at the end of the file.
"""

'''
body = src[i1:]

# 2. drop the CUTLASS route (second compile, flashinfer headers, 170 s join): from its banner to the weight-preparation banner
a = once(body, "# ---------------------------------------------------------------------------------------------- CUTLASS route")
b = once(body, "# Weight preparation (host side, runs in prepare())")
b = body.rindex("# ------", 0, b)
body = body[:a] + "_cut_ext, _cut_err = None, None\n\n\n" + body[b:]

# 3. drop everything from the old prepare() banner on (prepare, selftest, autotune, kernel_fn): replaced by the tail
c = once(body, "def _mt_key(m):")
body = body[:c]

# 4. wrap lines over 380 chars at a '"+f"(' operand boundary (asm constraint lists inside the CUDA strings; newline = whitespace)
out = []
for line in body.split("\n"):
    while len(line) > 380:
        j = line.rfind(', "+f"(', 0, 360)
        assert j > 0, "cannot split long line: " + line[:80]
        out.append(line[: j + 1])
        line = "               " + line[j + 2:]
    out.append(line)
body = "\n".join(out)

res = header + body + tail
assert all(len(l) <= 400 for l in res.split("\n")), "line over 400"
assert res.isascii(), "non-ascii"
open(D + "proj_mk/proj_mk.py", "w").write(res)
print("wrote proj_mk.py", len(res), "bytes", res.count("\n"), "lines")
