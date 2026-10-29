"""Check serve/patch.diff against serve/RULES.md patch rules."""
import re
import sys

d = open(sys.argv[1] if len(sys.argv) > 1 else "/home/vnspo/okc/serve/patch.diff", encoding="utf-8").read()
ok = True


def fail(msg):
    global ok
    ok = False
    print("FAIL", msg)


files = re.findall(r"^diff --git a/(\S+) b/(\S+)$", d, re.M)
print("files:", [b for _, b in files])
for a, b in files:
    if a != b:
        fail(f"rename {a} -> {b}")
    if not b.startswith("python/sglang/"):
        fail(f"outside python/sglang/: {b}")
    if "/model_loader/" in b or b.startswith("python/sglang/srt/model_loader/"):
        fail(f"model_loader: {b}")
    if not b.endswith((".py", ".cu", ".cuh", ".h", ".hpp", ".c", ".cc", ".cpp", ".json")):
        fail(f"extension: {b}")
for pat in (r"^(old|new) mode ", r"^rename (from|to) ", r"^Binary files", r"^GIT binary patch", r"^deleted file mode", r"^similarity index"):
    if re.search(pat, d, re.M):
        fail(f"forbidden header /{pat}/ present")
added = [l[1:] for l in d.splitlines() if l.startswith("+") and not l.startswith("+++")]
longest = max((len(l) for l in added), default=0)
print("added lines:", len(added), "longest added line:", longest)
if longest > 400:
    fail("added line over 400 chars")
size = len(d.encode())
print("diff bytes:", size, "(limit 737280)")
if size > 720 * 1024:
    fail("diff over 720 KB")
print("non-ascii added lines:", sum(1 for l in added if any(ord(c) > 127 for c in l)))
print("RULES OK" if ok else "RULES FAILED")
sys.exit(0 if ok else 1)
