"""Build serve/patch_full.diff (MoE + projections, flags default on = serve/patch.diff) and serve/patch_moe_only.diff (the same tree with
SGLANG_MUSE_PROJ_MK defaulting to off: the projection hooks and the bind are inert; the projection module is still shipped but never imported
at load). The nested checkout is restored byte for byte afterwards. Anchored: the flag line must match exactly once."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SG = ROOT / "serve" / "sglang"
ENV = SG / "python" / "sglang" / "srt" / "environ.py"
BASE = "a55ab7d5"


def diff():
    r = subprocess.run(["git", "-C", str(SG), "diff", BASE, "--", "python"], capture_output=True, check=True)
    return r.stdout


status = subprocess.run(["git", "-C", str(SG), "status", "--porcelain", "--", "python"], capture_output=True, text=True, check=True).stdout
assert status.strip() == "", "nested tree must be clean (everything committed): " + status
full = diff()
assert full == (ROOT / "serve" / "patch.diff").read_bytes(), "patch_full.diff must equal the committed serve/patch.diff"
(ROOT / "serve" / "patch_full.diff").write_bytes(full)
orig = ENV.read_bytes()
text = orig.decode()
old_flag = "    SGLANG_MUSE_PROJ_MK = EnvBool(True)\n"
old_doc = "    # Serve the dense BF16 projections (qkv_proj, o_proj, pooler, router) of up to 256 tokens with the public compressed-weight kernels (layers/muse_proj_mk.py); off = stock.\n"
assert text.count(old_flag) == 1 and text.count(old_doc) == 1
new_doc = old_doc.rstrip("\n") + " OFF by default in the moe-only patch variant.\n"
try:
    ENV.write_text(text.replace(old_flag, "    SGLANG_MUSE_PROJ_MK = EnvBool(False)\n").replace(old_doc, new_doc))
    only = diff()
finally:
    ENV.write_bytes(orig)
assert ENV.read_bytes() == orig
(ROOT / "serve" / "patch_moe_only.diff").write_bytes(only)
print("full", len(full), "bytes; moe_only", len(only), "bytes")
