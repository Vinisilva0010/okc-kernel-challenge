"""serve/patch_full_hybrid.diff = git diff a55ab7d5 of the (committed) nested tree: MoE + projections + hybrid drafting. Existing variants untouched."""
import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parent.parent
SG = ROOT / "serve" / "sglang"
st = subprocess.run(["git", "-C", str(SG), "status", "--porcelain", "--", "python"], capture_output=True, text=True, check=True).stdout
assert st.strip() == "", st
d = subprocess.run(["git", "-C", str(SG), "diff", "a55ab7d5", "--", "python"], capture_output=True, check=True).stdout
(ROOT / "serve" / "patch_full_hybrid.diff").write_bytes(d)
print(len(d), "bytes")
