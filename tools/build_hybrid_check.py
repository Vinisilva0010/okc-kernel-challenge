"""Build muse_bf16_proj/hybrid_check1.py (CHECK-only, never submitted): sub_v1.py + the embedded serve hybrid_draft.py + unit tests in prepare()."""
import base64, pathlib, zlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
base = (ROOT / "muse_bf16_proj" / "sub_v1.py").read_text()
assert base.count("def prepare(weights):") == 1
base = base.replace("def prepare(weights):", "def _prepare_base(weights):")
src = (ROOT / "serve/sglang/python/sglang/srt/speculative/hybrid_draft.py").read_bytes()
b64 = base64.b64encode(zlib.compress(src, 9)).decode()
blob = "_HY_BLOB = (\n" + "\n".join(f'    "{b64[i:i+100]}"' for i in range(0, len(b64), 100)) + "\n)\n"
body = (ROOT / "tools" / "hyb_body.txt").read_text()
(ROOT / "muse_bf16_proj" / "hybrid_check1.py").write_text(base.rstrip("\n") + "\n" + blob + body)
print("written", len(base) + len(blob) + len(body))
