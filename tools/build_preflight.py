"""Generate the CHECK-only preflight files (never submit them):
  muse_moe_w4a16/preflight_moe1.py  = sub_mk11.py (phase-1 MK MoE file) + patched serve/sglang texts embedded + PREFLIGHT prepare()
  muse_bf16_proj/preflight_proj1.py = sub_projmk1.py + patched serve/sglang texts embedded + PREFLIGHT prepare()
Anchored: every replacement asserts exactly one match."""
import base64
import hashlib
import pathlib
import subprocess
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SG = ROOT / "serve" / "sglang"
BASE = "a55ab7d5"
P = "python/sglang/srt/"


import re


def rewritten(rel):
    """The patched text with ONLY its cross-file imports and env-flag reads redirected (the judge rejects edits of loaded modules)."""
    text = (SG / P / rel).read_text()
    if rel == "layers/muse_spark_fp4_moe.py":
        m = re.search(r"from sglang\.srt\.layers\.moe\.muse_mk_moe import \(\n((?:    \w+,\n)+)\)\n", text)
        assert m, "mk import block"
        names = re.findall(r"(\w+),", m.group(1))
        text = text.replace(m.group(0), "".join(f'{n} = _PF_MK["{n}"]\n' for n in names))
        old = "from sglang.srt.layers.quantization.muse_spark_nvfp4_moe import (\n    MuseSparkNvFp4FusedMoEMethod,\n)\n"
        assert text.count(old) == 1
        text = text.replace(old, 'MuseSparkNvFp4FusedMoEMethod = _PF_NV["MuseSparkNvFp4FusedMoEMethod"]\n')
        assert text.count("envs.SGLANG_MUSE_MK_MOE.get()") == 1
        text = text.replace("envs.SGLANG_MUSE_MK_MOE.get()", '_pf_flag("SGLANG_MUSE_MK_MOE")')
    if rel == "models/muse_spark.py":
        old = "            from sglang.srt.layers.muse_proj_mk import bind_projection\n"
        assert text.count(old) == 1
        text = text.replace(old, '            bind_projection = _PF_PM["bind_projection"]\n')
        assert text.count("envs.SGLANG_MUSE_PROJ_MK.get()") == 1
        text = text.replace("envs.SGLANG_MUSE_PROJ_MK.get()", '_pf_flag("SGLANG_MUSE_PROJ_MK")')
    return text


def blob(rel):
    data = rewritten(rel).encode()
    return base64.b64encode(zlib.compress(data, 9)).decode()


def base_sha(rel):
    r = subprocess.run(["git", "-C", str(SG), "show", f"{BASE}:{P}{rel}"], capture_output=True)
    return hashlib.sha1(r.stdout).hexdigest() if r.returncode == 0 else None


def modname(rel):
    return "sglang.srt." + rel[:-3].replace("/", ".")


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def wrap(name, b64):
    lines = [b64[i:i + 100] for i in range(0, len(b64), 100)]
    return f"{name} = (\n" + "\n".join(f'    "{l}"' for l in lines) + "\n)\n"


def common():
    return (ROOT / "tools" / "pf_common.txt").read_text()


def build(base_file, out_file, body_file, blobs, base_map):
    base = (ROOT / base_file).read_text()
    base = once(base, "def prepare(weights):", "def _prepare_base(weights):")
    body = (ROOT / "tools" / body_file).read_text()
    body = body.replace("@@BASE_MOE@@", repr(base_map)).replace("@@BASE_PROJ@@", repr(base_map))
    assert "@@" not in body
    text = base.rstrip("\n") + "\n" + common() + "".join(wrap(n, blob(r)) for n, r in blobs) + body
    text = text.replace("# okc: problem=", "# CHECK-ONLY PREFLIGHT, never submit. okc: problem=", 1) if False else text
    (ROOT / out_file).write_text(text)
    print(out_file, len(text), "bytes")


env_ok = "SGLANG_MUSE_MK_MOE = EnvBool(True)" in (SG / P / "environ.py").read_text() and "SGLANG_MUSE_PROJ_MK = EnvBool(True)" in (SG / P / "environ.py").read_text()
assert env_ok

moe_files = ["environ.py", "layers/muse_spark_fp4_moe.py", "layers/quantization/muse_spark_nvfp4_moe.py", "layers/moe/muse_mk_moe.py"]
proj_files = ["environ.py", "layers/quantization/unquant.py", "models/muse_spark.py", "layers/muse_proj_mk.py"]
mm = {modname(f): base_sha(f) for f in moe_files}
pp = {modname(f): base_sha(f) for f in proj_files}
build("muse_moe_w4a16/sub_mk11.py", "muse_moe_w4a16/preflight_moe1.py", "pf_moe_body.txt",
      [("_PF_MK", "layers/moe/muse_mk_moe.py"), ("_PF_NV", "layers/quantization/muse_spark_nvfp4_moe.py"), ("_PF_FP4", "layers/muse_spark_fp4_moe.py")], mm)
build("muse_bf16_proj/sub_projmk1.py", "muse_bf16_proj/preflight_proj1.py", "pf_proj_body.txt",
      [("_PF_PROJ", "layers/muse_proj_mk.py"), ("_PF_UNQ", "layers/quantization/unquant.py"), ("_PF_MS", "models/muse_spark.py")], pp)
