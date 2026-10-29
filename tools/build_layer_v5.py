"""Build muse_decode_layer/sub_v5.py = sub_v4.py + the MoE problem's moe5 section (sonnet_v5 unified W4A16 template, muse_moe_w4a16/sub_v1.py) adapted to the layer:
the routes read local_start from the layer's device tensor with the layer's wrap rule ((id - local_start) & 255 < 32), exactly as the layer's smallm/largem routes already do."""
import re

D = "/home/vnspo/okc/"
L = open(D + "muse_decode_layer/sub_v4.py").read()
V = open(D + "muse_moe_w4a16/sub_v1.py").read()


def rep(text, old, new, cnt=1):
    assert text.count(old) == cnt, (old[:80], text.count(old))
    return text.replace(old, new)


# ---- 1. the moe5 section of the MoE board file, from its constants to the end of moe5_experts
i = V.index("M5_STAMPS, M5_NOCOMPUTE, M5_NOSTAGE = 1, 2, 4")
j = V.index("def _kernel_moe5(data, state):")
sec = V[i:j]
# small-M route: device local_start + wrap rule
sec = rep(sec, "int* __restrict__ slotl, int n, int lo, int nloc) {\n    __shared__ int wc[4][32];",
          "int* __restrict__ slotl, int n, const int* __restrict__ lo_ptr, int nloc) {\n    __shared__ int wc[4][32];")
sec = rep(sec, "    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n    pdl_trigger();\n    (&wc[0][0])[tid] = 0;",
          "    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n    const int lo = lo_ptr[0];  // local_start: a device tensor of the layer's inputs\n    pdl_trigger();\n    (&wc[0][0])[tid] = 0;")
sec = rep(sec, "        e = (gl >= lo && gl < lo + nloc) ? gl - lo : -1;\n        local_ids[tid] = e;",
          "        const int rel = (gl - lo) & 255;  // the layer's local-id rule\n        e = rel < nloc ? rel : -1;\n        local_ids[tid] = e;")
sec = rep(sec, "void moe5_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, int64_t lo, int64_t nloc, bool small) {",
          "void moe5_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor lo, int64_t nloc, bool small) {")
sec = rep(sec, "(int)ids.numel(), (int)lo, (int)nloc);", "(int)ids.numel(), lo.data_ptr<int>(), (int)nloc);", 2)
sec = rep(sec, '"void moe5_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, int64_t lo, int64_t nloc, bool small);\\n"',
          '"void moe5_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor lo, int64_t nloc, bool small);\\n"')
# experts: start is a tensor, the count is the layer's SM_NEXP
sec = rep(sec, "def moe5_experts(x, topk_ids, sm, lm, m5, start, count, cfg,", "def moe5_experts(x, topk_ids, sm, lm, m5, start, count, cfg,")
sec = rep(sec, "Route + GEMM1 (SwiGLU fused) + GEMM2 for 1..256 tokens with the unified kernel.", "Route (start: int32 device tensor [1] = local_start, wrap rule) + GEMM1 (SwiGLU fused) + GEMM2 for 1..256 tokens with the unified kernel.")

glue = '''

# ---- the layer's moe5 entry (S12): the MoE problem's board kernel (sonnet_v5 template) for every M <= 256 whose bucket passes the self-check against the layer's Marlin experts
S12_MOE5 = True


def _lm_scratch(sm):
    """The large-M scratch (chunk table, slot list, split-K partials, counters) without building the large-M extension."""
    d, dev = sm["dims"], sm["w1"].device
    need = max(d["g1"], d["g2"]) * 8
    nbn = max(d["g1"], d["g2"]) // 4
    return {
        "ctab": torch.zeros(4 + LM_CTW * LM_ZMAX, dtype=torch.int32, device=dev),
        "slotl": torch.zeros(8 * LM_MAX_M, dtype=torch.int32, device=dev),
        "part": torch.empty(LM_ZMAX * need * 128 * 4, dtype=torch.float32, device=dev),
        "cnt": torch.zeros((2, LM_ZMAX * nbn), dtype=torch.int32, device=dev),
        "zmax": LM_ZMAX, "max_m": LM_MAX_M,
    }


def _moe5_live(num_tokens, state):
    return S12_MOE5 and state.get("moe5") is not None and state.get("smallm") is not None and moe5_cfg(num_tokens) is not None


def _experts_moe5(x, topk_weights, topk_ids, local_start, state):
    """moe5_experts and the layer's post-expert epilogue exactly as the other paths end it (Triton below 32 tokens, SGLang's fused kernel from 32)."""
    m = x.shape[0]
    y, local_ids = moe5_experts(x.contiguous(), topk_ids, state["smallm"], state["moe5_lm"], state["moe5"], local_start, SM_NEXP, moe5_cfg(m))
    w32 = topk_weights.to(torch.float32).contiguous()
    if _epilogue_path(m) == "triton":
        return epilogue_triton(y, w32, local_ids, state["post_expert_w"], block_h=_marlin_knobs(m)[2])
    return fused_post_expert_norm_weighted_sum(y, w32, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)


def _moe5_selfcheck(state):
    """{M: worst rel L2 of the moe5 experts against the layer's Marlin experts} on seeded synthetic layer-style inputs, local_start 0 and a wrapping start (inf if anything raises)."""
    res = {}
    dev = state["post_expert_w"].device
    marlin = {**state, "smallm": None, "largem": None, "moe5": None}
    for m in MOE5_SELFCHECK_MS:
        worst = 0.0
        try:
            x, w, ids0 = _smallm_selfcheck_inputs(m, 3000 + m, dev)
            for s0 in (0, 100):
                ids = ((ids0.long() + s0) % N_EXPERTS).to(torch.int32)
                start = torch.full((1,), s0, dtype=torch.int32, device=dev)
                ref = _experts(x, w, ids, start, marlin).float()
                got = _experts_moe5(x, w, ids, start, state).float()
                torch.cuda.synchronize()
                err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
                worst = max(worst, err if err == err else float("inf"))
        except Exception as ex:  # noqa: BLE001
            print(f"okc-diag moe5 selfcheck m={m} raised {type(ex).__name__}: {str(ex)[:160]}")
            worst = float("inf")
        res[m] = worst
    return res


def _prepare_moe5_for_layer(state):
    """Build the moe5 extension on top of state["smallm"] and keep the buckets whose self-check passes; a failing build/check leaves the v4 paths in place."""
    state["moe5"] = None
    if not S12_MOE5 or state.get("smallm") is None:
        return
    try:
        state["moe5_lm"] = _lm_scratch(state["smallm"])
        state["moe5"] = prepare_moe5(state["smallm"], state["moe5_lm"])
    except Exception as ex:  # noqa: BLE001  (no nvcc, build error, out of memory)
        print(f"okc-diag moe5 unavailable: {type(ex).__name__}: {str(ex)[:300]}")
        state["moe5"] = None
        return
    errs = _moe5_selfcheck(state)
    print("okc-diag moe5 selfcheck rel L2 vs Marlin: " + " ".join(f"m{m}={e:.2e}" for m, e in errs.items()))
    for mx, _c1, _c2 in MOE5_CFG:
        lo_m = {k: v for k, v in errs.items() if moe5_bucket(k) == mx}
        MOE5_ENABLE[mx] = bool(lo_m) and all(e < SELFCHECK_TOL for e in lo_m.values())
    print(f"okc-diag moe5 active={MOE5_ENABLE}")
    if not any(MOE5_ENABLE.values()):
        state["moe5"] = None
# ============================================================================================ MOE5 END
'''
# ---- 2. insert before the layer's small-M entry
anchor = "# ---- the layer's small-M entry (S4) ----"
L = rep(L, anchor, "# ============================================================================================ MOE5 BEGIN (muse_moe_w4a16/sub_v1.py sonnet_v5 section, device-start routes)\n" + sec + glue + "\n" + anchor)
# ---- 3. dispatch: moe5 first
L = rep(L, "    if local_ids is None and _smallm_live(x.shape[0], state):\n        return _experts_smallm(x, topk_weights, topk_ids, local_start, state)",
        "    if local_ids is None and _moe5_live(x.shape[0], state):\n        return _experts_moe5(x, topk_weights, topk_ids, local_start, state)\n    if local_ids is None and _smallm_live(x.shape[0], state):\n        return _experts_smallm(x, topk_weights, topk_ids, local_start, state)")
# ---- 4. prepare
L = rep(L, "        _prepare_largem_for_layer(state)\n        _glue_selfcheck(state)", "        _prepare_largem_for_layer(state)\n        _prepare_moe5_for_layer(state)\n        _glue_selfcheck(state)")
L = rep(L, "# v4: T = 32 (decode 32) takes the MoE large-M CUDA path instead of Marlin (LARGEM_MIN_TOKENS 32).\n",
        "# v4: T = 32 (decode 32) takes the MoE large-M CUDA path instead of Marlin (LARGEM_MIN_TOKENS 32).\n# v5: the MoE problem's board kernel (sonnet_v5 'moe5' unified template, +3.1% over v4 there) serves every M <= 256 whose bucket passes prepare()'s self-check against the layer's Marlin experts (S12_MOE5).\n")
open(D + "muse_decode_layer/sub_v5.py", "w").write(L)
print("ok", len(L.splitlines()))
