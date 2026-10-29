# Serving patch v0: the MK MoE inside SGLang (Oct 29, afternoon)

Status: patch written and statically checked; NOTHING of SGLang could be run here (no GPU, no torch, no sglang in this shell). The kernels are validated through phase-1 checks only.
Nothing was submitted; no `okc serve` command was run (the diff was produced with git, because CLAUDE.md bans every `okc serve` command).

## What changed (serve/patch.diff, base a55ab7d5, 4 files, 32 KB)
| file | change |
|---|---|
| `python/sglang/srt/layers/moe/muse_mk_moe.py` (new) | `mk_moe/mk_moe.py` copied verbatim by `tools/build_sub_mk3.py`: CUDA source, `MkMoE` (build / alloc / bind_layer / forward), `mk_shared(device)` |
| `python/sglang/srt/layers/muse_spark_fp4_moe.py` | `MuseSpark1xFp4Experts.bind_mk_moe()` (new), `__init__` sets `_mk_moe/_mk_layer = None`, `_forward_marlin` calls the MK path when ready |
| `python/sglang/srt/layers/quantization/muse_spark_nvfp4_moe.py` | `_process_weights_for_marlin` ends with `layer.bind_mk_moe()` (after the Marlin repack and after the native tensors are freed) |
| `python/sglang/srt/environ.py` | `SGLANG_MUSE_MK_MOE = EnvBool(True)`: off switch for A/B and debugging (default on because the judge's flags/env are fixed) |

### Hook and contract
- Bind time: `bind_mk_moe` runs once per layer right after `prepare_moe_nvfp4_layer_for_marlin`; it takes the layer's own `w13_weight / w13_weight_scale / w13_weight_scale_2 / w2_*`
  (the repacked Marlin tensors, the same ones the stock path reads), converts the two per-expert global scales to fp32 once, and keeps a reference. One `MkMoE` (compiled module + 128 MB
  of scratch buffers) is shared by all layers of a device (`mk_shared`). It binds only if: env on, Marlin backend, no expert TP, 32 local experts, hidden 4096, intermediate 4096, top-8.
  Any exception (JIT build, tensor layout) is logged and leaves the stock path active, so a failure cannot stop the server from loading. Building at weight-processing time keeps compilation and allocation out of CUDA-graph capture.
- Call time (`_forward_marlin`, after the existing `post_expert_norm is None` error and after `pre_expert_norm`): if bound, `_post_expert_norm_weight` is set, `hidden_states` is bf16 and `0 < M <= 256`,
  return `MkMoE.forward(hidden_states, topk_ids, topk_scores, local_expert_offset, num_local_experts, layer, _post_expert_norm_weight, _post_expert_norm_eps, _post_expert_norm_weight_first)`.
  Same contract as the stock return: bf16 `[M, hidden]`, already post-expert-normed and summed over the 8 routes (`fused_post_expert_norm_weighted_sum` form); ids outside
  `[local_expert_offset, +count)` are dropped as the stock `slot_expert_ids` mask does; both norm forms are implemented (`weight_first` True = rms_norm(w*y), False = rms_norm(y)*w).
  M = 0, M > 256 and non-bf16 inputs go through the unchanged stock code. The only change to stock ordering: `set_post_expert_norm` is now called before `_marlin_expert_outputs` instead of after it (idempotent).
- Deviation from the plan's signature: `forward` takes a pre-bound `MkLayer` instead of the six raw weight arguments, so the fp32 scale conversion is not repeated per call.

## JIT decision
SGLang builds its own kernels through `sglang.kernels.jit.utils.load_jit` (tvm_ffi wrappers over `.cuh` files in `kernels/jit/csrc`, e.g. `gemm/marlin_moe/moe_wna16_marlin.cuh`).
I used torch's `load_inline` for ours instead: (1) the identical host code (torch::Tensor entry points) is what the phase-1 judge compiled and ran in every check above; a tvm_ffi
rewrite of the host side (TensorView, LaunchKernel) could not be compiled or run anywhere before Nov 5, (2) one source of truth for phase 1 and serving, (3) same toolchain requirement as SGLang's JIT (nvcc, ninja).
Cost: not cached by SGLang's JIT cache, built once per process at weight load (torch's file lock serializes the 8 ranks; later ranks load the cached .so). If this proves too slow or fragile on Nov 5, the fallback is
to port only the thin host wrappers to a `.cuh` + `load_jit` module; the device code does not change.

## Phase-1 numbers (muse_moe_w4a16, vs SGLang, all 5/5 tests pass)
| load | sub_mk2 (5ea1d8aa) | measure_mk3_self (2ceb32ff) | sub_mk3 clean (3bf003f0) |
|---|---|---|---|
| decode 1 | 1.663x | 1.698x | 1.698x |
| decode 16 | 1.287x | 1.288x | 1.293x |
| decode 32 | 1.083x | 1.094x | 1.096x |
| verify 1 | 1.334x | 1.351x | 1.371x |
| verify 16 | 1.482x | 1.501x | 1.492x |
| verify 32 | 1.484x | 1.453x | 1.484x |
| overall | 1.377x | n/a (log) | 1.393x (CI 1.372-1.414) |
The reduced prepare (Marlin repack + buffers, no v1 own-layout copies) loses nothing. Checks used today for this plan: 2 of 4 allowed (4 of 20 left).
MKSELF (job 2ceb32ff, offset 0 vs ids relabelled to offset 96 on the same weights): bitwise equal for M = 1, 33, 128, 256; M = 8 max |diff| 2.4e-4 and M = 32 3.9e-3 (one bf16 ulp; split-K
atomics with KS 8 / 4 add partial sums in a run-dependent order, KS 2 adds two numbers and is exact). Relative L2 vs the stock Marlin path 5.3-5.5e-3 on random routing. Both norm forms
match a torch twin to 1.7e-3 (= bf16 output rounding); the weight_first=False branch is only exercised by this self-check, never by the judge's tests.

## Patch rule checks (tools/check_patch_rules.py on serve/patch.diff)
4 files, all under python/sglang/ and none in model_loader/; extensions .py only; no renames, mode changes, deletions or binary hunks; 581 added lines, longest 197 chars (limit 400); 32,388 bytes (limit 720 KB);
0 non-ASCII added lines. `python3 -m py_compile` passes for all four files. The new module is byte-identical to muse_moe_w4a16/mk_moe/mk_moe.py (cmp). `git apply --check -R` is clean. pyflakes/flake8 were not available.
Precision rules: weights stay NVFP4 (Marlin tensors untouched), activations bf16, fp32 accumulation, bf16 intermediate between GEMMs and bf16 output, as stock keeps them (stock rounds GEMM2's output to bf16 before the norm; ours feeds fp32 into the norm, which is higher, not lower, precision).

## Open risks to test on Nov 5 (serving checks: 3 per day)
1. Exact-answer rule ("answers must match stock's"): numerics differ from stock at the 5e-3 relative level (different accumulation, fp32 rows into the norm, fp32 SiLU), so greedy outputs could diverge from stock in some token.
   First thing to read in the first serving check. Switch: `SGLANG_MUSE_MK_MOE=0` cannot be set by the judge, so a diverging answer means changing the default to False or narrowing M.
2. Run-to-run nondeterminism for M <= 32 (KS >= 4 atomics); stock's fp32-reduce path may be deterministic. Matters if the comparison or DFlash acceptance is sensitive to it.
3. Build at startup: 8 ranks calling load_inline at weight load (nvcc + ninja availability in the image, compile time, lock contention, server start timeout). The failure mode is a logged warning and stock Marlin (not a crash), but then no speedup.
4. CUDA-graph capture inside SGLang's decode graphs was proven only in phase-1's own graph capture (all kernels on the current stream, no host sync, fixed grids); SGLang's capture of this exact call is untested.
5. Shared scratch buffers per device: unsafe if two routed-expert calls ever overlap on different streams (two-batch overlap, drafter and target interleaving). I found no such overlap in `_forward_marlin`'s callers, but did not trace the scheduler.
6. Weight reload / memory release-resume would leave `MkLayer` pointing at old tensors (bound once at load).
7. Only M <= 256 and exactly 32 local experts; other EP layouts fall back to stock. Speed for M between the tested shapes was not measured (tests only check correctness at bs 4, 33, 257, 128, 1).
8. decode 32 gains only 1.09x; the serving concurrency 32 load will see little from this patch. Prefill chunks (> 256 tokens) are unchanged.
