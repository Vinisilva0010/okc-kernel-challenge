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


# v1: numerics parity (Oct 29, evening)

Why: the serving judge accepts outputs that agree with stock "nearly as well as it agrees with itself" (CLAUDE.md, Serving judge facts). Checks used: 3 of the 3 allowed by DAY_PLAN3 (1 of 20 daily checks left).
The patch (serve/patch.diff, now 639 added lines, 36 KB) carries `mk_moe/mk_moe.py` v1, the version validated by job 53b3070582c447a5. Candidate v2 (below) is staged but NOT in the patch.

## What v1 changed in mk_moe.py (v0 kept as mk_moe/mk_moe_v0.py, sub_mk3 still reproducible)
1. Deterministic split-K: each K slice stores its raw fp32 partial in its own plane `[KS, 8M, N]` (no atomics, no zeroing, no gather-time clears); the consumers add the planes in slice order 0..KS-1
   (loads batched 8 at a time, summation order unchanged). GEMM2 reuses GEMM1's plane buffer after the activation has consumed it.
2. Stock rounding points: GEMM outputs rounded to bf16 before SiLU; `silu_and_mul` as stock builds it for sm90 (bf16 gate/up, fp32 `__fdividef(g, 1 + __expf(-g)) * u`, one bf16 rounding);
   GEMM2 output rounded to bf16 (new small `mkp_reduce_k`), epilogue per slot in fp32 in ascending slot order exactly like `fused_post_expert_norm_weighted_sum`.
3. Scratch per device: 176 MB (limit 400): xg 16, inter 16, y 16, split-K planes 134 (M=256, KS 2), route tables < 1.
Consumers had to be parallelised across (row, 4 blocks): the first version (one block per row/token summing 16 planes serially) cost +37 us at M=1 (decode1 0.74x, job e029437b / eb13b2af).

## MKNUM (job 53b3070582c447a5; real-like routing; STOCK = SGLang's serving call sequence: fp32 reduce, no atomics, global-id align, fused post-norm)
| M | OURS vs STOCK rel L2 | max abs | elements differing | STOCK vs STOCK | OURS vs OURS |
|---|---|---|---|---|---|
| 1 | 5.79e-3 | 7.8e-3 | 70% | bitwise equal | bitwise equal |
| 8 | 5.79e-3 | 7.8e-3 | 68% | bitwise equal | bitwise equal |
| 16 | 5.70e-3 | 1.6e-2 | 68% | bitwise equal | bitwise equal |
| 32 | 5.72e-3 | 1.6e-2 | 66% | bitwise equal | bitwise equal |
| 128 | 5.65e-3 | 1.6e-2 | 63% | bitwise equal | bitwise equal |
| 256 | 5.71e-3 | 1.6e-2 | 56% | rel 6.2e-5 (not bitwise) | bitwise equal |
Determinism: achieved (v0 was not bitwise for M = 8, 16, 32). Parity: NOT improved (v0 vs stock 5.4-5.5e-3, v1 5.7e-3). Stock is bitwise reproducible with itself for M <= 128, so "as well as it agrees with itself" may be a very tight bar if the judge compares
single runs, or loose if it compares across batch compositions; unknown until a serving check.

## Where the difference comes from (stage study, M = 8 and 128 identical)
| stage | OURS vs STOCK rel L2 | equal elements |
|---|---|---|
| GEMM1 bf16 output | 2.8e-3 | 73% |
| silu * up (bf16) | 4.5e-3 | 62% |
| GEMM2 bf16 output | 5.4e-3 | 31% |
Against an fp32 dequant reference built with Marlin's own bf16-rounded global scale (unit-norm error of GEMM1, 87 slots): stock 2.3e-3, ours rounded to bf16 1.7e-3, **ours fp32 sum 1.7e-7 / 4.9e-7**.
So our GEMM is exact to fp32 roundoff and stock carries about 1.6e-3 of extra rounding. Source found in `kernels/jit/csrc/gemm/marlin/marlin_template.h`, `write` lambda (~lines 1355-1368):
`res = float2num(c0, c1)` (bf16 rounding of the fp32 accumulator) and THEN `res = __hmul2(res, global_scale)` (bf16 multiply by the bf16 global scale, second rounding). Stock output = `bf16(bf16(acc) * gs_bf16)`;
we multiplied the global scale in fp32 inside the GEMM and rounded once.

## Candidate v2 (staged, unchecked): mk_moe/mk_moe_v2.py
GEMM stores raw accumulators (global scale not applied); the consumers (`mkp_act_k`, `mkp_red_k`) compute `rb(rb(sum) * gs[expert of row])` with `exp_of_row` written by the route kernel.
Generated check files: `python3 tools/build_sub_mk4.py --v2 --study` -> `muse_moe_w4a16/measure_mk5_num.py` (MKNUM + stage study) and `sub_mk5.py` (clean). Only py_compile was possible here: the CUDA is unbuilt.
Expected result of the first check: GEMM1 bf16 equal to stock for nearly all elements (fp32 summation order differs from Marlin's block reduce at ~1e-7, rare one-ulp flips), final output rel L2 vs stock about 1e-4 or lower, speed unchanged.
If it holds: copy mk_moe_v2.py over mk_moe.py, rerun both generators, regenerate the patch, rerun tools/check_patch_rules.py. If not: v1 stays.

## Speed (judge, vs SGLang; measure_mk4_num.py = sub_mk4 + prepare-time study, the timed kernel_fn path is identical to sub_mk4's)
| load | sub_mk3 (v0, job 3bf003f0) | v1 (job 53b30705) |
|---|---|---|
| decode 1 | 1.698x | 1.624x |
| decode 16 | 1.293x | 1.276x |
| decode 32 | 1.096x | 1.095x |
| verify 1 | 1.371x | 1.325x |
| verify 16 | 1.492x | 1.516x |
| verify 32 | 1.484x | 1.508x |
| overall | 1.393x | 1.379x |
Small M is 1-4% lower (extra reduce kernel, plane reads), verify16/32 1.6% higher (no zeroing, no atomics). A separate clean `sub_mk4.py` check was not run (the plan's 3 checks went to the study files; the file differs only by the prepare-time study).

## Patch rule checks (tools/check_patch_rules.py on the regenerated serve/patch.diff)
4 files, all under python/sglang/ and not model_loader; .py only; no renames/modes/binary; 639 added lines, longest 210 chars (limit 400); 36,279 bytes (limit 720 KB); 0 non-ASCII lines; py_compile ok for all four files;
`muse_mk_moe.py` byte-identical to mk_moe/mk_moe.py (cmp); `git apply --check -R` clean. Nested repo serve/sglang: commit 44de60d on my-patch.

## Updated risks for Nov 5
- Exact-answer rule (risk 1 above) stands: v1 is deterministic and 5.7e-3 from stock at the bf16-output level; v2 is expected to remove most of that. Run v2's first check before the first serving check.
- Run-to-run nondeterminism (risk 2) is closed by v1 (bitwise reproducible in all tested M).
- Scratch grew from 128 MB to 176 MB per device (still far below the 400 MB limit of the plan).


# v2 result (Oct 29, evening): Marlin write-out rounding - ADOPTED, in the patch

Check: measure_mk5_num.py (= sub_mk5 + prepare-time MKNUM study), job 3a3931cbb62f4cfe, 5/5 tests pass, mk active. This was the last muse_moe_w4a16 check of the day (0 of 20 left).
v2 = v1 plus stock Marlin's write-out: the GEMM stores raw accumulators; the consumers compute `rb(rb(sum of K-slice planes) * gs_bf16[expert of row])` (`exp_of_row` from the route kernel).

## MKNUM table (OURS vs SGLang serving call sequence; v1 for comparison)
| M | v2 vs stock rel L2 | v2 max abs | v2 elements differing | v1 vs stock rel L2 | v1 elements differing | OURS vs OURS |
|---|---|---|---|---|---|---|
| 1 | 5.3e-5 | 9.8e-4 | 0.2% | 5.79e-3 | 70% | bitwise |
| 8 | 2.9e-4 | 3.9e-3 | 0.5% | 5.79e-3 | 68% | bitwise |
| 16 | 1.0e-3 | 1.2e-2 | 2.2% | 5.70e-3 | 68% | bitwise |
| 32 | 1.6e-4 | 3.9e-3 | 0.4% | 5.72e-3 | 66% | bitwise |
| 128 | 2.5e-4 | 7.8e-3 | 0.9% | 5.65e-3 | 63% | bitwise |
| 256 | 3.2e-4 | 7.8e-3 | 0.7% | 5.71e-3 | 56% | bitwise |
Stock vs stock: bitwise equal at M <= 128, rel 5.6e-5 at M = 256 (same run). Stage study (M = 8 / 128): GEMM1 bf16 output vs stock rel 2.4e-5 / 2.8e-5 with 100% of elements equal; silu * up 6.4e-5 / 1.3e-4 (100% equal);
GEMM2 bf16 output 3.0e-4 / 3.1e-4 (99% equal). So the Marlin write-out rounding was the missing piece: 20-100x closer to stock than v1. What remains: rare one-ulp flips from fp32 summation order (Marlin's block reduce vs ours),
which GEMM2 and the norm amplify to 1e-4..3e-4 at the output; M = 16 is the weakest (1.0e-3 from a few elements, max 1.2e-2). Not reached: "as well as stock agrees with itself" if that means bitwise (stock is bitwise self-equal for M <= 128);
reached: within the same order of magnitude as stock's own M = 256 self-disagreement (5.6e-5) at M = 1 and ~3-5x above it elsewhere.

## Speed table (judge, vs SGLang)
| load | v0 / sub_mk3 (job 3bf003f0) | v1 (job 53b30705) | v2 (job 3a3931cb) | v2 vs v1 |
|---|---|---|---|---|
| decode 1 | 1.698x | 1.624x | 1.583x | -2.5% |
| decode 16 | 1.293x | 1.276x | 1.249x | -2.1% |
| decode 32 | 1.096x | 1.095x | 1.081x | -1.3% |
| verify 1 | 1.371x | 1.325x | 1.313x | -0.9% |
| verify 16 | 1.492x | 1.516x | 1.468x | -3.2% |
| verify 32 | 1.484x | 1.508x | 1.473x | -2.3% |
| overall | 1.393x | 1.379x | 1.351x | -2.0% |
The extra cost is the consumers' second rounding step and the expert lookup per row, plus judge noise (SGLang's own time moved between jobs; per-case differences are of the size of earlier same-code reruns, about 2%). Accepted: parity matters more than 2%.

## Patch state
`mk_moe/mk_moe.py` is now v2 (v1 frozen as mk_moe_v1.py, v0 as mk_moe_v0.py, mk_moe_v2.py identical to mk_moe.py). Both generators rerun: `sub_mk4.py` / `sub_mk5.py` clean files regenerated (identical bodies; neither was checked as a clean file, the study file measure_mk5_num.py was, and it differs only
by the prepare-time study), `muse_mk_moe.py` in serve/sglang byte-identical to mk_moe.py (cmp). serve/patch.diff regenerated with `git diff a55ab7d5` in serve/sglang: 4 files, 649 added lines, longest 268 chars, 37,402 bytes, 0 non-ASCII lines;
tools/check_patch_rules.py: RULES OK; py_compile ok for all four files; `git apply --check -R` clean. Nested repo commit e8af105 on my-patch.

## Risks for Nov 5 (updated)
1. Exact-answer rule: now ~1e-4..1e-3 from stock at the bf16 output (was 5.7e-3) and deterministic, but not bitwise equal to stock; whether that passes the judge's comparison is still only knowable from a serving check.
2. Nondeterminism: closed (bitwise run-to-run in all tested M).
3. Startup build (8 ranks, load_inline), CUDA-graph capture inside SGLang, shared scratch buffers across streams, weight reload: unchanged from the first report; none testable here.
4. Speed margin: overall 1.351x in phase 1 (decode 32 only 1.08x); the serving gain at concurrency 32 will be small.
