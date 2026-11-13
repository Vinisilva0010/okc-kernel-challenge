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


# v3 round (Oct 30): MoE port study; only the gather removal is adopted (patch now carries mk_moe v6)

Question of the round: can the public MoE techniques (PUBLIC_REFS.md step 4b, ranking in MOE_PORT_PLAN.md) be applied to the Marlin-layout path without a second weight copy and with v2's numerics? Five judge checks (muse_moe_w4a16, off the board, 12 of 20 left afterwards).

| check | job | what | result |
|---|---|---|---|
| 1 | 08d246562aee43ce | v3 hot-expert kernel (1 producer warp dequantizes each tile once into a smem fragment ring, 4 consumer warps = 4 token groups) | compiles, no deadlock, **bitwise == v2** at M = 1..256; graph us M=128 314 -> 396, M=256 426 -> 526 (**slower**) |
| 2 | 59a50e3e6ccf4c94 | v4 = same with 4 producer warps | bitwise == v2; M=256 431 -> 541 (hot kernel 215 us vs 207 us: producer count irrelevant) |
| 3 | fe4ba01c7dc94c44 | v5: Marlin dequant shifts moved from the INT pipe to IMAD / IMAD.HI (runtime-opaque multipliers), 3 variants | bitwise == v2; every variant **slower** (V1 +5..7%, V2 +14..16%, V3 +27..31% at M >= 16) |
| 4 | be1f321646344708 | v6: GEMM1 reads x rows through slot_of_row, no gather kernel, no xg buffer | bitwise == v2; **-1.0..-1.9 us per call** at all six shapes; scratch 176 -> 160 MB |
| 5 | 2515725e18324d30 | official clean file sub_mk4.py (= adopted mk_moe.py v6) | 5/5, overall **1.372x** (decode1 1.634, decode16 1.280, decode32 1.086, verify1 1.324, verify16 1.504, verify32 1.477); v2 job 3a3931cb was 1.351x: inside the +-2% noise, so no speed claim beyond the 1-2 us per call measured in check 4 |

Findings (each from a job above, nothing assumed):
1. **Sharing the dequant of a hot expert through shared memory does not pay.** The hot kernel takes ~26 us per 4-group chunk (207 us / ~8 chunks at M = 256), the same as four plain passes (~7 us each), and quadrupling the producers changed nothing. The shared operand is the dequantized bf16 fragment (2 KB per lane-tile per group vs the 512 B packed tile), so the smem traffic per group-tile grows (~3.4 KB vs ~2.1 KB) instead of the instruction count shrinking. Register-level sharing failed earlier (NT2/NT4, regs 136/208). The 13x duplicate dequant of the hot expert at verify32 therefore stays.
2. **The INT-pipe-bound model was wrong.** Replacing SHF by IMAD / IMAD.HI is monotonically slower; the Marlin dequant mix is not improvable that way. (An attempt to read the SASS histogram through cuobjdump was rejected by the judge: "starts processes". No SASS was seen.)
3. The public kernels' advantage rests on a layout that is not Marlin's (bit-spread codes, 7 INT + 2 IMAD per 8 weights, 1 launch, weights as the A operand with x in registers, wgmma for >= 17 tokens). Their kernel cannot read the Marlin tensors (fp32 global scale and raw nibble order are gone, parity rounding would change) and needs up to four weight copies (~3.6 GB per layer and GPU).
   The structurally feasible port is an RS-wgmma kernel reading Marlin words as A fragments (one word = one m16k16 A fragment of two n-blocks); it is not started (days of work, accumulation order no longer bitwise v2).

## Patch state after this round
`mk_moe/mk_moe.py` = v6 (v2 frozen as mk_moe_v2.py, v3/v4/v5 kept as records: bitwise-equal but slower, or variants). Generators rerun (`tools/build_sub_mk4.py` wrote sub_mk4.py and the SGLang module; module byte-identical to mk_moe.py, cmp).
serve/patch.diff regenerated with `git diff a55ab7d5 HEAD -- python` in serve/sglang (nested commit 9d0a99c): 4 files, 650 added lines, longest 268 chars, 37,733 bytes, 0 non-ASCII lines; tools/check_patch_rules.py: RULES OK; py_compile ok for all four files; `git apply --check -R` clean.
Risks for Nov 5: unchanged from the v2 section (numerics ~1e-4..1e-3 from stock, deterministic, startup build, graph capture inside SGLang untested). New: the GEMM1 A-operand now indexes x through `slot_of_row[row] >> 3`; routes outside the local range never appear in `slot_of_row`, so no new out-of-range read (same rows as v2's gather).
Honest size of the result: the serving MoE is still ~1.35-1.37x vs stock in the phase-1 judge (public kernels 2.3-2.8x); this round bought 1-2 us per call and four documented negatives.


# v4 round (Nov 2): public bf16 projection kernels in the patch (proj_mk)

Task: put vjeux/okc-2026's `muse_bf16_proj` kernels (Apache-2.0, credit kept in the file header) next to the MK MoE. Nothing of SGLang could be run here; the kernels were validated through 4 phase-1 checks (4 of 8 budgeted used, 6 of 10 daily left), and the SGLang wiring is statically checked only.
Single source of truth: `muse_bf16_proj/proj_mk/proj_mk.py` (derived once from `sub_pub1.py` by `tools/derive_proj_mk.py`, then edited only through anchored scripts `tools/patch_proj_mk_v2..v4.py`). `tools/build_sub_proj_mk.py [--study]` writes the phase-1 files (`sub_projmk1.py` clean, `measure_projmk1.py` = clean + CHECK-only study) and copies the module byte-identically to `serve/sglang/.../layers/muse_proj_mk.py` (cmp). `tools/patch_serve_proj_mk.py` applied the SGLang hooks.

## Step 1: where the projections run (serve/sglang, base a55ab7d5) - read from the code, shapes equal the phase-1 shapes
| projection | SGLang site | weight (TP8) | stock path on the judge (H200, bf16 backend `auto`) |
|---|---|---|---|
| qkv_gate | `MuseSpark1xAttention.qkv_proj` (`QKVOGateParallelLinear`) -> `UnquantizedLinearMethod.apply` | [2304, 8192] bf16 | `F.linear` (the dispatcher is used only for backend cutedsl) |
| o | `o_proj` (`RowParallelLinear`, then all-reduce) -> same `apply` | [8192, 1024] | `F.linear` |
| pooler | `MuseSpark1xMoE._forward_experts`: `bf16_gemm_dispatch(hidden, pre_expert_proj.weight)` (`gemm_ag` only on sm103) | [4096, 8192] replicated | `_bf16_gemm_dispatch_impl` -> `F.linear` |
| unpooler | `_reduce_and_unpool`: `gemm_local`/`gemm_ag` if multicast comm supports it (m <= 8), else `bf16_gemm_dispatch(post_expert_proj.weight)` | [8192, 4096] | `F.linear` |
| router | `forward_split2` for m > 8 (hi/lo bf16 `torch.mm`, fp32 out); m <= 8 `F.linear` on fp32 | [256, 8192] fp32 | as left |
Consequence: the "SGLang baseline" of phase 1 (`_bf16_gemm_dispatch_impl`) IS what serving runs on this hardware. The unpooler's phase-1 shapes at m <= 8 are a 1024-row slice (the `gemm_local` output); serving runs full rows there, so those kernels cannot be reused.

## Step 2: the module
- **Fixed route table** (`ROUTES_SERVING`), read off the judge's autotune log of board job 65a6e80c32b04437, no timing at load: qkv_gate sk2 (m <= 32) + gqx (33..128); o sk2 (m <= 32); pooler sk2 (m <= 32) + gq (33..128); router sk2 (9..32) + wgmma `gemm` (33..256). Everything else returns None = the caller's stock path with SGLang's own bf16 weight: prefill, M > 256, m = 129..255 for non-router layers, router/unpooler at m <= 8, o at m >= 33, qkv/pooler at m >= 129, unpooler entirely (see memory).
- **Dropped on purpose** (each a documented decision, measured or derived from the tune log): CUTLASS pooler m=256 (second JIT build with flashinfer headers, 3 us at one load), `torchT` K-major copies (4 GB for <= 6 %), m=1 GEMVs (DFlash verify has m >= 8), router `rk` (check 2: not bitwise reproducible, det False; `gemm` is det True, 3.4-6.3 us faster than stock instead of 10).
- **Lossless packing verified at bind time** (`sk2_roundtrip_ok`: the packed bytes plus the escape list are decoded back and compared bit for bit with the original bf16 weight): True for all five weights in every check. A failed packing or round trip raises inside `bind_projection`, which logs and returns None (stock stays).
- **Bind-time selftest**: every routed bucket runs once against an FP64 reference (tol 5e-3, router 2e-5) at m = 1, 3, 8, 9, 16, 24, 32, 33, 64, 100, 128, 129, 200, 256 (partial tiles included so every kernel instantiation is launched before any CUDA-graph capture); a failing bucket is switched off. Result in checks 2-3: none off.
- **Extra memory per GPU** (`torch.cuda.memory_allocated` delta around the bind, job 8672b928): qkv_gate 27.0 MB, o 12.0, pooler 48.1, router 24.0 = **111.1 MB per layer = 6.73 GB over 62 layers** for the default set. Layouts kept in check 2's comparison: full per-bucket layouts would have cost 327.4 MB per layer (19.8 GB); the lean layout (one sk2 stream per weight) costs 0.0-0.6 us per call more. Binding the unpooler as well adds 48.0 MB per layer (+2.9 GB) for 3.5-4.7 us at m 9..32 only (`SERVING_WEIGHTS` in the module). The stock bf16 weights stay (prefill), so this is pure extra; it comes out of the KV pool, which I cannot size here.

## Step 3: phase-1 check file (the generator output)
| job | file | tests | overall | note |
|---|---|---|---|---|
| 65a6e80c32b04437 | sub_pub1.py (board file, autotune) | 9/9 | 1.499x | reference |
| 1c6488e02a414a71 | measure_projmk1 (first fixed table, full layouts, rk) | 9/9 | 1.486x | CUTLASS dropped |
| a6dd769d49d64eed | same + full/lean/gemm comparison study | 9/9 | 1.482x | noise between the two ~0.3 % |
| 8672b92866334e50 | measure_projmk1 (final: lean, router gemm) | 9/9 | 1.433x | router verify16/32 1.6x/1.3x instead of 2.7x/2.5x |
| e0fa30e810d84688 | **sub_projmk1.py (clean, final)** | 9/9 | **1.443x** (CI 1.422-1.465) | the generator's output |
The ~1.50x I expected holds only for the first fixed table; the final configuration trades ~3.5 % of the phase-1 score for determinism of the router and half the memory. The board entry stays sub_pub1 (1.499x).
PROJSRV parity of the serving plan against `_bf16_gemm_dispatch_impl` (final module, job 8672b928): bf16 buckets rel L2 3.2e-5..1.5e-4, 0.02-0.37 % of outputs differ, **not bitwise** (the earlier "bitwise at m >= 128" was the cuBLAS fallback; gqx m=128 is 1.0e-4, gq m=128 1.5e-4; gq at m = 33 and 64 was bitwise in check 1); deterministic run to run in every bucket. Router fp32 logits rel 1.1e-6..1.8e-6, deterministic. Per-call graph-replay times, stock -> final (us, cold L2): qkv m8 18.6 -> 13.7, m32 18.5 -> 14.9, m128 21.2 -> 18.6; o m8 10.6 -> 9.8; pooler m8 25.7 -> 19.3, m32 26.0 -> 19.2, m128 28.5 -> 25.9; router m16 18.9 -> 8.9, m32 19.3 -> 9.4, m128 20.1 -> 13.8, m256 20.9 -> 17.5.
Derived (arithmetic on those numbers, assumes 62 identical MoE layers, micro-benchmark conditions): per verify step about 0.75 ms saved at M = 8, 1.3 ms at M = 32, 0.7 ms at M = 128, 0.2 ms at M = 256. Not a serving measurement.

## Step 4: the hooks (git diff a55ab7d5 in serve/sglang, nested commit 2edc306)
| file | change |
|---|---|
| `layers/muse_proj_mk.py` (new) | `proj_mk.py` verbatim (byte-identical, cmp) |
| `environ.py` | `SGLANG_MUSE_PROJ_MK = EnvBool(True)` |
| `layers/quantization/unquant.py` | `UnquantizedLinearMethod.apply`: if the layer carries `_proj_mk` and no bias, `run(x)`; None -> stock (qkv_proj, o_proj) |
| `models/muse_spark.py` | router `forward_split2` tries `_proj_mk.run` BEFORE `_build_split2` (stock hi/lo copies are no longer built for served shapes); pooler and unpooler call sites try `_proj_pool` / `_proj_unpool`; `post_load_weights` ends with `_bind_proj_mk()` (per layer, after loading, `empty_cache`, one log line with the extra GB; any exception logs a warning and leaves stock) |
Contract: `run(x)` returns None unless x is a 2-D, contiguous, 16-byte aligned bf16 CUDA tensor of the bound K, m in the table, not under `torch.compile`; outputs are fresh tensors (bf16, router fp32) like the stock call; no host sync, no compile, no allocation other than the output in `run`. Compile (`load_inline`, same toolchain as the MoE) and all weight layouts happen at bind time; torch's file lock serialises the 8 ranks.
Patch checks (tools/check_patch_rules.py on serve/patch.diff): 7 files under python/sglang/, none in model_loader/, .py only; 4,199 added lines, longest 346 chars (limit 400; one over-long asm operand list in a CUDA string (495 chars in the public file) was wrapped by the derive script); 216,082 bytes (limit 720 KB); 0 non-ASCII lines; **RULES OK**; `py_compile` ok for all 7 files; `git apply --check -R` clean.
Rules: weights stay as stock keeps them (bf16 projections lossless-compressed and verified bitwise; router hi/lo as stock does), activations bf16, fp32 accumulation, no top-k/attention change, every norm kept.

## Risks for Nov 5 (new, on top of the MoE ones)
1. Never run inside SGLang: `post_load_weights` binding order, the attribute hooks, 8-rank concurrent `load_inline` of a second extension (the public file's own comment says ~25 s of nvcc for the main build; not measured by me; torch's lock serialises), and capture of these calls inside SGLang's decode graphs are untested. Failure modes are logged warnings and the stock path, except a hung or trapping kernel.
2. Extra 6.73 GB per GPU taken from the KV pool / hierarchical-cache headroom; unknown effect on cache hit rate at concurrency 16/32 (where only qkv m=128, pooler m=128 and the router gain anything). First serving check should read the "Muse projection kernels bound" line and the KV pool size.
3. Numerics: ~1e-4 relative from stock on qkv/o/pooler outputs (0.02-0.4 % of bf16 values differ by an ulp), deterministic; on top of the MoE's 1e-4..1e-3. Whether the combined deviation passes "agrees as well as stock agrees with itself" is only knowable from a serving check. `SGLANG_MUSE_PROJ_MK` cannot be set by the judge, so a divergence means shrinking `SERVING_WEIGHTS` / the table or flipping the default.
4. The router fp32 logits differ at 1e-6; a near-tie in the top-8 expert choice could flip (not observed, not tested on real activations).
5. Shared per-layer counters (`rk` not used; router `gemm` uses a per-layer counter/workspace, `sk2` none): safe if one layer's call never overlaps itself on two streams; the router already runs on a side stream per call but the same layer is not re-entered.
6. Weight reload / resume would leave the compressed copies stale (bound once at load), like the MoE.


# v0.5.23 port (Nov 13): patch rebuilt on 297e54fd (SGLang v0.5.23 release)

Why: the judge now serves SGLang `297e54fd` (v0.5.23); the old diffs (base a55ab7d5) no longer apply. Nothing was run on a GPU here, no `okc serve` command was used.
Outputs: serve2/patch_full.diff, serve2/patch_full_hybrid.diff, serve2/patch_moe_only.diff (git diff against 297e54fd). Generator: tools/port_patch_v0523.py (asserts every anchor matches exactly once, checks our modules against the old blob hashes).

## Rules and flags: serve/RULES.md (old, git HEAD) vs serve2/RULES.md
| item | old (a55ab7d5) | v0.5.23 |
|---|---|---|
| base | commit a55ab7d5 (dev-muse-spark-open image) | commit 297e54fd, v0.5.23 release; judge image `lmsysorg/sglang@sha256:f832b5c3...` |
| patch size | 720 KB | **400 KB** (our diffs: 216-233 KB) |
| added line length | 400 | 400 (unchanged; ours max 346) |
| files | .py .cu .cuh .h .hpp .c .cc .cpp .json under python/sglang/, no model_loader | "source files and JSON configs under python/sglang/", nothing under srt/model_loader/, no renames/binary; **readable source: nothing compressed, encoded or obfuscated** (new) |
| precision/model rules | explicit list (8 experts, 2048 window, every norm, BF16 stays BF16, FP8 KV stays FP8, DFlash drafter unchanged, other drafting only from the request: prompt lookup / n-grams) | replaced by: lossless ("no lower precision, approximations or skipped work beyond stock"), outputs agree with stock and every request returns exactly the tokens it asks for, **at most +2 GB GPU memory per GPU at peak vs stock** (new; each 0.01 of --mem-fraction-static ~1.4 GB), TTFT <= 2x stock, "write it like an SGLang pull request: nothing special-cased to the workload or load levels, nothing that breaks other models or GPUs" (new) |
| runs | 1 scored + 3 check per day | same; check = ~25 min at 16 users, `--users 1/4/16/32`; sampled greedily (temperature 0); refused patch/flag (422) costs no run, a crashed server does |
| score | not in the old file | per load: tokens/s/GPU at stock's per-user speed / stock; geometric mean over the 4 loads |
| launch flags | fixed list in RULES | now a `flags` file you can edit (or `--flag` on check/submit) |

serve2/flags vs the old judge command: new defaults `--speculative-draft-attention-backend fa4`, `--mm-feature-transport cpu`, `--hicache-size 160` (old command had `--enable-hierarchical-cache` without size). Editable: tp 8, ep 8, moe-runner-backend marlin, mem-fraction-static 0.85, speculative-algorithm DFLASH, num-draft-tokens 8, draft-attention-backend fa4, mm-feature-transport cpu, enable-hierarchical-cache, hicache-size 160, chunked-prefill-size 16384, random-seed 1234.
Fixed: model path, draft model, modelopt_fp4, context-length 131072, enable-cache-report, incremental-streaming-output, max-running-requests 32, cuda-graph-max-bs-decode 32. Long refuse list (e.g. --kv-cache-dtype, --quantization*, --speculative-token-map, --speculative-use-rejection-sampling, --enable-tf32-matmul, --load-format).
serve/flags is byte-identical to serve2/flags (cmp). `--moe-runner-backend marlin` is still the judge default, so the Marlin path we hook is the served one.

## Hook map (old location -> new location) and contract check
| hook | old | v0.5.23 | verdict |
|---|---|---|---|
| MK MoE bind + `_forward_marlin` path | layers/muse_spark_fp4_moe.py | layers/muse_spark/muse_spark_fp4_moe.py | moved; `bind_mk_moe` and the MK branch re-anchored. `expert_tp` was removed upstream, so the `not self.expert_tp` guard is dropped. Fields used (num_local_experts, hidden_size, intermediate_size_per_partition, top_k, local_expert_offset, use_marlin, _post_expert_norm_weight/_eps/_weight_first) all still exist. `_forward_marlin` still ends in `fused_post_expert_norm_weighted_sum(..., slot_expert_ids=...)` for the weight-first norm |
| Marlin weight prep | layers/quantization/muse_spark_nvfp4_moe.py | layers/muse_spark/muse_spark_nvfp4_moe.py | moved; same `_process_weights_for_marlin` (alias, `super().process_weights_after_loading`, free natives); `layer.bind_mk_moe()` before `return result` |
| projections: linear method | layers/quantization/unquant.py `UnquantizedLinearMethod.apply` | same file | same anchor; `_bf16_gemm_dispatch_impl` gained only a contiguity/16-byte-alignment condition on the GEMV route (same maths) |
| router / pooler / unpooler / bind | models/muse_spark.py | models/muse_spark.py | same structure (`forward_split2`, `bf16_gemm_dispatch` in `_reduce_and_unpool` and in the pooler branch, `post_load_weights`). New: `MuseSpark1xRouter.forward` has a CuTe GEMV route (`router_gemv_cute`) but the served split2 path still goes through `forward_split2`, where our hook is |
| hybrid drafting | speculative/dflash_worker_v2.py | same file, much refactored (draft_parallel_layout, selector warm-up, dp gather) | 4 call sites + `_init_hybrid_draft` re-anchored (`init_cuda_graphs` tail, prefill return, after `draft_tokens[:, 1:].copy_`, before `SpeculativeSamplingMaskCapture`); `self.model_runner`, `batch.prefix_lens/extend_lens/req_pool_indices/seq_lens`, `prefix_lens`, `commit_lens`, `out_tokens`, `_selector_sample` all still there |
| Envs fields | environ.py after SGLANG_MUSE_FLYDSL_DECODE | environ.py after SGLANG_ENABLE_MUSE_SPARK_AG_NORM_FUSION (the old anchor is gone) | re-anchored; same 2 (4 with hybrid) fields |

## Marlin layout check (our MoE kernel reads it): UNCHANGED
Byte-for-byte identical between a55ab7d5 and 297e54fd (diff empty): `gptq_marlin_repack.cuh`, `kernels/ops/quantization/gptq_marlin_repack.py`, `marlin/dequant.h`, `quantization/marlin_utils.py` (permutations, `marlin_permute_scales`).
`marlin_utils_fp4.py`: only `prepare_moe_nvfp4_layer_for_marlin(layer, group_size=None)` gained an optional argument (compressed-tensors callers); the default path (modelopt, group_size 16) does the same repack, `nvfp4_marlin_process_scales` untouched. The ModelOpt call site (`prepare_moe_nvfp4_layer_for_marlin(layer)` after the w13_weight_scale_2 collapse) is unchanged.
The Marlin MoE kernel (`marlin_moe/marlin_template.h`, `moe_wna16_marlin.cuh`, `ops/moe/moe_wna16_marlin.py`) did change, but the changes are routing-side: sorted expert ids, inactive-prefix blocks, zeroing of inactive output rows (`expert_ids_are_sorted`, `has_inactive_prefix_count`); no line in the template diff touches scales, dequant or B fragments. The tensors the MK kernel reads (w13/w2 int32 tiles, fp8 scales, bf16 global scale) are the same.
No stop condition was hit. This is a static diff reading, not a GPU run: the layout validation (rel L2 2.0e-3 vs torch dequant) was not re-run on 0.5.23.

## What changed in the stock MoE path that touches our numerics (read, not measured)
1. Stock `_marlin_expert_outputs` now maps ids with `_map_topk_ids_to_local_experts` and passes `slot_expert_ids` to the epilogue; inactive (non-local) slots are skipped, as before and as MK does (ids outside [offset, offset+local) are dropped).
2. `fused_post_expert_norm_weighted_sum` (now kernels/ops/moe/muse_spark_post_expert_finalize_triton.py): same accumulation (fp32, ascending slot); it always runs the Triton kernel when `slot_expert_ids` is given (before, tokens < threshold used a torch fallback). The new `round_product` option is only used by the no-post-norm fallback, not by the weight-first path.
3. Marlin GEMMs now run with sorted/inactive-prefix routing metadata; the arithmetic per expert is the same, so our MKNUM parity tables (job 53b3070...) should still hold, but they were measured against the old stock call sequence.

## Differences from the old patch
- Patch size limit is 400 KB: full 215,747 B, full_hybrid 232,794 B, moe_only 215,790 B (all < 409,600).
- The hybrid variant is the only one that touches dflash_worker_v2.py; full and moe_only do not.
- Memory rule (+2 GB/GPU at peak vs stock): the projection module takes 6.73 GB extra per GPU (v4 report, from the code path, not measured in serving) -> **full and full_hybrid very likely break the +2 GB rule**; moe_only ships the module but keeps it off (SGLANG_MUSE_PROJ_MK default False). The MoE path adds ~176 MB scratch per device. Recommendation: submit moe_only (or shrink the projection table) unless a check shows the peak delta is below 2 GB.
- Generic-code rule ("nothing special-cased to the workload or load levels"): MK MoE serves M <= 256 and 32 local experts, hidden/intermediate 4096, top-8 only; other shapes fall back to stock.

## Checks run
tools/check_patch_rules.py (updated: 400 KB limit, encoded-blob heuristic): RULES OK for all three. `py_compile` ok for all 9 changed files. `git apply --check` ok for all three on a clean `git archive 297e54fd` export. Modules are byte-identical to the old ones (git blob hashes 296aee8e, 946fa587, b6bb41f8). Nothing was imported or run: no torch or GPU in this shell.
