# Plan (Oct 29): numerics parity + determinism for the MK MoE (serving patch v1)

Why: the serving judge accepts outputs that agree with stock nearly as well as stock agrees with itself. Our MK path differs from stock Marlin
by ~5e-3 relative and is run-to-run nondeterministic for M <= 32 (split-K fp32 atomics). Both raise the risk of failing that rule.
Budget: muse_moe_w4a16 has 4 checks until 21:00 Brasilia; use at most 3. All rules in CLAUDE.md apply. Edit only mk_moe/mk_moe.py and the
generator; regenerate sub_mk3-style files and serve/sglang's muse_mk_moe.py from it (byte-identical), never hand-edit the copies.

## Changes (in mk_moe/mk_moe.py)
1. Round where stock rounds: read stock's path (v1's _kernel_sglang = SGLang's Marlin call sequence: GEMM1 -> bf16 cache1, silu_and_mul on bf16 ->
   bf16 cache2, GEMM2 -> bf16 [M, 8, H], then fused_post_expert_norm_weighted_sum). Make ours round GEMM1's output to bf16 before SiLU * up
   (compute SiLU the way silu_and_mul does: check its code for fp32 vs bf16 math), and round GEMM2's output to bf16 before the post-norm.
2. Deterministic split-K: each K slice writes its fp32 partial to its own buffer (no atomics, no zeroing); the consumer (activation for GEMM1,
   epilogue for GEMM2) sums the KS partials in slice order 0..KS-1, then rounds as in 1. Size the partial buffers for the worst case per M bucket
   (shape only) in alloc(); keep total scratch per device <= 400 MB and report it.
3. Keep everything graph safe, fixed grids, device-side counts.

## Check 1 (measure file built by the generator: sub_mk3 + a prepare()-time study, CHECK only)
Prints `MKNUM ...` lines for M in (1, 8, 16, 32, 128, 256) with real-like routing:
- rel L2 and max |diff| of OURS vs STOCK (v1's _kernel_sglang path, i.e. SGLang Marlin) on the final bf16 output;
- STOCK vs STOCK (same inputs, two calls) and OURS vs OURS (two calls): bitwise equal? rel L2?
- the same for the old mk_moe.py (before this change) to show the improvement.
Expect: OURS vs OURS bitwise equal everywhere; OURS vs STOCK rel L2 clearly below the old ~5e-3; per-case speed within ~3 % of sub_mk3.
The official tests must pass 5/5 and the judge's speedup table is the speed check.

## Check 2 (only if check 1 is good): the clean sub_mk4.py (no study code) for the speed numbers; regenerate serve/sglang's module and
## the patch (git -C serve/sglang diff > serve/patch.diff), re-run tools/check_patch_rules.py and py_compile.

## Output
Update SERVE_PATCH_REPORT.md (new section "v1: numerics parity"): the MKNUM table, speed table vs sub_mk3, scratch size, patch rule checks.
Log every check in EXPERIMENTS.md, commit + push after each step. Do NOT submit anything; no okc serve commands.
