# MoE v3 plan (Oct 29): make the MK MoE path faster on the serving shapes

Written 14:30 Brasilia, before any v3 check (muse_moe_w4a16 has 0 checks until 21:00). Patch version stays **v2** (`mk_moe/mk_moe.py`, SERVE_PATCH_REPORT "v2 result"); v3 lives in `mk_moe/mk_moe_v3.py` and replaces it only if it wins and keeps v2's numerics.
Every number below comes from judge logs/profiles already in the repo (job ids given) or is derived from them with the formula shown; derived numbers are marked (d) and carry a +-15% uncertainty. Nothing here was run tonight.

## 1. How GEMM1 and GEMM2 split the `mkg_k` time (from existing profiles)

The judge's kernel profile reports both launches as one row (`mkg_k<2,4>`, 2 calls), and its routings change from job to job (decode32 `mkg_k` = 150..234 us across jobs a0b20c3c / 5ea1d8aa / 3bf003f0 / 53b30705 / 3a3931cb, verify32 394..479),
so the split cannot be read off a single profile. It can be derived from three measurements taken on the **same** routing (the MK study distributions of CLAUDE.md):
- GEMM1 alone (A_G2P4 = the shipped kernel, event timing with L2 flush): job f6ffa925f34a4c82 (`measure_mk_v9.py`).
- The whole pipeline in a CUDA graph at the shipped KS: job 788e51010c7f4913 (`measure_mkp_v1.py`, KS column of the shipped table).
- The glue kernels of the same pipeline: judge profile of job 5ea1d8aa3b604660 (route + gather + act + epilogue, mk2 numbers).
GEMM2 (d) = pipeline - GEMM1 - glue. Bytes: GEMM1 = 18.87 MB per expert (W 16.78 + scales 2.10), GEMM2 = 9.44 MB per expert.

| shape (study routing) | experts / 16-token groups | GEMM1 us (TB/s on expert bytes) | glue us | pipeline us | GEMM2 us (d) | GEMM2 TB/s (d) | GEMM1 us per group | GEMM2 us per group (d) |
|---|---|---|---|---|---|---|---|---|
| decode 1 (KS16) | 3 / 3 | 32 (1.77) | 11.7 | 59.1 | 15 | 1.8 | 10.7 | 5.1 |
| verify 1 (KS8) | 5 / 5 | 48 at KS16 (1.96) | 13.9 | 77.5 | ~16-20 | 2.4-3.0 | 9.6 | ~3-4 |
| decode 32 (KS4) | 21 / 21 | 148 (2.68) | 15.1 | 239.2 | 76 | 2.6 | 7.0 | 3.6 |
| verify 16 (KS2) | 24 / 28 | 193 (2.34) | 18.8 | 311.9 | 100 | 2.3 | 6.9 | 3.6 |
| verify 32 (KS2) | 23 / 39 | 257 (1.69) | 25.8 | 419.9 | 137 | 1.6 | 6.6 | 3.5 |
Cross-check: verify32 GEMM1 + GEMM2 = 257 + 137 = 394 us equals the judge profile `mkg_k` of job 3bf003f0784c4e97 (394 us); verify16 193 + 100 = 293 vs 278 (-5%). The pipeline decode32 and verify16 sums match within 10%.

Findings:
1. **GEMM2 is not slow per byte.** It runs 2.3-2.6 TB/s at decode32 / verify16, same as GEMM1 (2.3-2.7); the "1.1 TB/s" figure was v1's own-layout GEMM2 (CLAUDE.md, job ca73777b), not `mkg_k`. Only verify32 shows GEMM2 at 1.6 TB/s - and GEMM1 is at 1.69 there too.
2. **Both GEMMs cost a constant amount per 16-token group-pass**: GEMM1 6.6-7.0 us (= 18.87 MB / 2.7 TB/s), GEMM2 3.5-3.6 us (= 9.44 MB / 2.66 TB/s), independent of how many tokens are in the group.
   The TB/s on *distinct expert bytes* drops only where passes > experts: verify32 has 23 experts but 39 passes (the 201-token expert is 13 passes, four more experts have 2 passes each), verify16 24 experts / 28 passes.
   So the verify32 shortfall is duplicate dequant/mma/copy of the same weight tiles, as MOE_ANALYSIS.md said, and it is the whole story for the gap between decode32 (2.6 TB/s) and verify32 (1.6-1.7).
3. Upper bound of removing duplicate passes (every expert exactly one pass): verify32 pipeline 419.9 - 16 extra passes x (6.6 + 3.5) = 258 us (-38%); verify16 311.9 - 4 x 10.4 = 270 us (-13%); decode shapes 0.
4. Glue (route, gather, act, reduce, epilogue) is 12-26 us of these pipelines: 20% at decode1, 8% at verify32.
5. The serving verify load is 8 x concurrency tokens: concurrency 32 -> 256 tokens = verify32 (hot expert and all), concurrency 16 -> verify16. These two shapes are where the MoE time and the duplicate passes are.
Check 1 tonight must replace the derivation with a direct measurement (per-stage CUDA events on identical inputs; see section 4).

## 2. Changes ranked by expected gain

| rank | change | expected effect on the call (judge-graph pipeline time) | kill line |
|---|---|---|---|
| 1 | **Dequant each weight tile once per CTA for all token groups of a hot expert** (producer warp dequantizes to a shared-memory fragment ring, up to 4 consumer warps - one 16-token group each - do the mma). Applies to experts with >= 2 groups (>= 17 tokens). | verify32 420 -> 320-360 us (-14% to -24%); verify16 312 -> 285-300 (-4% to -9%); decode and verify1: unchanged (no hot experts) | Stop if verify32 improves < 6% **or** output is not bit-identical to v2 at the same KS **or** a no-hot shape (decode1/16/32, verify1) regresses > 2% |
| 2 | **Per-GEMM K split (KS1, KS2) and tile counts** | 0 to -3% (sweep dc806c9b: best of 32 combinations was within 3% of the shipped table everywhere; verify32 KS 2/4 -1.1%, decode1 8/16 -3%) | Stop if no shape gains >= 2% in the graph-timed sweep, keep the shipped table |
| 3 | **Fewer / fused small kernels**: take the row gather out of the route->GEMM1 hop (GEMM1 loads A rows through `slot_of_row`, saves `mkp_gather`), fuse the `mkp_act` consumer and `mkp_reduce` into one launch per GEMM boundary where possible | -3 to -6 us per call: decode1 -5..-10%, verify -1..-2% | Stop if < 3 us total |
| 4 | **PDL (programmatic dependent launch) between the six kernels** so each kernel's prologue (weight prefetch for GEMM2/GEMM1) overlaps the previous kernel's tail | -2 to -6 us per call | Stop if < 2 us |
| 5 | Cheaper dequant per pass (fewer instructions per 512 B tile) | blocked: 85-120 warp instructions per tile is near the minimum of the Marlin trick set (MOE_ANALYSIS section "next") | not planned |

Ranks 1 and 2 are the requested candidates and are implemented first (section 3). Ranks 3-4 are only worth doing if tonight's per-stage timings show the glue/ramp costs the estimates assume (they do not touch numerics).

### Why rank 1 should work where the register-based NT2/NT4 grouping did not
NT2/NT4 (checks 92d20bf7, bc266bce) kept NT accumulator sets in the *same warp*: 136-248 registers, 2-3 CTAs per SM, and the mma/dequant chain of one warp got longer, so each pass got slower than the saving.
The hot kernel splits the roles instead: one producer warp per CTA does all dequantization (about 130 warp instructions per k-step: the existing code), and 4 consumer warps each hold only one 32-register accumulator tile and do 4 `lds.128` + 1 `ldmatrix` + 8 `mma` per step (about 26 instructions).
Per k-step a 4-group chunk costs ~130 + 4 x 26 = 234 issue slots instead of 4 x 120 = 480 (-51%); a 2-group expert costs 182 vs 240 (-24%). Accumulation order per output element is unchanged (same mma operand values, same k order, same split-K planes), so the result must be **bit-identical** to v2 at the same KS - the acceptance test.
Estimated pass costs: a 4-group chunk pass ~1.5 group-pass equivalents (issue-bound model), so verify32: 21 hot group-passes -> 8 chunk-passes ~12 equivalents; total 39 -> 30 passes-equivalent = -23% (matches the 14-24% range above after the producer-ramp overhead).

## 3. Implementation (mk_moe_v3.py, v2 numerics kept)
- **Route kernel** additionally classifies experts: 1 group -> small table (`gtab`, exactly as v2, `ctl[1]` = n_small), >= 2 groups -> hot-chunk table `htab` (e, first row, rows <= 64, i.e. up to 4 consumer groups per chunk; a 201-token expert is 4 chunks), `ctl[2]` = n_hot. Row order, `slot_of_row`, `row_of_slot`, `exp_of_row` are unchanged.
- **`mkh_k` hot kernel** (160 threads: warp 0 producer, warps 1-4 consumers), grid = worst-case hot CTAs of the call's M, CTA id -> (hot chunk, K slice, 64-column tile), exits when past `ctl[2]`:
  producer = v2's cp.async ring for the 512 B Marlin tile + 64 B scales (depth 12), `mk_dqs`/`mk_dq`/`__hmul2` exactly as in `mkg_k`, writes the 4 `uint4` fragments {blo0, blo1, bhi0, bhi1} per lane into a 4-slot fragment ring (`[slot][word][lane]` so consumer reads are conflict-free `lds.128`);
  handshake with named barriers (`bar.arrive/bar.sync`, ids 1-4 full, 5-8 empty, 160 threads each); consumers: own A cp.async ring (16 tokens x 16 k per step, same swizzle/`ldmatrix`), `mk_mma(acc[2*wi], a, f.x, f.y)`, `mk_mma(acc[2*wi+1], a, f.z, f.w)` - the same operand order as v2 - then the same plane store (rows beyond the chunk's count are skipped).
- **Small kernel** `mkg_k` stays v2's, restricted to single-group experts (`ctl[1]` = n_small). Hot kernel launched first (longest), small kernel second on the same stream; the single-launch fused variant is a follow-up if the two-launch tail costs more than ~5 us.
- **Per-GEMM KS table** `(top, ks1, ks2)`; default equals v2's (same ks for both) so the bitwise test against v2 is exact; the sweep (check 3) picks the entries.
- **Study file** (CHECK only, generator `tools/build_v3.py` -> `measure_mk6_v3.py`): for M in (1, 8, 16, 32, 128, 256) with the real-like routing: (a) v3 vs v2 outputs bitwise equal (and stage tensors: GEMM1 planes, y), (b) per-stage CUDA-event times of v2 and v3 (route, gather, GEMM1, act, GEMM2, reduce, epilogue) on identical inputs, (c) graph-replay latency of whole forward v2 vs v3, (d) MKNUM vs serving stock (rel L2) for v3.
  This also settles section 1 by direct measurement.

## 4. Tonight's check plan (from 21:00 Brasilia, at most 12 checks; each check = tests + timing + the study)
1. v3 study file: correctness (bitwise vs v2) + per-stage timings. Decision: bitwise fails -> fix before anything else; passes -> continue.
2. Fix-ups from check 1 (compile errors, deadlocks, register spills) - budgeted 2 checks; if it still fails after 3 checks, kill rank 1 and keep v2.
3. KS sweep for v3 (KS1 x KS2) on the six shapes in graph timing: decide the table (rank 2).
4. Tuning the hot kernel (consumers per chunk 4 vs 2, producer ring depth, fragment ring depth): 2-3 checks.
5. Only if time remains and glue shows up in the per-stage table: ranks 3-4.
6. Final: official clean check of the adopted `sub_mk6.py` (generated, no study) + regenerate the patch (`tools/check_patch_rules.py`, py_compile) + SERVE_PATCH_REPORT "v3" section. Adoption rule: bit-identical to v2 at equal KS (or MKNUM vs serving stock no worse than v2's 1e-4..1e-3 band when KS differs), 5/5 tests, and verify16/verify32 each >= +3% in the judge's speedup table vs v2 (job 3a3931cb: verify16 1.468x, verify32 1.473x) without any load below v2 by more than 2%.

## 5. Risks
- Everything here is written without a GPU or nvcc: the first check may fail to compile or hang (named-barrier mistakes deadlock). Mitigation: study file prints a heartbeat and the v3 forward is wrapped so a failed build falls back to v2 inside the study (the official tests then still pass).
- Register / occupancy: the hot kernel has to stay <= ~100 registers per thread for 4 CTAs per SM; the compiler may exceed it (spills invalidate the timing, CLAUDE.md discipline 3): `cudaFuncGetAttributes` is printed.
- Memory-level parallelism: one producer warp per CTA keeps ~12 steps of Marlin tiles in flight (7 KB) instead of 16 warps x 8 steps; with 4-5 CTAs per SM that is ~35 KB per SM, near the ~25-30 KB needed for 3 TB/s. If bytes in flight limit the kernel, deepen the ring (smem allows 24+).
- A producer-limited hot kernel (one warp issuing 130 instructions per step, ~2.4 TB/s upper bound by my estimate) could make a lone 2-group expert slower than two plain group-passes; the kill line on "no-hot shapes" does not cover that, so the study logs per-expert-size timings and the router can send 2-group experts to the plain kernel if needed (threshold knob).
