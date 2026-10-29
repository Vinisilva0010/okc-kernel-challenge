# MK path report (Oct 29, day): full MoE on SGLang's Marlin tensors

## Result
`muse_moe_w4a16/sub_mk2.py` (official check job 5ea1d8aa3b604660, 5/5 tests pass, kernel_fn takes the MK path for every M <= 256; M > 256 falls back to v1's path):
overall 1.377x vs SGLang (95% CI 1.356-1.397). The phase-2 target of this plan (>= 1.4x on verify16 and verify32 with all tests passing) is MET on those two loads
without any v1 fallback. It is NOT a phase-1 board candidate: v1 (sub_v1.py) is 1.480x.

| load | SGLang us | sub_mk2 us | vs SGLang | sub_mk1 | v1 path (job f6ffa925f34a4c82) | plan expectation |
|---|---|---|---|---|---|---|
| decode 1 | 79.1 | 47.5 | 1.663x | 1.568x | 1.829x | >= 1.3x: met |
| decode 16 | 187.4 | 145.6 | 1.287x | 1.262x | 1.427x | >= 1.2x: met |
| decode 32 | 205.3 | 189.4 | 1.083x | 1.085x | 1.097x | >= 1.2x: NOT met |
| verify 1 | 143.2 | 107.3 | 1.334x | 1.202x | 1.486x | >= 1.3x: met |
| verify 16 | 475.5 | 320.9 | 1.482x | 1.478x | 1.568x | >= 1.4x: met |
| verify 32 | 619.2 | 417.4 | 1.484x | 1.491x | 1.570x | >= 1.4x: met |
SGLang's own time varies 461-476 us (verify16) and 609-623 us (verify32) between jobs; speedups are per job. Against v1's own-layout kernels the MK path is 4-9% slower per case
(v1 uses its own repacked weight copy, which cannot exist in the serving track).

## What the path is (sub_mk2.py, section "MK path")
route kernel (1 block: local ids, per-expert counts, expert-sorted slot list, 16-token group table, device-side n_rows / n_groups; also row_of_slot) ->
gather (x rows in expert order) + zero the fp32 accumulators -> GEMM1 `mkg_k<2,4>` on bank w13 -> SiLU(gate) * up (fp32 -> bf16) -> GEMM2 `mkg_k<2,4>` on bank w2 ->
fused scatter + post-expert RMSNorm + weighted sum (`mkp_epi_k`, reads the fp32 rows, one bf16 rounding). KS (K split): M<=1: 16, <=16: 8, <=32: 4, else 2 for BOTH GEMMs.
Grids are sized for the worst case of the call's M (shape only); warps beyond the device-side group count exit at once.
Changes sub_mk1 -> sub_mk2: KS 8 for M=8 (verify1 +11%) and the fused epilogue (decode1 GPU time 54.5 -> 40.7 us); overall 1.336x -> 1.377x.

## Per-kernel profile (GPU us per call, job 5ea1d8aa3b604660; the profile pass uses its own routings, so only shares within a column are comparable)
| kernel | decode1 | decode16 | decode32 | verify1 | verify16 | verify32 |
|---|---|---|---|---|---|---|
| mkg_k (GEMM1 + GEMM2) | 29.1 | 124.6 | 233.2 | 123.9 | 290.9 | 457.8 |
| mkp_epi (scatter+norm+sum) | 4.3 | 5.4 | 6.0 | 5.6 | 6.8 | 7.4 |
| mkp_gather (+zero) | 2.8 | 3.1 | 3.2 | 2.8 | 5.0 | 8.7 |
| mkp_act | 2.8 | 3.7 | 4.0 | 3.8 | 4.9 | 6.6 |
| mkp_route | 1.8 | 2.0 | 1.9 | 1.7 | 2.1 | 3.1 |
| total | 40.7 | 138.9 | 248.3 | 137.9 | 309.7 | 483.6 |
GEMMs are 71% (decode1) to 95% (verify32) of the GPU time; glue is 2-3 us per kernel and only matters at decode1.

## Checks (4 of the 8 allowed; all logged in EXPERIMENTS.md)
1. sub_mk1.py, job a0b20c3c199d4c67: 5/5 pass, 1.336x, verify16 1.478x, verify32 1.491x.
2. measure_mkp_v1.py, job 788e51010c7f4913: whole call in a CUDA graph (graph safety confirmed), KS sweep per shape: only M=8 wanted a change (KS8, -12% GEMM time).
3. sub_mk2.py, job 5ea1d8aa3b604660: 5/5 pass, 1.377x.
4. measure_mkp_v2.py, job dc806c9b2b9643eb: KS1 x KS2 x ring-depth sweep: current table within 1-3% of the best everywhere, ring depth 3 never wins. Nothing to adopt.
Stop rule: check 4 moved no case >= 3% (and nothing was adoptable).

## Rules checklist for sub_mk2.py (as a serving candidate)
- Graph safe: yes, captured and replayed in CUDA graphs for all six shapes (job 788e51010c7f4913); no host syncs / .item() / tensor-dependent control flow in kernel_fn.
- No caching of outputs or inputs across calls: buffers are scratch, rewritten (gathered/zeroed) every call.
- prepare() only transforms weights and allocates fixed buffers: it builds v1's state, views the Marlin tensors, takes gs[:,0] as fp32; no study code in sub_mk2.py.
- Precision: NVFP4 weights stay NVFP4, bf16 activations, fp32 accumulation (split-K by fp32 atomics), intermediate and output bf16.
- 8 routed experts per token, every norm kept, every key attended (n/a).
- Not submitted; no READY_TO_SUBMIT.md (it does not beat the 1.480x board file).

## What is left
- decode32 (1.083x) is the weak load: ring-only reads 3.4 TB/s at this shape and the full kernel is dequant/mma issue-bound; only a new GEMM design helps (see muse_moe_w4a16/MOE_ANALYSIS.md).
- Serving integration: prepare() here still runs v1's whole preparation (its own repacked weight copies, small-M / large-M extensions, self-checks). In the serving track only the
  Marlin tensors exist, so the MK path needs a prepare() that skips all of that (it only needs state["banks"], norm_w and the extension build). Not done, not tested.
- Phase-2 specifics not covered: other EP layouts (the path requires local_start 0 and 32 local experts, as v1's own paths do), M > 256, DFlash-specific shapes beyond the 8-token verify blocks.
- Small glue ideas (each 1-3% at most, untested): PDL between act and GEMM2, partial-sum buffers instead of zero + atomics, indirect A loads instead of the gather copy.
