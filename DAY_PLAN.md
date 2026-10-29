# Day plan (Oct 29): full MoE on SGLang's Marlin tensors = the phase-2 MoE candidate

Why: mk6_k<2,4> (A_G2P4, measure_mk_v9.py) matches v1's GEMM1 speed (verify16 ~193 us vs 202, verify32 ~257 vs 263, decode32 ~148 vs 147)
while reading SGLang's Marlin tensors directly. In the serving track the experts live ONLY in Marlin format (a second ~57 GB copy per GPU
does not fit), so v1's own-layout kernels cannot be used there; this kernel can. Stock SGLang Marlin takes 469 us (verify16) and 574 us
(verify32) for both GEMMs. Target of this plan: a full MoE path that beats stock SGLang by >= 1.4x on verify16 and verify32 in the phase-1
judge (muse_moe_w4a16), with all tests passing. Beating v1 on the phase-1 board is NOT required.

Budget: muse_moe_w4a16 has 10 checks until 21:00 Brasilia; use at most 8. Rules: everything in CLAUDE.md still applies.

## File: muse_moe_w4a16/sub_mk1.py (start from sub_v1.py; keep v1's prepare and all its state)
kernel_fn: for every M <= 256 take the new MK path below (no v1 fallback for those M, so the judge's numbers measure the MK path itself);
M > 256 keeps v1's path. No study code in prepare(); the file must be submittable as is (graph safe, no caching, prepare only transforms
weights and allocates fixed-size buffers).

MK path, all on state["banks"][(0, 32)] (w13 / w13_s / w13_gs for GEMM1, w2 / w2_s / w2_gs for GEMM2):
1. Routing on the GPU (one small kernel, graph safe): local ids (expert_map), per-expert counts, an expert-sorted slot list
   (slot = token * 8 + k for local routes), and the group table (expert, first slot row, rows) with groups of the kernel's token-group size.
   The number of groups / items is data dependent: write it to device memory; launch a FIXED grid sized for the worst case and let
   warps with item >= n_items exit (the kernel must read n_groups from device memory, not from a host int).
2. Gather: Xg[row] = hidden_states[slot // 8] in expert-sorted order (bf16, fixed buffer of 256 * 8 rows).
3. GEMM1: mk6_k<2,4> on w13 -> fp32 [rows, 8192] (zeroed inside the call; atomics only where KS > 1).
4. Activation: SiLU(gate) * up in fp32 -> bf16 inter [rows, 4096] (gate = columns 0..4095, up = 4096..8191 of GEMM1's output;
   verify the column order against the torch reference before trusting it).
5. GEMM2: the same kernel on w2 (K = 4096 intermediate, N = 4096 hidden) -> fp32 [rows, 4096].
6. Epilogue: scatter rows back to [M, 8, 4096] (slot order) as bf16 and call v1's epilogue (post-expert norm + weighted sum) exactly as
   v1's _kernel_sglang does, with the same local_ids convention (-1 for non-local slots).
Global scales: out = acc * float(gs_p) (skip-flop, validated). Choose KS per M bucket from a small static table (shape only, never values).

## Check plan
Check 1: sub_mk1.py as above. Expect all 5 tests to pass; per-case speedups vs SGLang roughly: verify16 / verify32 >= 1.4x,
decode16 / decode32 >= 1.2x, decode1 / verify1 >= 1.3x. If a test fails: fix only that part, re-check.
Checks 2+: profile the slowest stage (GEMM2, routing, gather, epilogue) from profiles/*.txt and fix the largest one per check.
Stop when two checks in a row move no case by >= 3 %.

## Output
- EXPERIMENTS.md entries as before; MK_PATH_REPORT.md with the per-case table vs SGLang and vs v1, the per-kernel profile of
  verify16 / verify32, and what is left. Commit + push after each check. Do NOT submit anything.
