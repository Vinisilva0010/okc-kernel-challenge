# Night report (Oct 29): Task 1 run and killed, Task 3 analysis written

## Outcome
- okc worked this time (full path /home/vnspo/.okc/bin/okc). 7 checks spent on muse_moe_w4a16 (10 of 20 left today; plan allowed 15).
- Task 1 (Marlin-layout ring GEMM, GEMM1): improved but target missed. Best kernel `mk6_k<2,4>` ("A_G2P4", in measure_mk_v6..v9.py):
  verify16 2.33-2.35 TB/s (target 2.6), verify32 1.68-1.69 (target 2.2), decode32 2.66-2.68, verify1 1.96, decode1 1.79-1.80. Error 2e-3. 109 regs, no spills.
  Starting point (RING4, v2 design): verify16 1.87-2.00, verify32 1.34-1.43.
- Kill line hit: checks 5, 6, 7 (jobs bc266bce, dce14ea8, f6ffa925) all failed to beat the best by 5% on verify16 -> Task 1 stopped.
- Task 2 NOT started: gate (>= 2.4 TB/s on verify16) not reached. No READY_TO_SUBMIT.md. Board file unchanged: sub_v1.py 1.480x. Nothing was submitted.
- Task 3 done: muse_moe_w4a16/MOE_ANALYSIS.md (twin numbers, dead ends, what is needed next).

## What gave gains (same-run comparisons, verify16 TB/s)
1. Batching P = 4 k-steps per cp.async wait/commit group (all smem loads up front): RING4 1.87 -> G2P4 2.20 (job 750936fa; G3P4 +13% in job 60d631ee).
2. One A ring shared by the 4 warps of a CTA (cuts L2->SM bytes per k-step from ~1088 to ~704): 2.20 -> 2.35 (job 24627bd8).

## What did not (all measured)
Token groups NT2/NT4 (regs 136/208/248 -> lower occupancy; split launches slower), operand swap with weights as the A fragment (correct, 9% slower),
KS multiplier, epilogue atomics (cost ~0), L2::256B hint, 5/8 CTAs per SM launch bounds, 8 warps per CTA, deeper P8.

## Limiter findings (job 92d20bf7 twins, verify16 us)
RING4 226; no-mma 190; no-dequant 176; ring-only 159 (2.86 TB/s). After the shared A ring the ring-only floor is 143 us (3.18 TB/s) vs the full kernel 193 us.
So both the ring (below the 4.0 TB/s pure stream) and dequant+mma (~50 us, poorly overlapped) matter; the hot expert in verify32 is re-copied/re-dequantized 13x.

## Files / commits
measure_mk_v3..v9.py (each a full check file; official tests pass 5/5 and the v1 path stays ~1.47-1.48x in all of them), EXPERIMENTS.md (question/expectation/decision/result
for every check), MOE_ANALYSIS.md. All pushed to origin main. Note: the kernel_fn in these files is still v1's; the MK study runs inside prepare() (study prints only, CHECK use).
Do NOT submit any measure_mk_*.py file.

## Suggested next steps
- Per MOE_ANALYSIS.md: only a design change is likely to reach 2.6 (dequant once into smem + wgmma / producer-consumer split), not more tuning of this kernel.
- Whether A_G2P4 as GEMM1 would beat v1 end-to-end was not measured; with the board at 1.480x a GEMM1-only gain is diluted by GEMM2/routing/epilogue.
- 10 MoE checks remain today (reset 21:00 Brasilia); other problems were not touched, per the plan.
