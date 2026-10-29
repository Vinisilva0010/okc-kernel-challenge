# Night plan (Oct 29). Work through the tasks in order. Budget: muse_moe_w4a16 has ~17 checks until 21:00 Brasilia; spend at most 15.

## Task 1: make the Marlin-layout ring GEMM fast (file series muse_moe_w4a16/measure_mk_v3.py, v4, ...)
Target: >= 2.6 TB/s on verify16 and >= 2.2 on verify32 (GEMM1, the MK study's real distributions), error stays ~2e-3.
Kill line: if three consecutive checks gain < 5 % on verify16, stop Task 1 and write the analysis.
Ideas, roughly in expected value order (one or two per check, always against RING4/RING8 in the same run):
1. Token groups of 32/64 (NT 2/4 m-tiles per item): dequant each weight word once for all m-tiles (A tiles: NT x 512 B per stage);
   this removes verify32's 13x re-dequant of the hot expert. Use NT by group size (template), KS chosen so items >= ~2 waves.
2. Locate the limiter with valid twins on RING4 (results stored so nothing is deleted): NOMMA (dequant kept, mma replaced by an XOR into
   a stored accumulator), NODQ (raw words fed to mma), RINGONLY (copies + lds/ldmatrix + fold). Compare to RING4 at verify16.
3. Cheaper scales: build the duplicated bf16x2 scale pairs directly with __byte_perm from the 8 scale bytes (same bits as dqs, check
   error), removing the __low2/__high2 extraction; precompute nothing input-dependent.
4. Two 64-column tiles per warp (128 columns) sharing the A fragments (halves ldmatrix + A copies per weight byte).
5. Epilogue: with KS = 1 use plain stores (no atomics) and measure the atomic cost; keep atomics only where KS > 1.
6. Occupancy: __launch_bounds__(128, minBlocks) to reach 16-24 warps/SM if regs allow; check smem per CTA vs 228 KB.

## Task 2 (only if Task 1 reaches >= 2.4 TB/s on verify16): a full MoE path on the Marlin tensors in a phase-1 file
New file muse_moe_w4a16/sub_v6.py from sub_v1.py: for M where it is faster than v1, route (local ids, per-expert token lists:
reuse v1's route kernels or a small new one), GEMM1 (ring kernel on bank w13), SiLU(gate) * up -> bf16 intermediate, GEMM2 (ring kernel
on bank w2, K = 4096, N = 4096), then v1's epilogue (post-expert norm + weighted sum). Keep v1's path for every other M.
All of it CUDA-graph safe (fixed-size buffers allocated in prepare, zeroed inside the call). Check it. If the overall speedup beats 1.480x
by >= 1 % with all tests passing, write READY_TO_SUBMIT.md: file, job id, per-case table, and a rules checklist (graph safe, no caching,
precision, prepare only transforms weights, study code removed). Do NOT submit.

## Task 3 (if Task 1 is killed): write MOE_ANALYSIS.md with the twin numbers and what would be needed next. Do not start other problems.

## Always
- Log every check in EXPERIMENTS.md (question, expectation, decision; then job id and numbers). Commit + push after each experiment.
- Before ending (credits or budget), write NIGHT_REPORT.md: what was tried, the numbers, the best file, what to do next.
