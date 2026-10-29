# MoE Marlin-layout ring GEMM: analysis (night of Oct 29, Task 1 killed by the 3-flat-checks rule)

All numbers: judge logs in muse_moe_w4a16/profiles/ (job ids below), GEMM1 only (K = 4096, N = 8192), the MK study's real per-expert token
distributions, TB/s on weight+scale bytes (18.87 MB per expert, derived from the tensor shapes), median of 12 timed launches after a 256 MB flush.
Run-to-run noise on the same kernel is about 6% (RING4 verify16 read 2.00 in job 92d20bf7 and 1.88 / 1.87 in jobs 60d631ee / 750936fa), so
only same-run comparisons are used for decisions. Error vs a torch dequant reference stays 2e-3 for every real (non-twin) variant.

## Best kernel
`mk6_k<G=2, P=4>` in `measure_mk_v6.py` ... `measure_mk_v9.py` ("A_G2P4"): per-warp cp.async ring for the Marlin weight tile (512 B) + scales (64 B),
ONE A ring (16 tokens x 16 k, 512 B per k-step) shared by the 4 warps of a CTA, commit groups of P = 4 k-steps (one wait + one `__syncthreads` per 4 steps),
skip-flop dequant, mma.sync m16n8k16, atomic epilogue. 109 regs, 0 spills.

| case (GEMM1) | RING4 (v2 design) | A_G2P4 | gain |
|---|---|---|---|
| verify16 | 1.87-2.00 | 2.33-2.35 (193-194 us) | +17..25% (same-run: 1.87 -> 2.20 with batching, -> 2.35 with the shared A ring) |
| verify32 | 1.34-1.43 | 1.68-1.69 | +18..25% |
| decode32 | 2.16-2.31 | 2.66-2.68 | +16..23% |
| verify1 | 1.78-1.84 | 1.96 | +7..10% |
| decode1 | 1.61-1.71 | 1.79-1.80 | +5..11% |

Targets were 2.6 (verify16) and 2.2 (verify32): NOT reached. For reference v1's profile shows verify16 GEMM1 202 us; A_G2P4 reads 193 us on the study's
24-expert distribution (not a like-for-like with the v1 profile, so no claim of a win).

## What each check showed
| job | question | result |
|---|---|---|
| 92d20bf704e6423a | limiter twins on RING4, token groups NT2/NT4, 8 CTAs/SM | verify16 us: RING4 226, NOMMA 190, NODQ 176, RINGONLY 159 (2.86 TB/s). NT2 246, NT4 334 (regs 136/208), MB8 251 (spills). |
| 60d631ee23b34c35 | P k-steps per wait/commit; KS x2/x4; atomic epilogue | G3P4 +13% over same-run RING4; NOEPI == RING4 (atomics ~0); KS multiplier flat. |
| 750936fa4fd347e4 | P sweep | G2P4 best (2.20 vs RING4 1.87); G2P8/G3P8 higher regs, worse; MB5 worse. |
| 24627bd85684493f | A ring shared per CTA | +6.8% (verify16), +8% (verify32); RINGONLY of it 3.18 TB/s (143 us). |
| bc266bce4355434c | groups > 16 tokens on an NT2/NT4 kernel (2 launches) | SLOWER (verify32 1.52 / 1.51 vs 1.58; verify16 1.96 / 1.92 vs 2.20): regs 176 / 248 -> 2 CTAs/SM, serialized launches. |
| dce14ea8a1574ec5 | operands swapped (Marlin words = m16k16 A fragment, tokens = n8 B, 4 mma instead of 8 for <= 8 tokens) | numerically correct (2e-3) but 9% slower (142 regs, transposed atomics). |
| f6ffa925f34a4c82 | L2::256B prefetch hint, launch bound 5, 8 warps/CTA | PF256 == control; MB5 2.25, W8_P4 2.28, W8_P8 2.23 vs control 2.34. |

## Where the time goes (verify16, us, from the twins; jobs 92d20bf7 and 24627bd8)
- RING4 226 -> remove mma (NOMMA) 190 (-36) -> remove dequant+scales (NODQ) 176 (-50) -> ring only (copies + lds/ldmatrix + fold + epilogue) 159.
- With the shared A ring the ring-only floor is 143 us (3.18 TB/s) while the full kernel is 193 us: dequant + mma add ~50 us, only partly overlapped.
- The pure weight stream of infraax reached 4.0 TB/s (113 us for these bytes) -> the cp.async ring itself loses ~30 us to the stream, and dq + mma ~50 more.
- Instruction estimate (no SASS available locally, so this is arithmetic, not a measurement): ~100-120 warp instructions per 512 B tile (dequant ~85); at 4 schedulers x 1.755 GHz
  that is ~95-115 us of pure issue slots for 885K tiles, i.e. the kernel is close to issue-bound once the ring is fed, and 193 us means ~50% issue efficiency.
- Dead ends (measured): more registers per thread (NT2/NT4, P8) always lost occupancy; fewer registers (MB5, MB8) lost more through scheduling/spills;
  atomics in the epilogue cost ~0; KS multiplier flat; L2 prefetch hint no effect.
- verify32 stays far from 2.2 (best 1.69): the 201-token expert is re-copied and re-dequantized by 13 token groups (ring-only costs 173 us vs 143 us at verify16 on similar bytes).
  The NT-grouping fix did not work because accumulators in registers cost occupancy.

## What would be needed next (not tried; each needs a design, not a tweak)
1. Dequantize each weight tile ONCE into shared memory (bf16, swizzled) and feed wgmma (or ldmatrix + mma) for all token groups of the expert: removes the 13x re-dequant of the hot expert
   and decouples the dq instruction stream from the mma warps. Needs a producer/consumer split (dq warps vs mma warps) with mbarriers; infraax's v7 warp-specialised build
   without dequant-in-smem was slower than v5 (ring-only 214 us at bs256), so only worth trying if the ring-only reading stays >= 3.3 TB/s on this layout (ours: 3.18 for cp.async per-warp rings).
2. Cut dequant instructions below ~85 per 512 B: e.g. FP4 -> FP8(e4m3) conversion with the scale folded in the exponent field is not exact for e4m3 scales (3-bit mantissa), so a real
   saving needs a different trick (a packed prmt LUT was estimated at 9 ops per 4 values vs ~6 now, so no).
3. A split-by-expert-size launch with persistent CTAs (hot expert on its own big-tile kernel, small experts on the ring kernel) running concurrently on one stream via two CTA classes
   in one grid (my two-launch version serialized and lost).
4. Integration value is unmeasured: GEMM1 is only part of the MoE call (GEMM2, routing, epilogue are separate). v1's profile gives GEMM2 144 us at verify16 and 190 us at verify32,
   so a GEMM1 gain alone is diluted; whether A_G2P4 clears the +1% bar over the 1.480x board file was not tested (Task 2 gate not met, sub_v6.py not started).
