# MOE_PORT_PLAN: what to take from the public MoE kernels into the Marlin-layout serving path (Oct 30)

Inputs: PUBLIC_REFS.md "Step 4" and "Step 4b", MOE_V3_PLAN.md (its derived numbers are marked (d), +-15%), the judge profiles in muse_moe_w4a16/profiles/ (measure_mk5_num.3a3931cbb62f4cfe is the shipped v2).
Constraint set: no second weight copy (SGLang's prefill/chunks > 256 tokens still run Marlin on the same tensors, so the layout cannot be changed in place), v2 numerics (Marlin write-out rounding, deterministic split-K), CUDA-graph safe.

## Where v2 stands (judge, job 3a3931cb) vs the public kernels
| load | v2 us | public us (vjeux/Madhavan) | v2 TB/s on expert bytes (d) | public TB/s (d) |
|---|---|---|---|---|
| decode 1 | 49.0 | 25.2 | 1.8 | ~3 |
| decode 32 | 187.8 | 126-128 | 2.3-2.7 | ~3.6 |
| verify 1 | 114.3 | 63-67 | 2.0 | ~3.5 |
| verify 16 | 310.4 | 183-186 | 2.3 | ~3.5 |
| verify 32 | 418.9 | 217-222 | 1.6 | ~3.2 |
mkg_k is 87-94% of the profile in every load except decode 1 (70%). Two separate gaps:
(G1) loads with no duplicate passes (decode 1/16/32, verify 1): v2 runs at ~2.3-2.7 TB/s, the public kernel ~3.5: a per-byte instruction/pipeline gap of ~1.4-1.5x;
(G2) verify 16/32: passes > experts (39 passes for 23 experts at verify 32): an extra 1.2-1.6x on top of G1.

## Ranking (what can be applied to the Marlin tensors as they are)
| rank | technique (public source) | applicable without a 2nd copy? | gap addressed | expected gain | numerics | risk / cost |
|---|---|---|---|---|---|---|
| 1 | Dequantize each weight tile once for all token groups of a hot expert (public: fat jobs of <= 64 tokens on RS-wgmma). Our form: mk_moe_v3 `mkh_k`, one producer warp dequantizes into a smem fragment ring, 4 consumer warps (one 16-token group each) run v2's mma sequence | yes (reads Marlin tiles, writes nothing back) | G2 | verify 32 pipeline 420 -> 320-360 us (-14..-24%), verify 16 -4..-9%, no-hot loads 0 (d, MOE_V3_PLAN) | bit-identical to v2 at equal KS (same B bits, A tiles, k order, planes) | written Oct 29, never compiled or run; producer warp may cap at ~2.4 TB/s; kill lines in MOE_V3_PLAN |
| 2 | Balance the dequant between the INT/ALU pipe and the FMA pipe (public DQ7 counts 7 ALU + 2 IMAD per 8 weights and states the shifts go to IMAD; ours is ~15 INT ops per 8 weights: `q<<8`, `q<<4`, 2x(and, shr, lop3)) | yes (same bits out, only the instruction mix changes: shifts as IMAD multiplies, the >>6 as mul.hi) | G1 | unknown until SASS is counted; if the ALU pipe is the limit, up to ~1.2x on the GEMM at all loads | bit-identical (identical bf16 operands) | needs a SASS/pipe histogram first (no nvcc here: do it in a judge study with cuobjdump) |
| 3 | One launch instead of seven; first weight TMA before routing finishes (public) | partly: drop `mkp_gather` (GEMM A loads read x rows through slot_of_row), PDL between kernels | glue is 12-26 us of 55-510 (MOE_V3_PLAN s1.4) | -3..-6 us per call (decode 1 -5..-10%, verify 1-2%) | unchanged | low risk, small |
| 4 | RS-wgmma with weights as the A operand (m64nNk16), stream-K, GEMM1->GEMM2 hand-off in registers (public WGS build) | the A fragment is already in the Marlin word: one word = rows g and g+8 x k {2t,2t+1,2t+8,2t+9} of one m16k16 tile (CLAUDE.md layout), so a warpgroup's 4 warps can take 4 Marlin n-block pairs of one k-tile (smem reads of word i at stride 16 B: 4-way bank conflict on LDS.32, tolerable) | G1 and G2 | the public numbers (~3.2-3.6 TB/s) are the ceiling | the accumulation order differs from v2's mma.sync k-chains (wgmma k16 steps are the same k order, but the fp32 internal accumulation inside one wgmma is not mma.sync's): bit-identity to v2 is NOT guaranteed, parity vs stock would have to be re-measured | days of work (tens of lines of wgmma/descriptor asm, producer warps, fixup); not for today |
| 5 | Run the public kernel itself on the Marlin tensors | NO | | | | (a) its codes need the bit-spread layout (DQ7: and-masks on a permuted word), Marlin words have a different nibble order; (b) it needs the fp32 global scale (Marlin keeps a bf16 x 2^119 copy; the fp32 value cannot be recovered exactly); (c) it applies the global scale in fp32 and keeps y in fp32, so it would drop v2's bf16 write-out rounding and move away from stock parity; (d) its M >= 17 build needs a 4th weight copy (WGS) and the tiny builds two more: ~3.6 GB per layer and GPU, impossible next to Marlin. |

Dequant arithmetic in the Marlin layout cannot reach the public 9 ops per 8 weights: their code bits are spread by the repack so a single AND yields each bf16 pair, ours needs AND + SHR + LOP3 per pair (Marlin's own trick set, MOE_ANALYSIS "next"). Only the pipe mix (rank 2) can still move.

## Plan for today (checks: 16 left on muse_moe_w4a16, at most 10 used)
1. Rank 1: run the existing v3 study file (measure_mk6_v3.py: bitwise v3 == v2, per-stage event times, regs/spills probe, graph latency v2 vs v3). Adopt v3 only if bitwise equal at equal KS, 5/5, verify 16 / verify 32 each >= +3% and no load > 2% below v2.
2. Rank 2: add an instruction histogram of the shipped kernel (cuobjdump -sass on the extension, compact `MK ...` prints) and a twin that moves the shifts to IMAD; decide from the histogram first, from timing second.
3. Rank 3 only if ranks 1-2 are done and the per-stage table shows the glue.
Stop rule: two consecutive checks that do not move the target metric end that line.
