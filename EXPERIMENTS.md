# Experiment log

Every judge run, with the hypothesis and the measured result. Times in µs; "verify full" = batch 16 / batch 32.

## Board candidates

| version | job | change | result | verdict |
|---|---|---|---|---|
| sub_v5 | 6821872e91094b4c | infraax sonnet_v5 as starting point | 1.788x; verify full 164.9 / 325.8 | baseline |
| sub_v6 | 895aca84535640fd | P.V as FP8 MMA in Triton (P x 2^8, as FA3) | 1.555x; verify full 245.6 / 483.9 | rejected |
| sub_v7 | a593f19f83944a17 | verify full b>=16 on warp-specialized Gluon kernel (G2d) | 1.827x [1.799, 1.854]; verify full 148.3 / 284.2 | submitted |

## Measurement runs (verify full, us: b16 / b32 / b1)

| file | job | finding |
|---|---|---|
| probe_env | 0ed0a394b6184fd1 | image: python 3.12, torch 2.14.1+cu130, triton 3.8.0 (warp_specialize, Gluon), flashinfer 0.7.0, tilelang 0.1.12, cutlass 4.8.0, H200 |
| measure_ws_v1 | 0debed83e5724584 | Triton tl.range(warp_specialize=True) is silently ignored on this loop (identical SASS counts) |
| measure_occ_v1 | ad46803096454dc8 | maxnreg 168/128 for 3-4 programs/SM: spills (10-162), all slower (best 170 vs 163) |
| measure_gluon_v1 | 295c3017d5d54526 | G1 Gluon (cp.async ring + async wgmma, no WS): correct, 177 / 347 / 30 |
| measure_ws2_v1 | c47e56891ef2477e | G2 warp-specialized: correct, no deadlock, but 240 / 471; deeper rings change nothing -> serial cost |
| measure_ws2_v2 | 6e2409da37de4d18 | page ids prefetched 2 tiles ahead: 240 -> 205; V convert costs 25 us |
| measure_ws2_v3 | d11b94ffbc114a2f | twins: producer-only 123, consumer-only 161 -> consumers are the bottleneck; FP8 P.V in Gluon 287 (rejected) |
| measure_ws2_v4 | (timeout) | consumer holding tile i-1 + producer PRE = NBUF-1 -> deadlock |
| measure_ws2_v5 | 0231e0c08f654640 | PRE = NBUF-2 for overlapped consumers: B128N3_OV 146 / 280 / 22.7, correct |

## Lessons

- Triton FP8 P.V: the FP32 accumulator layout of an FP8 MMA differs from the FP8 A-operand layout; Triton routes P through shared memory every tile. Same in Gluon (convert_layout). FA3 uses in-register byte permutes.
- Time twins (a copy of the kernel with one cost removed) inside one check run answer "which part is the bottleneck" for one check.
- In a warp-specialized ring, a consumer that keeps tile i-1 while waiting for tile i needs the producer to run one tile less ahead, or it deadlocks.

## Other problems (first board entries, 2026-10-28)

| problem | file | job | result |
|---|---|---|---|
| muse_decode_layer | sub_v1.py (infraax sonnet_v5f) | 198ce00e37d742d4 | 1.429x, 13/13 tests |
| muse_moe_w4a16 | sub_v1.py (infraax sonnet_v5) | e65993b3c77c4052 | 1.480x, 5/5 tests |
| muse_bf16_proj | sub_v1.py (infraax director_v2) | d4d1752631d84ec3 | 1.171x, 9/9 tests |

## MoE (muse_moe_w4a16), 2026-10-28

New W4A16 path (prepare-time repack: E2M1 bits pre-placed for rotate + mask -> m16n8k16 A registers; e4m3 scales bit-built as bf16 x 2^96;
mma.sync with FP32 accumulation). Exact vs a torch dequant reference (rel L2 1e-7 alone, 2.5e-3 end to end incl. bf16 intermediate).

| file | job | finding |
|---|---|---|
| measure_moebw_v1 | 934f0cc826cd4939 | contiguous NVFP4 stream 4.2 TB/s; stream + dequant (shift/mask) 3.1 TB/s |
| measure_moebw_v2 | bcaadcc9481b4e70 | rotate + mask + mma.sync without scales 4.5 TB/s; with smem LUT scales 3.3 TB/s |
| measure_moebw_v3 | 3b081a86865d43fe | HMUL2 is free (4.6 TB/s with a constant scale); fetching the scales is the cost |
| measure_moebw_v4 | 672e42f839804a20 | scales loaded but unused 4.0 TB/s; uint4 scale loads + bit-built bf16 3.55 TB/s |
| measure_w4g_v1 | 27be432ad71c41b5 | standalone GEMM: rel L2 1e-7..1e-6; 3.0 TB/s at 1-4 tokens/expert, 1.48 at 32 |
| sub_v2..v5 | be2472a3.. a00d9ec7 | integrated path slower than v1 in every scored decode case (rejected) |
| measure_w4v_v1 | f23796ee316646ee | no spills; registers (occupancy) beat prefetch depth: 80 regs 2.8 TB/s, 162 regs 1.45 |
| diag_route_v1 | ca73777bbf8e4953 | real routing (below) |

Real local routing (tokens per active expert, sorted): decode1 2-3 experts x 1; verify1 5-8 experts, max 6; decode16 10-14 experts, max 5-10;
decode32 16-21 experts, max 5-14; verify16 6-24 experts, hot ones 45-97; verify32 23-25 experts, hot ones 110-201.

Lessons:
- The judge's speed-of-light column under-counts the active experts. With the real routing, v1 already streams ~2.2-3.1 TB/s in decode16/32
  (ceiling ~4.2): there was no 2x there. The MoE headroom is in verify16/32 (hot experts with 45-201 tokens: compute reuse) and decode1 (fixed cost).
- Measure the real input distribution before estimating a gain.

## Attention decode-full (decode16 / decode32), 2026-10-28 night

| file | job | finding |
|---|---|---|
| measure_dec_v1 | 327804f0c752413d | (patch did not apply: anchor matched twice; ran infraax's original sweep) no decode config beats C64S3W2 (105 / 192 us) |
| measure_dec_v1 | a7ea45428c234d9d | load-only twin is invalid: without a dot consuming the tiles Triton drops cp.async (cp.async=0) and runs 132 us |
| measure_dec_v2 | 0362b4f2f51f49f8 | D1 (Gluon, 1 warp/program, cp.async ring, mma_v2): correct (2e-4) but slower: best 129.9 us (KL1) vs 105.6; direct smem->dot-operand loads of uint8 (KL0) 179-186 us |

Lessons:
- Direct shared-memory loads of 8-bit tiles into mma_v2 operand layouts are very slow in Gluon; a blocked load + convert_layout is better but still pays an extra smem round trip per tile.
- One warp per program cannot hide the per-tile work (4 tile conversions + 2 MMAs + softmax) at ~6 programs/SM. A faster decode would need the tiles converted once into a fp16 staging buffer (producer stage). Estimated ceiling ~15-20% on decode16/32 only; deferred in favor of the MoE verify16/32 path (1.8x / 2.5x measured headroom).
- Command hygiene: chain the patch and the check with && so a failed anchor stops the run.

## MoE large M: prior art from infraax (read 2026-10-28 night, no runs spent)

v1 profile (job ca73777bbf8e4953): verify32 GEMM1 263 us (1.65 TB/s on 23 experts), GEMM2 190 us (1.14 TB/s); verify16 202 / 144 us;
decode32 147 / 113 us. v1 = infraax sonnet_v5: mma.sync consumers, register-prefetched weights, 32-token chunks, 3 CTAs/SM.

infraax findings (design/breakdown c30b, c30d, c31; sonnet_v6, sonnet_v7):
- WSTREAM (pure weight stream, same bytes and addresses, 16 independent 16-B loads per lane): 4.0 TB/s -> layout and DRAM locality are not the cap.
- v5 is latency-bound: ~4.6 KB in flight per warp, ~1.45 us loaded round trip, ~1/3 of each item spent outside the stream (staging, fixup, epilogue).
- v6 (deeper register / per-warp cp.async ring): no win (limit moved to the issue chain).
- v7 (warp-specialised, 1 bulk-copy producer + 12 mma.sync consumers, mbarrier ring, 1 CTA/SM): slower than v5 in all three builds.
  C31 bisect: the ring alone (bulk copy -> mbarrier -> lds -> release, no unpack, no mma) costs 214 us at bs256 ~= v5. wgmma would save only 7 %.
  Their rule: do not re-propose a ring without a ring-only reading >= 3.3 TB/s.

Our evidence they did not have: the Gluon cp.async ring (no mbarrier, 3 CTAs/SM, cooperative copy + wait_group) streams 3.88 TB/s on random
128-B rows and 4.2 sequential (checks 54bc1970, measure_moebw_v1). Untested path: that ring + dequant once per weight tile into smem + wgmma over
64-201 tokens. Gate before any integration: a single-expert microbenchmark; kill the MoE large-M work if it stays below 2.6 TB/s.

## MoE: Marlin-layout GEMM study (2026-10-29 00:00-00:45)

| file | job | finding |
|---|---|---|
| diag_marlin_v1 | a2e50f7340eb4e10 | tuned Marlin (infraax CONFIG_TABLE) vs stock: decode1 1.60, verify1 1.30, decode16 1.22, decode32 1.02, verify16 1.01, verify32 1.01 |
| measure_mk_v1 | fc739e0c29aa4f63 | own GEMM reading SGLang's Marlin tensors directly: rel L2 2.0e-3 (layout decoded); register prefetch best 1.65 TB/s (verify16), deeper prefetch slower (regs 233-255, spills) |
| measure_mk_v2 | 9eeae71bac414240 | per-warp cp.async ring, 80 regs: verify16 2.04 TB/s, decode32 2.37, verify32 1.47 (hot expert re-dequantized 13x); skip-flop dequant exact (2e-3) and faster than the 2^126 multiply; ring depth 4 = 8 -> instruction-bound now |

## measure_mk_v3 (2026-10-29 night, check 1 of <=15): limiter twins + token groups NT 2/4 + occupancy
Question: what limits RING4 at verify16 (dequant, mma, or the ring itself), and does grouping 32/64 tokens per item (dequant once per word
for NT m-tiles) lift verify32 (1.47) without hurting the others?
Variants (all GEMM1, same real distributions, same kernel template mk3_k<S,NT,MODE,MB>): RING4, RING8, NOMMA (dq kept, mma -> XOR fold into stored acc),
NODQ (raw words to mma), RINGONLY (copies + lds/ldmatrix + XOR fold), RING4_NT2, RING4_NT4, RING4_MB8 (launch bounds 8 CTAs/SM).
Expected: RING4 ~ v2 (verify16 2.04, verify32 1.46). NOMMA ~ RING4 (mma is cheap) ; NODQ and RINGONLY clearly faster. NT2/NT4: verify32 1.8-2.2 if
regs stay < 128 w/o spills, others about equal.
Decisions: RINGONLY << 2.6 -> ring/lds cost is the limiter -> idea 4 (128 cols/warp, shared A). NODQ ~ RING4 but NOMMA >> -> mma/accumulate is the limiter.
RINGONLY ~ RING4 -> memory side/latency, not instructions: go for occupancy/deeper config. NT2/NT4 >= +15% at verify32 -> adopt NT by group size.
RESULT job 92d20bf704e6423a (official 5/5 pass, v1 path 1.478x; MK numbers TB/s on weight bytes, GEMM1):
regs/local: RING4 110/0, RING8 112/0, NOMMA 88, NODQ 87, RINGONLY 64, NT2 136, NT4 208, MB8 64/16 (spills).
verify16 (us): RING4 226 (2.00) RING8 225 (2.01) NOMMA 190 (2.38) NODQ 176 (2.57) RINGONLY 159 (2.86) NT2 246 (1.84) NT4 334 MB8 251 (1.80)
verify32 (us): RING4 304 (1.43) NOMMA 256 NODQ 235 RINGONLY 217 (2.00) NT2 291 (1.49) NT4 378 (1.15) MB8 333
decode32: RING4 2.31, RINGONLY 3.51, NT2 2.01.   err 2e-3 for all real variants (verify16/32, decode1).
Reading: (1) NT2/NT4 do NOT help: regs 136/208 cut occupancy; verify32 +4% only for NT2, verify16 -8%. Idea 1 as written is dead (hot expert would need dequant->smem staging).
(2) ring-only (copies+lds+fold+atomic epilogue) already costs 159 us at verify16 = 2.86 TB/s: the ring floor is below the 4 TB/s stream. Dequant costs ~50 us, mma ~36 us, additive (poor overlap).
(3) verify32 ring-only 217us vs verify16 159us on similar bytes: the hot expert's 13 groups each re-copy its weights (L2 traffic), not only dequant.
Decision for check 2 (v4): attack overlap + floor: P steps per wait/commit group (G4P2, G3P2, G3P4), KS x2/x4 (wave quantization: ~29 groups*128*KS2 = 7.4K items / 2.1K resident warps = 3.5 waves), NOEPI twin (atomic epilogue cost).

## measure_mk_v4 (check 2): overlap (P steps per wait/commit), KS multiplier (wave quantization), atomic-epilogue twin
Question: does batching P k-steps per wait/commit (all lds up front, dq of step 2 can overlap mma of step 1) lift RING4 verify16 (2.00 TB/s)?
Is wave quantization (3.5 waves at KS2) or the atomic epilogue a visible cost (KS2x/4x, NOEPI twin = full compute + one folded plain store)?
Variants: RING4, R4_KS2x, R4_KS4x, G4P2 (6 steps in flight), G3P2 (4), G3P4 (8), NOEPI. Expected: G*P >= +5% if the lds->dq latency chain is the issue (2.1-2.3);
KS2x +/-5%; NOEPI -5..-15 us vs RING4 (verify16 226 us).
Decisions: any variant >= +5% at verify16 -> adopt and stack; NOEPI gain > 10% -> epilogue redesign (plain stores when KS=1, or smem reduce); all flat -> third flat check would
approach the kill line: the remaining lever is dequant instruction count (idea 4: 128 columns/warp) .
RESULT job 60d631ee23b34c35 (official 5/5, 1.483x v1 path). regs: RING4 103, G4P2 111, G3P2 111, G3P4 121, NOEPI 99 (all local 0).
verify16 TB/s: RING4 1.88 (241us; note the same kernel read 2.00 in job 92d20bf: run-to-run noise ~6%), KS2x 1.87, KS4x 1.83, G4P2 2.05, G3P2 2.07, G3P4 2.12 (214us), NOEPI 1.90.
verify32: RING4 1.34, G4P2 1.46, G3P2 1.47, G3P4 1.51.  decode32: RING4 2.17, G3P4 2.44.  verify1: 1.78 -> 1.88.  decode1 1.61 -> 1.7.
Reading: batching P steps per wait/commit: +13% at verify16 vs same-run RING4 (+6% vs the earlier 2.00), +13% verify32, +12% decode32; larger P better. Epilogue (atomics) costs ~0
(NOEPI == RING4) and KS multiplier is flat -> wave quantization/epilogue are not limiters. Idea 5 (plain stores) dropped.
Next (check 3, v5): sweep P further (G2P4, G4P4, G2P8, G3P8) and G3P4 with __launch_bounds__(128,5), control RING4 in-run. Expect G3P8/G2P8 2.2-2.3 if lds->dq latency is the issue; flat -> issue-bound.
RESULT job 750936fa4fd347e4 (official 5/5, 1.475x v1 path). regs: RING4 103, G3P4 121, G2P4 123, G4P4 120, G2P8 152, G3P8 168, G3P4_MB5 96 (local 0).
verify16 TB/s: RING4 1.87, G3P4 2.11, G2P4 2.20 (206us), G4P4 2.12, G2P8 2.13, G3P8 2.01, G3P4_MB5 1.95.  verify32: 1.34 / 1.50 / 1.56 / 1.51 / 1.51 / 1.41 / 1.39.
decode32: 2.16 / 2.44 / 2.53 / 2.47 / 2.47 / 2.30 / 2.27.  verify1 best G2P4 1.87. err 2e-3.
Reading: plateau at P=4; G2P4 +3.8% over check 2's best (G3P4 2.12) -> counts as a flat check (1 of 3 toward the kill line, vs the best so far). Higher regs (G2P8 152, G3P8 168) and
forced lower regs (MB5, 96) both worse.
Hypothesis for check 4: L2->SM traffic. Each of the 128 warps of a token group copies the SAME 512 B A tile per k-step next to its own 576 B W+S (traffic 1088 vs 576 B/step):
RINGONLY 2.86 TB/s weights ~ 4.7 TB/s L2->SM, near the practical L2 limit. Fix: share the A ring across the 4 warps of a CTA (items of a CTA share g, ks; ntiles%4==0): each warp loads
P/4 of the group's A tiles, one __syncthreads per P steps. Variants (v6, mk6_k<G,P,MODE>): control G2P4 (old kernel, same run), A_G2P4, A_G3P4, A_G4P4, A_G2P8, RINGONLY_A_G2P4 (floor).
Expect A_G2P4 2.5-2.9 at verify16 if traffic-bound (verify32 hot expert too); flat (<2.3) -> L2 traffic is not it, then only dequant instruction count remains.
RESULT job 24627bd85684493f (official 5/5, 1.56x verify16 v1 path). regs: G2P4 123, A_G2P4 109, A_G3P4 115, A_G4P4 105, A_G2P8 128, RONLY_A_G2P4 89 (local 0).
verify16 TB/s: G2P4 2.20 (206us) A_G2P4 2.35 (193us) A_G3P4 2.25 A_G4P4 2.28 A_G2P8 2.28 | RINGONLY_A_G2P4 3.18 (143us).
verify32: 1.55 / 1.68 / 1.60 / 1.63 / 1.63 | RINGONLY 2.52 (173us).  decode32: 2.53 / 2.67 ... RINGONLY 3.42.  verify1: 1.87 / 1.96.  decode1 1.71 / 1.79.  err 2e-3.
Reading: shared A ring +6.8% (verify16), +8% (verify32) -> above the 5% line, kill counter reset. Best so far: A_G2P4 (2.35 / 1.68 / 2.67 / 1.96 / 1.79).
Ring floor now 3.18 TB/s (verify16); dq+mma add ~50 us on top (193 vs 143). verify32: RINGONLY 173 vs 143 at verify16 -> hot expert's 13 re-copies cost ~30 us, plus 13x dequant.
Next (check 5, v7): groups with >16 tokens go to an NT2 or NT4 kernel with the shared A ring (regs permitting), groups <=16 stay on NT1; two launches into the same out (disjoint rows).
Variants: A_G2P4 (control, all NT1), SPLIT2 (chunk 32), SPLIT4 (chunk 64). Expect verify32 1.9-2.2 if the hot expert dominated; verify16 slightly up (45-token expert).
Decision: SPLIT gives >= +8% at verify32 and no loss elsewhere -> adopt; NT regs kill it again (<3%) -> stay NT1 and spend remaining checks on dq issue count.
RESULT job bc266bce4355434c (official 5/5, v1 path). regs: mk7 NT1 123, NT2 176, NT4 248 (local 0).
verify16: A_NT1 (mk7<2,4,1>) 2.20 (206us; mk6 read 2.35 in job 24627bd -> mk7's nm predicates cost regs 109->123), SPLIT2 1.96 (231us), SPLIT4 1.92. verify32: 1.58 / 1.52 / 1.51. Others identical (no group > 16).
Reading: separate NT2/NT4 launch is SLOWER (regs 176/248 -> 2 CTAs/SM, and a serialized second launch). NT grouping is dead for good (checks 1 and 5). mk6 stays the base.
Counter vs best (mk6 A_G2P4 2.35): check 5 flat -> 1 of 3.
Next (check 6, v8): swap operands. Marlin's two n-blocks (2w, 2w+1) of a word are exactly an m16k16 A fragment {blo0, bhi0, blo1, bhi1} (rows g / g+8 = blocks 2w / 2w+1); tokens become the
n8 B operand (b0 = ldmatrix reg 0/1, b1 = reg 2/3 of the existing x4 A load). Groups of <= 8 tokens need 4 mma per step instead of 8; 9..16 tokens 8 (unchanged). Output D[col][token]
-> transposed atomics. Most experts have <= 8 tokens. Variants: control A_G2P4 (mk6), N8_G2P4, N8_G3P4 (mk8_k).
Expect: NOMMA delta was ~36us/226 at verify16; halving mma for ~3/4 of groups -> +4..8%. Error stays 2e-3 (same products, same fp32 accumulate order).
Decision: >= +5% at verify16 -> adopt; correctness off (err>1e-2) -> debug fragment mapping by reading the epilogue; flat -> counter 2 of 3.
RESULT job dce14ea8a1574ec5 (official 5/5, v1 path). regs: A_G2P4 109, N8_G2P4 142, N8_G3P4 144 (local 0).
verify16: A_G2P4 2.33 (194us), N8_G2P4 2.11 (215us), N8_G3P4 2.04. verify32: 1.68 / 1.50 / 1.45. decode32 2.66 / 2.44 / 2.40. verify1 1.96 / 1.81 / 1.89. err 2e-3 for N8 (fragment mapping CORRECT).
Reading: swapped operands are correct but 9% SLOWER (regs +33, 4 atomics per acc with row stride). Flat/negative check -> counter 2 of 3 vs best (A_G2P4 ~2.33-2.35).
Check 7 (v9) is the last bundle on this line before the kill rule applies (needs >= 5% over A_G2P4 at verify16): levers that attack the ring floor (3.18 TB/s vs 4.0 stream):
  PF256: cp.async.cg ... .L2::256B on the weight copies (DRAM/L2 sector efficiency, hint only);  MB5: __launch_bounds__(128,5) on A_G2P4 (5 CTAs/SM);
  W8_P4 / W8_P8: 8 warps per CTA share one A ring (A traffic per warp-step 128 -> 64 B; A copied by the first P warps; one __syncthreads per P steps).
Control: A_G2P4 in the same run. If none >= +5%: Task 1 killed -> MOE_ANALYSIS.md (Task 3). If any >= +5%: continue stacking.
