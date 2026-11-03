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
RESULT job f6ffa925f34a4c82 (official 5/5, v1 path). regs: A_G2P4 109, PF256 109, MB5 90, W8_P4 111, W8_P8 128 (local 0).
verify16 TB/s: A_G2P4 2.34 (193us), PF256 2.34, MB5 2.25, W8_P4 2.28, W8_P8 2.23.  verify32: 1.69 / 1.69 / 1.62 / 1.56 / 1.52.  decode32: 2.68 / 2.67 / 2.58 / 2.52 / 2.45.  err 2e-3.
Reading: nothing beats the control. L2 prefetch hint = no effect; fewer regs (90) worse; 8 warps per CTA worse (CTA-wide barrier over more warps, 111-128 regs).
KILL LINE: checks 5, 6, 7 gained < 5% (all <= 0%) on verify16 vs the best A_G2P4. Task 1 stopped. Task 2 gate (>= 2.4 TB/s on verify16) NOT reached (best 2.34-2.35). -> Task 3: MOE_ANALYSIS.md.
Checks spent: 7 of 15 (10 of 20 left on muse_moe_w4a16 today).

## DAY PLAN (Oct 29 day): sub_mk1.py = full MoE on SGLang's Marlin tensors (mk6_k<2,4> for GEMM1 and GEMM2)
Check 1 (sub_mk1.py): Question: do all 5 tests pass with a complete graph-safe MK path (route kernel -> gather+zero -> GEMM1 -> SiLU*up -> GEMM2 -> scatter -> v1 epilogue),
and where does each case stand vs SGLang? Expect: tests pass (rel L2 ~2e-3 + bf16 roundings < 1.5%); speedups vs SGLang roughly verify16/32 1.2-1.5x, decode16/32 1.0-1.3x, decode1/verify1 1.2-1.5x
(GEMM1 ~= v1's speed per MK study; GEMM2 and glue unmeasured). Decisions: test failure -> fix that part only; passes -> profile the largest stage (profiles/*.txt) and attack it next.
RESULT check 1, job a0b20c3c199d4c67: sub_mk1.py 5/5 tests pass, mkp active, overall 1.336x (v1 1.480x). Per case vs SGLang: decode1 1.568, decode16 1.262, decode32 1.085, verify1 1.202,
verify16 1.478 (317.2 us vs 468.9), verify32 1.491 (408.5 vs 608.8). Targets (verify16/32 >= 1.4x) MET; decode16/32 >= 1.2x: decode16 met, decode32 NOT; decode1/verify1 >= 1.3x: decode1 met, verify1 NOT.
Profile (GPU us): mkg_k (GEMM1+GEMM2) 39.8 / 164.1 / 164.9 / 102.9 / 275.8 / 394.2 for decode1/decode16/decode32/verify1/verify16/verify32 (73-94% of the call);
glue per call: gather 2.7-6.9, act 2.8-5.9, scatter 1.3-3.7, route 1.7-2.8, epilogue 3.6-6.0 (decode1: glue+epilogue = 14.6 of 54.5 us).
Reading: GEMMs dominate. KS table (8:16, 16:8, 32:4, else 2) was a guess; verify1 (103 us for ~6 experts) looks far from the stream rate.
Check 2 plan: measure_mkp_v1.py = sub_mk1 + a prepare() study (CHECK only): time the whole _kernel_mkp inside a CUDA graph (also proves graph safety) on routing built from the real
tokens-per-expert distributions of CLAUDE.md, for KS in {1,2,4,8,16} per shape. Question: which KS per M is best? Expect verify1/decode1 to prefer a different KS than 16 (perhaps 8), decode32 KS 4-8.
Decision: adopt the best KS per M bucket in sub_mk2.py (shape-only table) if it gains >= 3% on any case; otherwise attack glue fusion.
RESULT check 2, job 788e51010c7f4913 (measure_mkp_v1; official 5/5 pass; judge numbers ~= sub_mk1). Whole _kernel_mkp captured in a CUDA graph for all 6 shapes x 5 KS: OK (graph safe confirmed).
Study routing = CLAUDE.md distributions (absolute us are higher than the judge's own routings; compare KS within a row). us:
decode1:  KS1 104.4 KS2 85.8 KS4 72.1 KS8 60.0 KS16 59.1    decode16: 191.0 / 162.0 / 142.8 / 142.8 / 154.7    decode32: 292.6 / 250.5 / 239.2 / 240.4 / 263.2
verify1:  KS1 119.3 KS2 105.2 KS4 86.5 KS8 77.5 KS16 87.7   verify16: 326.5 / 311.9 / 314.4 / 331.9 / 410.0   verify32: 437.5 / 419.9 / 423.2 / 468.2 / 593.6
(output vs KS1: 4e-4..6e-4, i.e. only split-K atomic order). Reading: current table (M<=8: 16, <=16: 8, <=32: 4, else 2) is right except M=8 (verify1): KS8 is 12% faster than KS16.
Check 3 plan (sub_mk2.py): (1) KS table M<=1:16, M<=8:8; (2) fused CUDA scatter+epilogue kernel: block per token, reads fp32 GEMM2 rows via row_of_slot (written by the route kernel),
two passes (sumsq of y*w per slot, then sum of y*coef, times w), one bf16 rounding at the store; removes mkp_scatter (1.3-3.7 us), the bf16 round-trip of y and the Triton/SGLang epilogue (3.6-6.0 us).
Expect: verify1 +6-8% (GEMM 103 -> ~92 us), decode1 -4 us (52 -> ~48), others -3..-6 us (1-2%). Decision: tests pass and no case regresses -> keep; if the new epilogue fails a test -> fix numerics only.
RESULT check 3, job 5ea1d8aa3b604660: sub_mk2.py 5/5 pass, overall 1.377x (sub_mk1 1.336x). decode1 1.663 (47.5 us), decode16 1.287, decode32 1.083, verify1 1.334 (107.3), verify16 1.482 (320.9), verify32 1.484 (417.4).
Kernel profile (GPU us, decode1/dec16/dec32/ver1/ver16/ver32): mkg_k 29.1/124.6/233.2/123.9/290.9/457.8 (71-95%); mkp_epi 4.3/5.4/6.0/5.6/6.8/7.4; gather 2.8/3.1/3.2/2.8/5.0/8.7; act 2.8/3.7/4.0/3.8/4.9/6.6; route 1.8-3.1.
(The profile pass uses its own routings: its decode32 GEMM sum is 233 us vs 165 us in check 1, so compare profiles only within one job.)
Reading: KS fix + fused epilogue = +3% overall (verify1 +11%, decode1 +6%). Glue is now 3-25% (decode1 only); GEMMs 90%+ elsewhere. Rule "two checks in a row < 3%": not triggered (check 3 moved verify1 +11%).
Check 4 plan (sub_mk3 base + study measure_mkp_v2.py): GEMM2 has 64 n-tiles per group vs GEMM1's 128, so its best KS may be larger; ring depth G3P4 (more bytes in flight) was never tried inside the full pipeline.
Sweep per shape: KS1 in {2,4,8,16} x KS2 in {2,4,8,16} x ring {G2, G3} (same ring for both GEMMs), graph-timed. Expect: KS2 = 2 x KS1 wins by 2-5% on some shapes; G3 +-2%.
Decision: adopt a per-M table (KS1, KS2, ring) if it beats the current table by >= 3% on any shape without losing > 1% elsewhere (then final official check = check 5).
RESULT check 4, job dc806c9b2b9643eb (measure_mkp_v2 = sub_mk2 + KS1 x KS2 x ring sweep, graph-timed, study routings; official 5/5, ~sub_mk2 numbers). Best vs current table (us):
decode1 8/16/G2 55.1 vs 16/16 56.8 (-3%); decode16 4/8/G2 139.0 vs 8/8 140.8 (-1.3%); decode32 4/8 238.5 vs 4/4 239.7 (-0.5%); verify1 8/8 76.7 (= current); verify16 2/2 310.0 (= current);
verify32 2/4 412.5 vs 2/2 417.2 (-1.1%). Ring G3 never wins (best G3 entries are 0.5-3% behind their G2 twins). max |diff| between configs 2e-4..4e-4.
Reading: the table is at its optimum (<= 3%, within run noise ~1-2%); GEMM2 does not want a different KS from GEMM1; deeper ring does not help.
STOP: check 3 was the last one that moved a case by >= 3%; check 4 found nothing to adopt. Budget: 4 of the 8 allowed checks used (6 of 20 left today). Final candidate = sub_mk2.py (job 5ea1d8aa3b604660).
sub_mk3.py (ring/KS-per-GEMM plumbing, never run as an official check) removed to avoid confusion; measure_mkp_v2.py keeps the study.

## DAY_PLAN2 (serving patch v0), Step 1-2: mk_moe/mk_moe.py -> sub_mk3 (reduced prepare, MKSELF)
Check 1 of <=4 (measure_mk3_self.py = sub_mk3 + CHECK-only self-check in prepare): Question: does the single-source MkMoE module (kernels copied from sub_mk2, epilogue now with a weight_first flag,
routing for any local_offset, fp32 global scales bound once per layer) reproduce sub_mk2's 5/5 pass and per-case speedups with a prepare() reduced to Marlin repack + buffers?
Expect: 5/5; per case within ~3% of sub_mk2 (decode1 1.66, decode16 1.29, decode32 1.08, verify1 1.33, verify16 1.48, verify32 1.48; overall ~1.377). MKSELF: offset0==offset96 bitwise True for M=33,128,256
(KS 2: two partial sums, exact), maxabs ~0..1 bf16 ulp for M<=32 (KS>=4: atomic order), rel_vs_marlin ~3e-3, epilogue torch twin rel ~1e-3 or better for BOTH norm forms.
Decisions: pass -> generate the SGLang module (done by the tool) and check the clean sub_mk3.py once (check 2); MKSELF off96 != off0 beyond atomic order -> routing bug, fix before Step 3;
wf=False epilogue error > 1e-2 -> fix kernel.
RESULT check 1, job 2ceb32ffa89d4b4d (measure_mk3_self.py): 5/5 pass, okc-diag mk active=True. Per case: decode1 1.698 (45.7 us), decode16 1.288, decode32 1.094, verify1 1.351, verify16 1.501 (312.6), verify32 1.453 (430.3 vs 625.2).
vs sub_mk2 (1.663 / 1.287 / 1.083 / 1.334 / 1.482 / 1.484): within ~2% except verify32 -2.1% (SGLang's own time differs between jobs; the new module's work is identical). Reduced prepare works.
MKSELF (random 8-of-256 routing, offset 0 vs ids+96 mod 256 with offset 96): M=1 trivially no local routes; M=33/128/256 bitwise equal; M=8 maxabs 2.4e-4, M=32 3.9e-3 (bf16 ulp level; KS 8 / 4
split-K atomic order, not a routing difference); rel vs stock Marlin path 5.3-5.5e-3 (< 1.5% limit; stock itself rounds y to bf16); epilogue vs torch twin from the kernel's own rows: 1.7e-3 for
weight_first True AND False (= bf16 output rounding). Decision: proceed to Step 3 (SGLang module is generated by the tool; patch below).
Check 2 of <=4 (sub_mk3.py, clean, no study/selfcheck code): Question: does the generated submittable file reproduce check 1 (5/5, per case within noise)? Expect: 5/5, overall 1.37-1.40x, each case within ~3% of job 2ceb32ff
(decode1 1.70, decode16 1.29, decode32 1.09, verify1 1.35, verify16 1.50, verify32 1.45-1.48). Decision: matches -> this is the file that mirrors the serving module; stop checking (nothing else changes before Nov 5);
differs by > 3% -> look for a difference between measure_mk3_self and sub_mk3 (only the prepare() self-check block).
RESULT check 2, job 3bf003f0784c4e97 (sub_mk3.py clean): 5/5 pass, overall 1.393x (CI 1.372-1.414), mk active. decode1 1.698 (45.6 us), decode16 1.293, decode32 1.096, verify1 1.371 (103.9), verify16 1.492 (307.6), verify32 1.484 (420.0 vs 623.4).
Identical to job 2ceb32ff within noise (<= 2%). Decision: stop checking (2 of the 4 allowed used; 4 of 20 daily checks left); nothing changes before the Nov 5 serving window except what a serving check reveals.

## DAY_PLAN3 (numerics parity + determinism for the MK MoE = serving patch v1)
Changes in mk_moe/mk_moe.py (v0 frozen as mk_moe_v0.py): (1) GEMM1 and GEMM2 outputs are the bf16 rounding of the fp32 sum of the K-slice partials (stock Marlin fp32-reduce then bf16 store);
silu_and_mul as stock builds it for sm90 (bf16 gate/up in, fp32 __fdividef(g, 1 + __expf(-g)) * u, one bf16 rounding); epilogue per slot in fp32 in ascending slot order like stock's fused kernel
(weight_first: ((x*w)*rsqrt(mean((x*w)^2)+eps))*s, else ((x*rsqrt(mean(x^2)+eps))*w)*s). (2) Split-K without atomics and without zeroing: slice s stores its fp32 partial in its own plane [KS, 8M, N]
(one shared buffer, GEMM2 reuses GEMM1's after the activation), consumers add planes in order 0..KS-1.
Check 1 of <=3 (measure_mk4_num.py = sub_mk4 + CHECK-only MKNUM study): Question: how close is the new path to stock Marlin, and is it deterministic? Real-like routing at M in (1, 8, 16, 32, 128, 256).
Expect: new_vs_new bitwise equal at every M; new_vs_stock rel L2 well below old (~5e-3), likely 1e-3..3e-3 (bf16 ulp scale since both round to bf16); old_vs_old not bitwise for M <= 32; stock_vs_stock = reference for 'agrees with itself'.
Speed within ~3% of sub_mk3 (overall 1.393x): the GEMM epilogue now stores instead of atomics and the zeroing is gone, so it may be slightly faster; consumers re-read KS planes (M=1: 16).
Decisions: new_vs_new bitwise and new_vs_stock <= old and within about 2x stock_vs_stock -> check 2 (clean sub_mk4); not deterministic -> look for a race (e.g. partial planes overlapping, KS plane stride);
new_vs_stock not better than old -> the rounding points differ from stock: inspect which stage (compare intermediates) before spending check 2.
RESULT check 1, job e029437b9fb44025 (measure_mk4_num.py; official 5/5 pass). MKNUM (stock = v1 _kernel_sglang with v1's tuned config, NOT the serving config):
new_vs_new bitwise equal at all M (old_vs_old not for M=8/16/32). new_vs_stock rel 6.8e-3 (M1) 5.7e-3 (8,16,32,128,256) vs old_vs_stock 6.6e-3 / 5.5e-3: NO improvement; diff_frac 0.56-0.75 of output elements differ. stock_vs_stock bitwise
equal except M=1 (rel 4.0e-3, v1 uses atomic reduce there) and M=256 (6.7e-5). scratch: new 160 MB, old 128 MB.
SPEED REGRESSION at small M in this job: decode1 0.742x (104.6 us), decode16 0.982x, decode32 0.974x, verify1 0.966x; verify16 1.432x, verify32 1.481x (sub_mk3: 1.70 / 1.29 / 1.09 / 1.37 / 1.49 / 1.48).
Reading: (1) determinism achieved. (2) matching stock's rounding points did not reduce the difference to stock, so the 5.7e-3 is not rounding: some systematic difference. Unknown which stage: need stage-wise compare
(GEMM1 bf16 output, silu*mul output, GEMM2 bf16 output) against the SERVING stock config (atomic False, fp32 reduce True, serving block-size rule, align with 256 experts) AND against a true fp32 dequant reference
(who is closer to the truth?). (3) Small-M slowdown = the consumers (act, epilogue) add KS planes (KS 16 at M=1) with a serial dependent-free loop that is not unrolled: latency bound, 2 passes. Fix: batch the plane loads
(same summation order).
Check 2 of <=3 (measure_mk4_num.py again, new study): Q: which stage differs from serving stock and who is nearer the fp32 truth; does batched plane loading restore small-M speed?
Expect: speed back to within ~3% of sub_mk3 for M<=32. Stage table: if c1 (GEMM1 bf16) differs from stock by ~5e-3 while both are ~equally far from truth -> Marlin's own numerics (e.g. fp16-style or per-group
rescale ordering) differ; if ours is much closer to truth than stock -> stock has the error and exact parity needs mimicking it.
Decision: stage that diverges first tells what to mimic; if truth-error of stock is ~5e-3 and ours ~2e-3, parity with stock at < 1e-3 is out of reach without reproducing Marlin's exact arithmetic -> report that (and the risk) instead of chasing it.
RESULT check 2, job eb13b2af5c23469d (measure_mk4_num.py, stage study vs SERVING stock = no atomics, fp32 reduce, global-id align): official 5/5. Speed still bad at small M: decode1 0.757x (102.4 us), decode16 1.025, decode32 0.977,
verify1 1.035, verify16 1.435, verify32 1.471. Profile (GPU us) decode1: mkp_epi 41.0, mkg 27.9, mkp_act 12.5; decode16: mkg 111.2, epi 30.0, act 13.2 -> the CONSUMERS are the cost (one block summing 16 planes, chained latencies).
MKNUM vs serving stock: serving stock is BITWISE deterministic (stock_vs_stock eq=True) for M<=128 (M=256: rel 1.3e-4); new_vs_new bitwise at every M; new_vs_servingstock rel 5.7e-3..6.1e-3, 56-73% of output elements differ.
STAGES (M=8 and M=128 identical): ours-vs-stock GEMM1 bf16 output rel 2.8e-3 (73% elements equal), after silu*mul 4.6e-3 (62% equal), GEMM2 bf16 output 5.4e-3 (30% equal).
Unit-norm error vs fp32 truth (GEMM1, fp32 global scale): stock 2.4e-3, ours 1.7e-3 -> ours is nearer the truth than stock; the difference to stock is stock's own arithmetic.
Reading: rounding points were not the cause; the stages differ already at GEMM1 (27% of bf16 elements one ulp off = fp32 values differ ~5e-4). Check 3 (last allowed by the plan): parallel plane reduction
(act grid (rows,4) with all plane loads in flight; a small reduce kernel writing dense bf16 y; epilogue reads bf16 y) to restore small-M speed, and the truth study repeated with Marlin's bf16-rounded global scale
(separates the common scale rounding from real arithmetic differences) incl. our fp32 GEMM1 sum before rounding.
Expect: speed back within ~3% of sub_mk3 (decode1 ~1.7, decode16 ~1.29, decode32 ~1.09, verify1 ~1.37, verify16/32 ~1.49); ours_fp32 vs truth(bf16 gs) ~1e-6..1e-4 if our GEMM is exact; stock vs truth(bf16 gs) tells stock's extra error.
RESULT check 3, job 53b3070582c447a5 (measure_mk4_num.py with parallel plane reduction; official 5/5): overall 1.379x. decode1 1.624 (47.1 us), decode16 1.276, decode32 1.095, verify1 1.325, verify16 1.516 (301.6), verify32 1.508 (408.3).
Speed restored (sub_mk3: 1.698 / 1.293 / 1.096 / 1.371 / 1.492 / 1.484; small-M within noise, verify16/32 +1.6% / +1.6%). Determinism: new_vs_new bitwise at all M; serving stock bitwise self-equal for M<=128 (M=256 rel 6.2e-5). scratch 176 MB.
new_vs_servingstock rel 5.65e-3..5.79e-3 (63-70% elements differ) - unchanged. STAGE (M=8,128): GEMM1 vs stock 2.8e-3; truth with Marlin's bf16 global scale: stock 2.3e-3, ours(bf16) 1.7e-3, ours FP32 sum 1.7e-7 / 4.9e-7.
=> our GEMM is exact to fp32 roundoff; stock has an extra ~1.6e-3 error. Found its source by reading marlin_template.h (write lambda, lines ~1355-1368): res = float2num(c0,c1) [bf16 round of the fp32 accumulator] THEN
res = __hmul2(res, global_scale) [bf16 multiply, second rounding]. Stock output = bf16(bf16(acc_raw) * gs_bf16). We applied gs in fp32 inside the GEMM and rounded once.
Fix (candidate v2, NOT yet checked - plan budget of 3 checks is spent): GEMM stores raw acc (no gs); consumers do rb(rb(sum) * gs[expert of row]) (act for GEMM1, reduce for GEMM2); needs exp_of_row from the route kernel.
Expect: GEMM1 bf16 output equal to stock for ~all elements (differences only from fp32 summation order, rare 1-ulp flips), final output rel vs stock ~1e-4 or less, speed unchanged. First thing to run next time.

## v2 check (last muse_moe_w4a16 check of Oct 29): measure_mk5_num.py = sub_mk5 (mk_moe_v2.py: stock Marlin write-out bf16(bf16(acc)*gs_bf16)) + CHECK-only MKNUM study
Question: does reproducing Marlin's write-out rounding bring our output to stock's? Expect: GEMM1 bf16 equal to stock for nearly all elements (rare 1-ulp flips from fp32 summation order), final output rel L2 vs stock ~1e-4 or lower,
OURS vs OURS bitwise, 5/5 tests, speed within ~2% of v1 (1.379x). Decision: holds -> v2 replaces v1 in mk_moe.py, generators rerun, patch regenerated + rule check; otherwise v1 stays and the numbers are written down.
RESULT v2 check, job 3a3931cbb62f4cfe (measure_mk5_num.py): 5/5 pass, mk active, overall 1.351x (v1 1.379x). decode1 1.583 (49.0 us), decode16 1.249, decode32 1.081, verify1 1.313, verify16 1.468 (310.4), verify32 1.473 (418.9).
MKNUM vs serving stock: M=1 5.3e-5, M=8 2.9e-4, M=16 1.0e-3, M=32 1.6e-4, M=128 2.5e-4, M=256 3.2e-4 rel L2 (v1: 5.7e-3 everywhere); elements differing 0.2-2.2% (v1: 56-70%). OURS vs OURS bitwise at all M; stock self-equal except M=256 (5.6e-5).
STAGES (M=8, 128): GEMM1 bf16 vs stock rel 2.4e-5 / 2.8e-5 with 100% of elements equal; silu*up 6.4e-5 / 1.3e-4 (100% equal); GEMM2 bf16 3.0e-4 / 3.1e-4 (99% equal). (study's ours_fp32 truth column is meaningless in v2: raw sum has no global scale.)
Decision: holds (parity 20-100x better, deterministic, 5/5); speed -0.9..-3.2% per case (-2.1% overall; judge noise ~2%) accepted. mk_moe_v2.py copied over mk_moe.py (v1 kept as mk_moe_v1.py), generators rerun,
patch regenerated (649 added lines, 37,402 bytes), rules OK, py_compile OK, nested commit e8af105. daily budget: muse_moe_w4a16 0 of 20 checks left.

## v3 planning (Oct 29 afternoon, no checks available until 21:00 Brasilia)
Derived from existing jobs only (details in MOE_V3_PLAN.md): GEMM1/GEMM2 split of mkg_k at the MK-study routings: GEMM1 (job f6ffa925) + glue (job 5ea1d8aa profile) + graph pipeline (job 788e5101) -> GEMM2 = 15 / ~18 / 76 / 100 / 137 us
at decode1 / verify1 / decode32 / verify16 / verify32 = 1.8 / ~2.7 / 2.6 / 2.3 / 1.6 TB/s. Cross-check verify32: 257 + 137 = 394 = mkg_k of job 3bf003f0 (394).
Finding: GEMM2 is NOT near 1.1 TB/s (that was v1's own kernel); both GEMMs cost a constant per 16-token group-pass (GEMM1 6.6-7.0 us, GEMM2 3.5 us ~ 2.7 TB/s per pass), so the verify32 shortfall is the 39 passes for 23 experts
(upper bound of removing duplicates: verify32 -38%, verify16 -13%). Plan ranks: (1) smem-staged dequant for experts with >= 2 groups, (2) per-GEMM KS, (3) glue fusion, (4) PDL.
v3 implementation (14:00-17:00, nothing compiled or run: no nvcc/GPU here): mk_moe/mk_moe_v3.py = v2 + (a) hot-expert kernel mkh_k (producer warp dequantizes each Marlin tile once into a shared-memory fragment ring, 4 consumer warps = 4 token
groups run v2's mma sequence; named barriers; 160 threads; route kernel emits hot chunks (up to 4 groups) for experts with >= 2 groups, plain groups otherwise; used only for M > 16), (b) per-GEMM KS table (default = v2's, same KS for both),
(c) a stage-timer hook in forward. Generator tools/build_v3.py: sub_mk6.py (clean), measure_mk6_v3.py (study: bitwise v3==v2 and v3(hot off)==v2, v3 vs serving stock, graph latency v2/v3off/v3, per-stage event times, register/spill probe;
disables the hot kernel for the official run if the bitwise check fails), measure_mk7_ks.py (graph-timed KS1 x KS2 sweep on v3). Checks start 21:00.

## muse_decode_layer (from 14:50; priority change: phase-1 rank). Board sub_v3.py 1.438x. Plan: DECODE_LAYER_PLAN.md
Layer check 1 (sub_v4.py = sub_v3 with LARGEM_MIN_TOKENS 33 -> 32): Question: does the MoE-v4 large-M CUDA path beat Marlin(+S1 knobs) at T = 32 (decode 32, the weakest load: 1.287x; MoE 133 us sliding / 179 us full of 234 / 459)?
Expect: MoE -8..-12% at T=32 (MoE problem: ~8-9% over Marlin at 32 tokens) => decode 32 sliding 235 -> ~222, full 413 -> ~395, load 1.287 -> ~1.35, overall 1.438 -> ~1.45. All other cases unchanged (within +-1.5%).
Decision: decode 32 load >= +3% and 13/13 -> keep (then port moe5); not better than Marlin -> revert (the Marlin knobs stay) and go to the moe5 port / projections.
RESULT layer check 1, job cf810019b3fe4889 (sub_v4.py): 13/13 pass, overall 1.4415 (CI 1.420-1.463; board job 1.4375). Ours (us), sub_v3 -> sub_v4: decode32 sliding 235.3 -> 214.9 (-8.7%), full 413.0 -> 393.7 (-4.7%); decode1 106.9/117.1 -> 107.2/115.0; decode16 186.8/276.9 -> 181.2/273.3;
verify1 160.0/171.5 -> 158.7/171.6; verify16 403.0/515.0 -> 397.0/501.0; verify32 574.8/802.9 -> 554.4/793.5 (unchanged paths drift -0..-3.5% between jobs; the SGLang column of this job is also 2-5% lower: 256.3 vs 269.8 on decode32 sliding).
Scores: own-SGLang 1.4415; with the board job's SGLang times held fixed 1.4750 (includes the drift); drift-corrected from decode32 only: load ratio 0.929 => x1.0762 => overall x1.0123 => ~1.455. Judge noise of the overall ratio is about +-1.5%.
Decision: KEEP (real gain at T=32, 13/13). Not yet >= +1% by the job's own number (1.4415 < 1.4525): no READY_TO_SUBMIT yet. Next: moe5 port (v5 template) for the other MoE shapes, stacked on sub_v4.
Layer check 2 (planned, NOT run: usage limit): sub_v5.py = sub_v4 + the MoE problem's moe5 template (tools/build_layer_v5.py: device local_start routes, wrap rule, per-bucket self-check vs the layer's Marlin experts, S12_MOE5 switch). Compiles as Python only; CUDA unbuilt.
Expect: MoE -3% over v4 on average, overall +1..1.5%; kill if the judge shows no gain over sub_v4 (1.4415) or 13/13 fails (then S12_MOE5 = False restores v4 paths).
RESULT layer check 2, job 00913137abbe4a30 (sub_v5.py, moe5 active in all 9 buckets, selfcheck rel L2 5.5e-3..6.8e-3 vs Marlin): 13/13 pass, overall 1.435 (CI 1.414-1.457). Ours (us) v4 -> v5: decode1 107.2/115.0 -> 108.3/116.8; decode16 181.2/273.3 -> 180.1/274.9;
decode32 214.9/393.7 -> 217.7/401.4; verify1 158.7/171.6 -> 162.3/168.9; verify16 397.0/501.0 -> 395.5/503.5; verify32 554.4/793.5 -> 544.0/769.0. Fixed-baseline score 1.4750 (v4) vs 1.4717 (v5): no measurable gain from moe5 in the layer (flat within noise, small-T slightly worse, verify32 -2..-3%).
Decision: moe5 port does NOT win in the layer; best layer file stays sub_v4.py (1.4415 own-baseline / ~1.455 drift-corrected; below the +1% bar of 1.4525 by the job's number, no READY_TO_SUBMIT). sub_v5 kept only as a record (S12_MOE5 can be False).
Layer checks left today: 18. MoE v3 checks (21:00) not started.

## decode_layer plan STOPPED (Oct 29, priority change to public refs)
State: board sub_v3.py 1.438x. Best candidate sub_v4.py (job cf810019b3fe4889: own-baseline 1.4415, drift-corrected ~1.455, below +1% bar 1.4525, no READY_TO_SUBMIT). sub_v5 (moe5 port, job 00913137abbe4a30) flat, 1.435. Layer checks left: 18. MoE v3 checks (21:00 Brasilia) not started. New task: PUBLIC_REFS.md / ref_ checks of vjeux + Madhavan files.

## PUBLIC REFS (Oct 29 evening): check the public files as is
Question: what do vjeux/okc-2026 (attention 2.610x, bf16_proj 1.501x) and Madhavan113/kerneloptimizer (decode_layer 2.139x) score per case on the judge, vs our board (1.892 / 1.171 / 1.438)?
Expect: within +-3% of their public scores, 13/13-style tests pass. Decision: pass -> record per-case table in PUBLIC_REFS.md, build candidate = public file + our paths where faster per case; test failure or score far below public -> record and try the other repo's file. MoE: no checks until 21:00 Brasilia.
RESULT public refs (judge, check, off board): attention ref_vjeux_attn.py job f867482c37984518 16/16 pass 2.683x (CI 2.642-2.740; board sub_v10 1.892x); bf16_proj ref_vjeux_proj.py job 4fe090fa5fda4cd9 9/9 pass 1.498x (board sub_v1 1.171x); decode_layer ref_madhavan_layer.py job 9f1575bda4a6430a 13/13 pass 2.127x (board sub_v3 1.438x). Tables in PUBLIC_REFS.md.
Per-case absolute time, ours vs public: attention and decode_layer ours slower in ALL 12 cases (ours/ref 1.14-1.79); bf16_proj ours slower in 28/30 cases, ahead only on o at verify16/verify32 (0.99, 0.97 = noise; those cells are plain cuBLAS = SGLang baseline). So no per-case dispatch pays: candidates = public file + attribution header only.
Candidate checks (question: does the header-only file reproduce the public score and pass tests? expect same +-2%; decision: ok -> READY_TO_SUBMIT.md, fail -> investigate).
RESULT candidate checks (sub_pub1.py = public code + header, same code as ref_*): attention job 00884744e0ab4bbe 16/16 2.602x (ref 2.683x); bf16_proj job 65a6e80c32b04437 9/9 1.499x (ref 1.498x); decode_layer job 8e11ba740b75421d 13/13 2.153x (ref 2.127x). Same-code noise 0-3%.
Decision: all three beat our board by +0.33..+0.72x; READY_TO_SUBMIT.md + PUBLIC_REFS.md written, nothing submitted. MoE: ref_madhavan_moe.py + sub_pub1.py staged, checks start 21:00 Brasilia. Checks left: attention 15, proj 18, layer 16, moe 0.

## MoE public ref (Oct 29 21:05 Brasilia, checks reset)
Question: what does Madhavan's muse_moe_w4a16 (listed 2.357x) score per case on the judge vs our board sub_v1 1.480x? Expect 2.25-2.45x, tests pass. Decision: pass -> check header-only sub_pub1.py, then per-case compare with sub_v1 / MK v2 (dispatch only where ours is faster); fail -> try vjeux's moe file.
RESULT MoE: ref_madhavan_moe.py job 9b10a32ed95b4477 5/5 2.311x; sub_pub1.py (header only) job d97cc04bbb7745ed 5/5 2.347x vs board 1.480x; ours slower in all 6 cases, no dispatch. READY_TO_SUBMIT.md complete for all four. MoE checks left 18.

## PUBLIC REFS 2 (Oct 30 03:35 UTC): the four missing public files
Question: per scored case, is the other public author's file faster than the one in sub_pub1 (attn: Madhavan vs vjeux; proj: Madhavan vs vjeux; layer: vjeux vs Madhavan; moe: vjeux vs Madhavan)? Needed to build sub_pub2 = per-case best.
Expect: each within +-10% of the sub_pub1 overall; tests pass. Decision: if a file wins some cases by >3% (beyond the 2-3% judge noise) -> build a shape-only dispatch sub_pub2 (both credited); if it wins none -> no sub_pub2 for that problem, its sub_pub1 stays.
Files: ref_madhavan_attn.py, ref_madhavan_proj.py, ref_vjeux_layer.py, ref_vjeux_moe.py (unchanged bytes of the public files).
RESULT PUBLIC REFS 2 (judge checks, off board): ref_madhavan_attn.py job 602524390e8b46aa 16/16 2.600x; ref_madhavan_proj.py job 9764ddc69a634518 9/9 1.475x; ref_vjeux_layer.py job c2ed2912923d4096 13/13 2.101x; ref_vjeux_moe.py job f5da266ac4f041c8 5/5 2.346x.
Per case vs the two samples of the sub_pub1 path (ref + header-only candidate, same code): attention other/mean 0.991-1.022 (12 cases); moe 0.963-1.019 (6); layer 0.973-1.061 (vjeux slower on 4 of the 5 cases that differ >3%; faster 0.973 only on verify1-sliding); proj 0.948-1.112 (vjeux faster on decode32 qkv_gate/o by 9-11%, Madhavan faster only on decode1 o 0.948, within the A-vs-A' spread 0.95-1.10).
The same code differs between its own two samples by up to 5-10% on single cells, so cell-level wins are noise except proj decode32 qkv_gate / o, where the sub_pub1 path (vjeux) is already the faster one.
Decision: the two public paths are the same lineage and tie; per-case best = picking the minimum of noisy samples, no real gain. No sub_pub2 built, no check spent on it. sub_pub1 stays the best file per problem. Remaining levers need new kernel work (not a mix of public files).

## MoE port, check 1 (Oct 30): v3 hot-expert kernel (shared dequant), first run ever
Question: does mk_moe_v3 (mkh_k: one producer warp dequantizes each Marlin tile once, 4 consumer warps = 4 token groups) compile, run without deadlock, match v2 bitwise, and speed up verify 16 / verify 32 (the public kernels dequantize hot experts 4x instead of 13x)? File: muse_moe_w4a16/measure_mk6_v3.py (= sub_mk6 + MK3 study, v2 embedded as reference). Reads: `grep MK3` in the job log.
Expect: compile ok; hot regs <= ~100, 0 spill; bitwise v3hot==v2 True for all M (v3hot-off==v2 True); graph us: verify 32 v2 ~420 -> v3 320-360, verify 16 312 -> 285-300, others within +-2%.
Decision: bitwise False -> fix first (no timing is adopted); True and verify16/32 each >= +3% with no other load > 2% worse -> tune hot kernel (check 2-3), then adopt; compile/hang -> fix, at most 2 repair checks; verify32 gain < 6% -> kill the line (producer-bound) and move to the dequant pipe-mix study.
RESULT MoE check 1, job 08d246562aee43ce (measure_mk6_v3.py): compiled, no deadlock, 5/5 pass, regs hot 72 / plain 108, 0 local bytes. Bitwise v3hot == v2 TRUE and v3hot-off == v2 TRUE at M = 1, 8, 16, 32, 128, 256 (maxabs 0); v3 vs serving stock rel 1.7e-4..4.7e-4.
Graph us (v2 / v3 hot-off / v3 hot-on): M=32 241.9 / 242.3 / 256.1; M=128 314.4 / 314.7 / 396.0; M=256 425.5 / 426.4 / 525.5 (hot is +6%, +26%, +23% SLOWER). Judge table (hot on): decode32 0.963x, verify16 1.240x, verify32 1.125x (v2: 1.081 / 1.468 / 1.473).
Stages M=256 (us): hot-off g1_small 269.9 + g2_small 140.4; hot-on g1_hot 207.3 + g1_small 128.2 + g2_hot 102.1 + g2_small 75.8. The plain kernel costs ~7 us per group-pass (128 us for ~18 groups, consistent with MOE_V3_PLAN); the hot kernel takes ~26 us per 4-group chunk (207 us / ~8 chunks), i.e. the same as 4 plain passes: no sharing gain.
Reading: one producer warp per CTA (4 CTAs/SM = 1 producer warp per scheduler) is latency-bound (~730 cycles per k-step for ~130 instructions, IPC ~0.18), so the dequant does not get cheaper, it just moves to a warp that cannot overlap itself. Kill line "verify32 gain < 6%" is hit for THIS design. Numerics goal met (bitwise).
Decision: do not adopt v3. One repair attempt (the design idea is not refuted, only the single latency-bound producer): v4 = 4 producer warps per CTA, producer p handles k-steps i = p mod 4, one fragment slot each (R = NP = 4), own cp.async ring of 4 own-steps; 256 threads. Check 2 = study of v4. If v4 is not >= +3% at verify16/32 over v2 the hot-expert line ends (two checks without moving the metric).
Checks left on muse_moe_w4a16: 16.

## MoE port, check 2 (Oct 30): v4 = v3 hot kernel with 4 producer warps
Question: does spreading the dequant of the hot-expert tile stream over 4 producer warps (k-steps interleaved, fragment slot per producer) remove the single-producer latency bound seen in job 08d246562aee43ce (hot chunk 26 us = 4 plain passes)? File: muse_moe_w4a16/measure_mk8_v4.py (study, v2 embedded as reference). Reads: `grep MK3` in the log.
Expect: compile ok, no deadlock, regs <= 85 and 0 spill (launch bounds 256 x 3), bitwise v4hot == v2 True for all M. Hot chunk cost per 4-group chunk 26 -> 9-14 us (2x-3x faster producers), so M=256 graph us 425 -> 330-370, M=128 314 -> 285-300, M=32 not worse than +2%.
Decision: bitwise True and M=128 and M=256 each >= -3% graph time vs v2 and M=32 within +2% -> run the official clean file (sub_mk8.py), tune (hot_min per M, KS), then adopt; bitwise ok but gain < 3% -> stop the hot-expert line (2 checks without moving the metric), go to the dequant pipe-mix study; bitwise False or hang -> one repair.
RESULT MoE check 2, job 59a50e3e6ccf4c94 (measure_mk8_v4.py): compiled, no deadlock, 5/5 pass, hot regs 72/0B, bitwise v4hot == v2 TRUE at all M (maxabs 0). Graph us (v2 / v4 hot-off / v4 hot-on): M=32 243.0 / 244.1 / 261.9; M=128 316.7 / 317.1 / 384.3; M=256 430.6 / 429.6 / 541.4.
Stages M=256: g1_hot 215.2 (v3: 207.3), g2_hot 111.6 (v3: 102.1): 4 producer warps changed NOTHING. Judge table: decode32 0.964x, verify16 1.268x, verify32 1.181x (v2 1.081 / 1.468 / 1.473).
Reading: the hot kernel is not producer-latency-bound. What it adds is shared-memory traffic: the shared fragments are bf16 (2 KB per lane-tile read per group, 4x the 512 B packed tile), so sharing the dequant through smem costs more smem bandwidth per group-tile (~3.4 KB: frag read 2 + A write 0.5 + ldmatrix 0.5 + (W + frag write)/4) than the plain kernel's ~2.1 KB. Two checks, no movement => hot-expert (shared dequant) line ENDED by rule 6. v3/v4 stay as records (bitwise = v2, slower). Register-level sharing (NT2/NT4) failed earlier on registers (checks 92d20bf7, bc266bce).
New model (derived, not measured): H200 INT32 pipe (LOP3, SHF, PRMT, IADD3) is 16 lanes/clk/SMSP = half rate; per 512 B tile (4 words) ours issues ~15 INT ops per word in dq + ~30 in the scale path (dqs, bf16 broadcast) => ~92 INT warp-instr x 2 clk = ~184 clk per SMSP per tile, against 99 clk per SMSP per tile at 4.8 TB/s (132 SMs x 4 SMSP x 1.755 GHz): cap = 4.8 x 99/184 = ~2.6 TB/s, which is what v2 measures (2.3-2.7 TB/s on non-duplicated loads). The public layout needs ~7 INT + 2 IMAD per word => ~40 INT ops per tile => ~80 clk < 99, HBM-bound (3.5+ TB/s). Lever: move shifts of the Marlin dequant to the FMA pipe (IMAD.SHL / IMAD.HI with an opaque multiplier, as the public kernel's `k16 = 16 + zero` trick). Bit-identical operands.
Checks left on muse_moe_w4a16: 15.

## MoE port, check 3 (Oct 30): dequant instruction mix (INT pipe -> FMA pipe), mk_moe_v5 study
Question: is the v2 GEMM limited by the INT/ALU pipe (model above: ~92 INT warp-instr per 512 B tile vs ~50 allowed at 4.8 TB/s)? File: muse_moe_w4a16/measure_mk10_v5.py (= v2 kernel as template mkg_k<2,4,V>; V0 = v2 code, V1 = left shifts as IMAD, V2 = + right shifts of dq as IMAD.HI (umulhi), V3 = + dqs shifts as IMAD/IMAD.HI; multipliers are runtime-opaque so ptxas cannot turn them back into SHF). Study prints: regs/spills per variant, bitwise vs v2 and graph us per variant per M (`MK5 M=`). (The first submit of this file was REJECTED by the judge, 'starts processes': a cuobjdump subprocess call for a SASS histogram; removed, no SASS is available, so the verdict rests on timing.)
Expect: all four bitwise == v2; SASS: LOP3+SHF count drops V0 -> V3 by ~7 per word-quad... (IMAD up by the same); regs <= 128, 0 spill; graph us: non-duplicated loads (M=8, 16, 32 and the GEMM-heavy M=128/256) -5..-25% at V2/V3 if the INT pipe is the limit; ~0% if the limit is elsewhere (then the model is wrong and the line ends).
Decision: best variant >= -5% on M=8/16/32 with bitwise True -> check 4 = tune (per-variant P/G, scale path), then adopt into the patch module; <= -2% nowhere -> end the pipe-mix line (second no-move result after the hot-expert line is already closed, so move to glue fusion); bitwise False -> repair.
RESULT MoE check 3, job fe4ba01c7dc94c44 (measure_mk10_v5.py): compiled, 5/5, regs v0/v1/v2/v3 = 108/99/100/96, 0 spill; bitwise == v2 for ALL variants at ALL M (torch.equal). Graph us (v2 a/b | V0 V1 V2 V3):
M=1 60.1/59.7 | 60.0 62.4 65.2 69.3; M=8 79.0/78.2 | 78.1 81.7 86.5 93.4; M=16 142.1/141.9 | 140.8 148.3 159.7 175.3; M=32 241.9/242.0 | 238.5 254.7 276.9 308.4; M=128 314.4/314.2 | 309.8 331.2 358.3 400.2; M=256 425.3/424.7 | 419.0 447.7 487.7 542.2.
=> every step of moving shifts to IMAD / IMAD.HI is SLOWER (V1 +5..7%, V2 +14..16%, V3 +27..31% vs V0 at M >= 16). The INT-pipe-bound model is REFUTED: IMAD.HI/IMAD are not free substitutes (FMA-heavy pipe is also loaded by the 16 HMUL2 + MMA address work, and mul.hi is slow). V0 == v2 within 1% (module identical). The Marlin dequant mix stays as is.
Decision: pipe-mix line ENDED (one clear negative; the direction is monotone, the reverse direction has nothing to move to the INT pipe). Hot-expert line ended earlier. Both are recorded negatives; v2 stays. Checks left on muse_moe_w4a16: 14. Judge-table side effect: this run's v2-equivalent kernel_fn scored 1.370x overall (verify16 1.480x, verify32 1.506x, decode32 1.096x), consistent with job 3a3931cb (1.351x) within noise.

## MoE port, check 4 (Oct 30): v6 = v2 without the gather kernel (rank 3 of MOE_PORT_PLAN)
Question: does letting GEMM1 read x rows directly through slot_of_row (no mkp_gather launch, no 16 MB xg write/read) save the ~2-3 us the gather costs per call, bitwise-equal to v2? File: muse_moe_w4a16/measure_mk11_v6.py (study: bitwise v6 == v2, graph us v2 vs v6, both measured twice). Reads: `grep MK6`.
Expect: bitwise True at all M; graph us -2..-4 us at every M (gather is 2.0-3.2 us in the judge profiles; one launch gap saved): decode1 60 -> ~57 (-5%), verify1 79 -> ~76, verify32 425 -> ~422 (-0.7%).
Decision: bitwise True and >= 2 us saved at M <= 32 -> adopt into mk_moe.py/patch (and run the official clean file once); < 1 us -> the launch gap is hidden by graph replay, end the glue line.
RESULT MoE check 4, job be1f321646344708 (measure_mk11_v6.py): compiled, 5/5, bitwise v6 == v2 TRUE at M = 1, 8, 16, 32, 128, 256. Graph us (v2 twice | v6 twice): M=1 60.8/59.7 | 59.2/59.0; M=8 78.8/78.3 | 77.0/77.0; M=16 142.7/142.1 | 140.9/141.0; M=32 242.3/242.1 | 241.0/240.7; M=128 314.2/314.6 | 312.5/312.5; M=256 425.3/425.3 | 424.3/423.8. Scratch 176 -> 160 MB.
Saving 1.0-1.9 us per call at every shape (0.3% at M=256, 2.5% at M=8, 1.8% at M=1). Judge table of this job: decode1 1.607x, decode16 1.263x, decode32 1.087x, verify1 1.336x, verify16 1.496x, verify32 1.468x, overall 1.365x (noise +-2%, not a speed claim).
Decision: below the 2 us line I set, above the 1 us kill line, consistent in all 6 shapes, bitwise and smaller scratch => ADOPT v6 (small but free). Next: copy over mk_moe.py (v2 stays frozen as mk_moe_v2.py), rerun the generators, official clean check of the adopted file, regenerate patch + rules + py_compile, SERVE_PATCH_REPORT section.

## MoE port, check 5 (Oct 30): official clean file of the adopted module (sub_mk4.py = mk_moe.py v6, no study code)
Question: does the clean file (generated by tools/build_sub_mk4.py from the adopted mk_moe.py = v6) pass 5/5 and score like the study runs of the same module (job be1f3216: overall 1.365x; v2 job 3a3931cb 1.351x)? Expect 5/5, overall 1.34-1.40x, no case below v2's table by > 3%.
Decision: pass and in band -> regenerate patch (git diff a55ab7d5 in serve/sglang), rules check, py_compile, write the SERVE_PATCH_REPORT section; fail -> revert mk_moe.py to v2 (mk_moe_v2.py) and rerun the generator.

## Serving parity of the public kernels (Oct 30, chat-driven)
| study | job | result | decision |
|---|---|---|---|
| measure_attnum_v1 (vjeux attention vs SGLang baseline, 16 shapes) | 480a6901bcbd4c45 | rel L2 3.3e-2..9.1e-2, 70-96 % of bf16 outputs differ; k-cache bytes differ on some new tokens, v identical; SGLang vs SGLang bitwise; vjeux vs itself ~1e-5 or bitwise | NOT for the serving patch as is (serving rule: agree with stock nearly as well as stock agrees with itself) |
| measure_projnum_v1 (vjeux bf16_proj vs SGLang _bf16_gemm_dispatch_impl, 37 shapes) | eb2b2f19347e4b4d | rel L2 0..1.5e-4, 0-0.3 % of outputs differ, bitwise at m >= 128 (and o at m = 1); deterministic; router vs an fp32 torch reference 1e-7..3e-6 | candidate for the serving patch (Monday task) |

## Projection port into the serving patch (Nov 2, chat-driven): proj_mk module, budget 8 muse_bf16_proj checks
Step 1 facts (read from serve/sglang, base a55ab7d5): TP8 shapes equal the phase-1 ones (qkv_gate 2304x8192, o 8192x1024, pooler 4096x8192, unpooler 8192x4096, router fp32 256x8192; hidden 8192, 62 layers).
On the judge (H200, `--bf16-gemm-backend` unset = auto) stock runs plain cuBLAS `F.linear` for qkv_proj / o_proj (UnquantizedLinearMethod.apply: the dispatcher is used only for backend cutedsl), for pooler (`pre_expert_proj`) and unpooler (`post_expert_proj`) through `bf16_gemm_dispatch` -> `_bf16_gemm_dispatch_impl` -> `F.linear`. Router: `F.linear` on `hidden.float()` for m <= 8 (fused boundary), `forward_split2` (hi/lo bf16 `torch.mm`, fp32 out) above 8.
Unpooler at m <= 8 may go through `gemm_ag`/`gemm_local` (multicast all-gather); phase 1 times a 1024-row slice there, so the serving table has no unpooler bucket at m <= 8 and no router bucket at m <= 8 (stock stays).
Fixed route table (judge log of board job 65a6e80c32b04437; ties with cuBLAS go to stock = no extra layout): qkv sk2/sk2/sk2/gqx (m 8/16/32/128), o sk2 x3, pooler sk2 x3 + gq (128), unpooler sk2 (16, 32), router sk2 (16, 32) + rk (128, 256). Dropped on purpose: CUTLASS pooler m=256 (second compile, 3 us at one load), torchT copies (4 GB for <= 6 %), m=1 GEMVs in serving (DFlash verify never has m < 8). Phase-1 plan keeps m=1 cgemv/rgemv and the unpooler slice so the check file is comparable to 1.499x.

### Check 1: measure_projmk1.py (= sub_projmk1.py + CHECK-only study)
Question: does the fixed-table module (a) pass 9/9, (b) score like the autotuned file (board 1.499x, job 65a6e80c; CUTLASS pooler m=256 missing costs ~0.3 %), (c) verify every compressed weight bitwise (PROJMEM roundtrip True), (d) what do the serving-plan layouts cost per layer (PROJMEM), (e) is serving-plan parity vs stock at the earlier level (PROJSRV rel <= 1.5e-4, bitwise at m >= 128, det True)?
Expect: 9/9, overall 1.47-1.51x; roundtrip True for all; serving extra memory 200-350 MB per layer (3 pooler streams + 2 unpooler streams + 3 o streams + qkv + router), i.e. 12-22 GB over 62 layers.
Decisions: 9/9 and 1.47+ and roundtrip True -> keep the table; memory > 10 GB per GPU -> check 2 tries a lean layout (one sk2 stream per layer, configs sharing a layout) and compares speed; roundtrip False or not bound -> repair the decoder/binding before anything else; PROJSRV rel > 5e-4 or det False for a bucket -> that bucket goes back to stock.
RESULT check 1, job 1c6488e02a414a71 (measure_projmk1.py, 9 of 10 checks left): 9/9 passed, overall **1.486x** (CI 1.464-1.509; board file sub_pub1 job 65a6e80c: 1.499x, so the dropped CUTLASS pooler m=256 and noise cost <= 1 %). Case table as the board file within noise (verify16/32 pooler 1.087x/1.014x, qkv 0.994x at m=256).
PROJMEM: roundtrip **True** for all five weights (bitwise decode of the compressed streams, both plans). Serving plan extra memory per layer: qkv 27.0 MB, o 36.1, pooler 144.2, unpooler 96.1, router 24.0 = **327.4 MB, 19.82 GB over 62 layers** (too much: ~half the KV pool of a 0.85 mem-fraction H200 would go; the extra streams are the per-bucket sk2 layouts: pooler 3, unpooler 2, o 3).
PROJSRV (serving plan vs `_bf16_gemm_dispatch_impl`, random inputs): sk2 / gq / gqx buckets rel L2 1.5e-6..1.6e-4, 0.0-0.4 % of bf16 outputs differ, det True everywhere; pooler gq at m = 33, 64 bitwise. Router sk2 (m 9..32) rel ~1.2e-6 (fp32), det True. **Router rk (m 33..256): det False** (rel ~1e-6, run-to-run different bits) -> by my rule it goes back to stock unless a deterministic replacement is found.
Decision: keep the table; check 2 tests (a) lean layout: o / pooler / unpooler use ONE sk2 layout per weight (their m = 32 config for every token bucket; SK2_PICK_LEAN) and router m 128/256 on the deterministic-by-design wgmma 'gemm' route instead of rk; (b) timing vs stock per bucket (graph replay, cold L2) for both layouts.
Board read after check 1: overall rank 348 (okc leaderboard; was 129 on Oct 30, others moved, our entries did not).

### Check 2: measure_projmk1.py (study: full vs lean layouts, router gemm vs rk)
Question: how much speed does the lean layout (one sk2 stream per weight) lose per bucket, and is router 'gemm' bitwise reproducible?
Expect: lean memory ~150 MB/layer (~9 GB); lean sk2 times within +0..1.5 us of full at m = 8/16/32 for o/pooler/unpooler (a tokr-8 launch of a tokr-32 layout wastes ring stages); router gemm det True with time between rk (9-10 us) and stock (19-21 us); the phase-1 table repeats check 1 (noise estimate).
Decision: lean costs <= 1 us per op and memory <= 10 GB -> adopt lean for the patch; a bucket where lean > stock -> that bucket goes to stock; router gemm det False -> router m >= 33 stays stock; any lean launch failure/compile error -> keep full layout for the offending weight only and rethink memory (drop o and unpooler streams first: smallest gain per MB).
RESULT check 2, job a6dd769d49d64eed (measure_projmk1.py with study; 8 of 10 checks left): 9/9, overall 1.482x (CI 1.459-1.504; same phase-1 code path as check 1 = 1.486x: noise estimate +-0.3 %).
PROJMEM: lean layout 167.2 MB per layer (o 12.0, pooler 48.1, unpooler 48.0, qkv 27.0, router 32.0 incl. 8 MB of temporaries still alive at measurement) = 10.1 GB over 62 layers vs 327.4 MB / 19.8 GB full. Round trip True everywhere.
PROJSRV us (graph replay, cold L2; stock / full / lean): qkv m8 18.3/13.6/13.4, m16 17.8/13.5/13.5, m32 18.0/14.7/14.7, m128 20.9/18.5/18.6; o m8 10.4/9.6/9.8, m16 10.5/9.3/9.5, m32 10.6/9.6/9.6; pooler m8 25.4/18.6/19.1, m16 25.4/18.5/18.8, m32 25.8/19.3/19.3, m128 28.8/25.6/25.6; unpooler m16 22.6/17.9/18.5, m32 22.7/19.4/19.2. => lean costs 0.0-0.6 us per call (largest: unpooler m16 +0.6, pooler m8 +0.5).
Router: sk2 (m 9..32) 8.7/9.2 us vs stock 18.8/19.1, det True. m 128: stock 20.0, rk 9.8 (det **False**), gemm 13.7 (det **True**); m 256: stock 20.8, rk 10.7 (False), gemm 17.4 (True). Parity of all buckets as check 1 (rel <= 1.6e-4 bf16 outputs; router fp32 ~1e-6).
Decisions: (1) adopt lean layout (memory halves for <= 0.6 us); (2) router m 33..256 -> gemm (deterministic, 3.4-6.3 us faster than stock), rk dropped for serving; (3) serving weight set made configurable (SERVING_WEIGHTS); default drops the unpooler: it is stock at m <= 8 (gemm_ag), gives nothing at m = 128/256 (stock), only 3.5-4.7 us at m 9..32, and costs 48 MB x 62 = 3 GB; keeps qkv_gate, o, pooler, router (~112 MB/layer, ~6.8 GB). Phase-1 plan also moves to lean + gemm so the check file exercises the final configuration.
Board read after check 2: see BOARD_LOG.

### Check 3: measure_projmk1.py with the FINAL module (lean layout, router gemm, SERVING_WEIGHTS = qkv_gate, o, pooler, router; selftest rows widened; router temporaries freed)
Question: (a) does the final configuration still pass 9/9 with every selftest bucket on (PROJMEM off [] for all weights), (b) what is the phase-1 score with router on gemm instead of rk, (c) what is the real serving memory per layer (temporaries freed before measuring), (d) are the widened selftest rows (partial tiles m = 3, 9, 24, 64, 100, 129) all accepted?
Expect: 9/9; overall 1.45-1.48x (router verify16/32 lose ~3-6 us each vs rk: 2.7x/2.5x -> ~1.8x/1.5x); serving memory ~105-115 MB per layer (qkv 27 + o 12 + pooler 48 + router ~25) = 6.5-7.1 GB; no bucket off.
Decision: all as expected -> patch is final, run the clean generated file once (check 4); any bucket off or a launch failure on a partial tile -> that bucket goes to stock in ROUTES and the widened row stays as a guard; memory > 8 GB -> drop o.
Patch state before this check: serve/patch.diff regenerated (7 files, 4199 added lines, 216,082 bytes, RULES OK, py_compile OK, `git apply --check -R` clean); nested commit 2edc306 on my-patch.
RESULT check 3, job 8672b92866334e50 (measure_projmk1.py, final module; 7 of 10 checks left): 9/9 passed, overall **1.433x** (CI 1.412-1.455) - below my 1.45-1.48 expectation by ~1.2 %: the final plan runs the router on gemm (m=128: 10.07 us, 1.615x; m=256: 13.76 us, 1.256x; rk gave 2.8x / 2.5x in check 1) and the lean pooler/o layouts, and lacks the board file's CUTLASS pooler m=256. Phase 1 is not the target of this plan; the board entry stays sub_pub1.
PROJMEM final (serving plan): qkv_gate 27.0 MB, o 12.0, pooler 48.1, unpooler not bound (SERVING_WEIGHTS), router 24.0 = **111.1 MB per layer = 6.73 GB over 62 layers** (phase-1 plan with unpooler: 25/60/96/60/55 MB). Round trip True for all; **no bucket off** in any weight (selftest rows now include m = 3, 9, 24, 64, 100, 129 partial tiles).
PROJSRV us (stock / final): qkv m8 18.6/13.7, m16 18.2/13.8, m32 18.5/14.9, m128 21.2/18.6; o m8 10.6/9.8, m16 10.5/9.8, m32 10.7/9.9; pooler m8 25.7/19.3, m16 25.5/18.8, m32 26.0/19.2, m128 28.5/25.9; router m16 18.9/8.9, m32 19.3/9.4, m128 20.1/13.8, m256 20.9/17.5. Parity: bf16 buckets rel 3.2e-5..1.5e-4, 0.02-0.37 % of outputs differ, det True; router fp32 rel 1.1e-6..1.8e-6, det True.
Decision: module and patch are final for this task. Check 4 runs the clean generated file (sub_projmk1.py, no study code) once.

### Check 4: sub_projmk1.py (clean generated file, final module, no study code)
Question: does the clean file (what the generator emits for phase 1) pass 9/9 and score like the study run of the same module (job 8672b928: 1.433x)? Expect 9/9, 1.41-1.46x (study code only runs in prepare). Decision: pass and in band -> the generator output is validated, write the report; fail -> compare with the study file (the difference would be the STUDY hook only) and repair.
RESULT check 4, job e0fa30e810d84688 (sub_projmk1.py clean; 6 of 10 checks left): 9/9 passed, overall **1.443x** (CI 1.422-1.465), inside the predicted band and within noise of the study file (1.433x). Generator output validated. Budget: 4 of 8 muse_bf16_proj checks used for this task.
Report: SERVE_PATCH_REPORT.md section "v4 round (Nov 2)". Patch: serve/patch.diff (7 files, 216,082 bytes, RULES OK, py_compile OK, apply -R clean), nested commit 2edc306 on my-patch.

## Preflight round (Nov 3): smoke test of the patched SGLang code inside the judge image
The phase-1 judge image is the serving image (lmsysorg/sglang:dev-muse-spark-open, SGLang a55ab7d5). CHECK-only files (never submitted) built by tools/build_preflight.py: sub_mk11.py / sub_projmk1.py + the patched serve/sglang texts embedded (zlib+base64) and loaded with importlib under the installed module names; prepare() prints compact `PREFLIGHT ...` lines. Budget: at most 4 muse_moe_w4a16 and 2 muse_bf16_proj checks.

### Preflight MoE check 1: preflight_moe1.py
Question: does the patched muse_spark_fp4_moe.py (bind_mk_moe, reordered set_post_expert_norm, MK branch of _forward_marlin) import in the image, bind through the REAL MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading (a layer built without __init__: that needs the distributed groups), and agree with the stock Marlin path of the same layer for M = 1, 8, 32, 128, 256 (257 = stock), under CUDA-graph capture, with the env flag off and with a failing build?
Expect: import ok; installed files == a55ab7d5 blobs; bound True; rel(patched, stock_fwd) 1e-3..3e-3 (phase-1 file passed the 1.5% tolerance), bitwise equal to the phase-1 MK output; det True; graph replay bitwise == eager; flag off -> not bound; forced failure -> no exception. The real process_weights_after_loading may need server globals (stubbed and reported) - if it cannot run at all the fallback binds the phase-1 tensors.
Decisions: any FAIL line -> fix the patch (regenerate, rules check, py_compile) and rerun once; all ok -> next check covers the projection side (separate problem budget). Things that cannot run outside a server (distributed groups, forward_batch, the MoE block's call sites in muse_spark.py) are listed in PREFLIGHT_REPORT.md.
