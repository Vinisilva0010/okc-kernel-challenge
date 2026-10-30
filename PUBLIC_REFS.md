# PUBLIC_REFS: public Apache-2.0 entries checked on the judge (Oct 29 evening)

Sources: `ref-vjeux/` = https://github.com/vjeux/okc-2026 (Copyright (c) 2026 vjeux, Apache-2.0); `ref-madhavan/` = https://github.com/Madhavan113/kerneloptimizer
(Copyright (c) 2026 Madhavan113, derived from vjeux, Apache-2.0, see its NOTICE). All runs are `okc check` (off the board). Public files copied as `ref_*.py`
(unchanged bytes); candidates `sub_pub1.py` = same code + a 2-3 line attribution header. SGLang times differ per job (judge drift), so compare the time columns.
License files: ref-vjeux has a LICENSE (stock Apache-2.0 text, owner placeholder unfilled); ref-madhavan has only a NOTICE (states Apache-2.0 and the derivation from vjeux) and no LICENSE file in the clone. The README of vjeux says Apache 2.0.

## Result summary
| problem | public file checked | job | tests | public ref | candidate job | candidate | our board |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | vjeux (2.610 listed) | f867482c37984518 | 16/16 | 2.683x | 00884744e0ab4bbe | 2.602x | 1.892x |
| muse_bf16_proj | vjeux (1.501 listed) | 4fe090fa5fda4cd9 | 9/9 | 1.498x | 65a6e80c32b04437 | 1.499x | 1.171x |
| muse_decode_layer | Madhavan (2.139 listed) | 9f1575bda4a6430a | 13/13 | 2.127x | 8e11ba740b75421d | 2.153x | 1.438x |
| muse_moe_w4a16 | Madhavan (2.357 listed) | 9b10a32ed95b4477 | 5/5 | 2.311x | d97cc04bbb7745ed | 2.347x | 1.480x |

Same code, two samples: attention 2.683 vs 2.602 (-3%), proj 1.498 vs 1.499, layer 2.127 vs 2.153: judge noise is about +-2-3% on these.
Only the vjeux or Madhavan version of each problem was checked (the listed faster one). Madhavan's attention/proj are different code from vjeux's (4374 / 2468 differing lines) and were not checked.

## Per-case tables (board ours vs public ref vs header-only candidate)
"ours/ref" = our board time / public ref time (>1 means the public file is faster). Our paths are slower in every attention and decode_layer case (ratio 1.14-1.79);
in bf16_proj only o at verify 16 / verify 32 is ahead (0.99 / 0.97, noise, and those cells are plain cuBLAS = the SGLang baseline path). A per-case dispatch to our
code therefore gains nothing: the candidates are the public code unchanged.

### muse_attention_decode: vjeux/okc-2026
Public file as is (`ref_*.py`): job f867482c37984518, 16/16 tests, **2.683x** (CI 2.642-2.740). Header-only candidate (`sub_pub1.py`): job 00884744e0ab4bbe, 16/16 tests, **2.602x** (CI 2.563-2.660). Our board file: sub_v10.py (job a095ad7c9ccb4136, 1.892x).

| case | board ours us | public ref us | candidate us | board speedup | ref speedup | cand speedup | ours/ref (time) |
|---|---|---|---|---|---|---|---|
| decode 1 · sliding | 8.19 | 5.10 | 5.40 | 2.549 | 4.068 | 3.851 | 1.61 |
| decode 1 · full | 15.69 | 10.80 | 10.90 | 2.034 | 2.936 | 2.904 | 1.45 |
| decode 16 · sliding | 13.38 | 8.10 | 7.70 | 2.026 | 3.382 | 3.488 | 1.65 |
| decode 16 · full | 106.16 | 83.30 | 86.60 | 1.889 | 2.428 | 2.321 | 1.27 |
| decode 32 · sliding | 17.08 | 9.70 | 10.20 | 1.858 | 3.231 | 3.096 | 1.76 |
| decode 32 · full | 181.50 | 159.20 | 166.20 | 2.132 | 2.421 | 2.314 | 1.14 |
| verify 1 · sliding | 9.46 | 6.50 | 6.70 | 2.456 | 3.565 | 3.399 | 1.46 |
| verify 1 · full | 21.06 | 14.10 | 14.30 | 1.724 | 2.604 | 2.549 | 1.49 |
| verify 16 · sliding | 17.65 | 11.50 | 11.60 | 1.687 | 2.580 | 2.527 | 1.53 |
| verify 16 · full | 136.26 | 98.00 | 99.60 | 1.513 | 2.102 | 2.058 | 1.39 |
| verify 32 · sliding | 25.93 | 14.50 | 14.50 | 1.345 | 2.386 | 2.372 | 1.79 |
| verify 32 · full | 259.42 | 180.20 | 182.20 | 1.508 | 2.174 | 2.138 | 1.44 |

SGLang column differs per job (judge drift), compare the time columns with that in mind.

### muse_bf16_proj: vjeux/okc-2026
Public file as is (`ref_*.py`): job 4fe090fa5fda4cd9, 9/9 tests, **1.498x** (CI 1.476-1.521). Header-only candidate (`sub_pub1.py`): job 65a6e80c32b04437, 9/9 tests, **1.499x** (CI 1.476-1.521). Our board file: sub_v1.py (job d4d1752631d84ec3, 1.171x).

| case | board ours us | public ref us | candidate us | board speedup | ref speedup | cand speedup | ours/ref (time) |
|---|---|---|---|---|---|---|---|
| decode 1 · qkv_gate | 14.06 | 9.10 | 8.60 | 1.006 | 1.568 | 1.607 | 1.55 |
| decode 1 · o | 6.58 | 5.40 | 5.30 | 1.050 | 1.316 | 1.329 | 1.22 |
| decode 1 · pooler | 19.01 | 13.20 | 13.60 | 1.132 | 1.597 | 1.591 | 1.44 |
| decode 1 · unpooler | 5.84 | 3.50 | 3.90 | 1.232 | 1.989 | 1.861 | 1.67 |
| decode 1 · router | 6.43 | 3.60 | 3.70 | 1.594 | 2.858 | 2.821 | 1.79 |
| decode 16 · qkv_gate | 14.23 | 9.80 | 9.90 | 1.004 | 1.457 | 1.426 | 1.45 |
| decode 16 · o | 6.63 | 5.70 | 5.70 | 1.014 | 1.198 | 1.178 | 1.16 |
| decode 16 · pooler | 19.46 | 14.40 | 14.20 | 1.100 | 1.502 | 1.511 | 1.35 |
| decode 16 · unpooler | 18.60 | 13.80 | 13.80 | 0.999 | 1.353 | 1.342 | 1.35 |
| decode 16 · router | 6.92 | 4.70 | 4.80 | 2.185 | 3.119 | 3.062 | 1.47 |
| decode 32 · qkv_gate | 14.44 | 10.10 | 10.40 | 0.999 | 1.399 | 1.337 | 1.43 |
| decode 32 · o | 6.95 | 5.70 | 6.30 | 0.974 | 1.192 | 1.121 | 1.22 |
| decode 32 · pooler | 20.12 | 14.90 | 15.00 | 1.092 | 1.481 | 1.467 | 1.35 |
| decode 32 · unpooler | 18.53 | 14.90 | 15.10 | 0.999 | 1.264 | 1.255 | 1.24 |
| decode 32 · router | 7.14 | 5.30 | 5.50 | 2.110 | 2.856 | 2.843 | 1.35 |
| verify 1 · qkv_gate | 14.08 | 9.70 | 9.40 | 0.998 | 1.477 | 1.497 | 1.45 |
| verify 1 · o | 6.69 | 5.70 | 5.30 | 1.034 | 1.170 | 1.174 | 1.17 |
| verify 1 · pooler | 19.50 | 14.50 | 14.30 | 1.112 | 1.492 | 1.513 | 1.34 |
| verify 1 · unpooler | 5.93 | 4.00 | 4.30 | 1.217 | 1.678 | 1.642 | 1.48 |
| verify 1 · router | 6.70 | 5.10 | 4.70 | 3.920 | 5.211 | 5.530 | 1.31 |
| verify 16 · qkv_gate | 17.15 | 14.90 | 14.60 | 1.020 | 1.166 | 1.153 | 1.15 |
| verify 16 · o | 7.54 | 7.60 | 7.10 | 1.000 | 0.984 | 1.021 | 0.99 |
| verify 16 · pooler | 26.29 | 21.80 | 22.10 | 0.976 | 1.146 | 1.131 | 1.21 |
| verify 16 · unpooler | 19.95 | 19.30 | 19.00 | 1.014 | 1.035 | 1.049 | 1.03 |
| verify 16 · router | 9.31 | 6.20 | 5.80 | 1.780 | 2.639 | 2.800 | 1.50 |
| verify 32 · qkv_gate | 19.98 | 19.80 | 20.00 | 1.008 | 0.999 | 0.999 | 1.01 |
| verify 32 · o | 9.28 | 9.60 | 9.50 | 1.000 | 0.976 | 1.025 | 0.97 |
| verify 32 · pooler | 29.43 | 27.90 | 27.40 | 1.020 | 1.117 | 1.118 | 1.05 |
| verify 32 · unpooler | 24.79 | 24.30 | 23.90 | 0.996 | 1.010 | 1.016 | 1.02 |
| verify 32 · router | 11.40 | 7.10 | 7.10 | 1.566 | 2.432 | 2.480 | 1.61 |

SGLang column differs per job (judge drift), compare the time columns with that in mind.

### muse_decode_layer: Madhavan113/kerneloptimizer
Public file as is (`ref_*.py`): job 9f1575bda4a6430a, 13/13 tests, **2.127x** (CI 2.095-2.159). Header-only candidate (`sub_pub1.py`): job 8e11ba740b75421d, 13/13 tests, **2.153x** (CI 2.120-2.185). Our board file: sub_v3.py (job b8797228d4e54e03, 1.438x).

| case | board ours us | public ref us | candidate us | board speedup | ref speedup | cand speedup | ours/ref (time) |
|---|---|---|---|---|---|---|---|
| decode 1 · sliding | 106.89 | 76.00 | 73.30 | 1.597 | 2.230 | 2.326 | 1.41 |
| decode 1 · full | 117.07 | 81.20 | 78.50 | 1.557 | 2.227 | 2.291 | 1.44 |
| decode 16 · sliding | 186.81 | 128.60 | 122.20 | 1.447 | 2.015 | 2.035 | 1.45 |
| decode 16 · full | 276.90 | 213.70 | 203.90 | 1.593 | 2.075 | 2.091 | 1.30 |
| decode 32 · sliding | 235.25 | 155.10 | 153.60 | 1.147 | 1.706 | 1.695 | 1.52 |
| decode 32 · full | 413.00 | 309.80 | 288.90 | 1.516 | 2.022 | 2.088 | 1.33 |
| verify 1 · sliding | 160.03 | 110.50 | 110.60 | 1.643 | 2.370 | 2.415 | 1.45 |
| verify 1 · full | 171.54 | 116.90 | 114.40 | 1.595 | 2.330 | 2.384 | 1.47 |
| verify 16 · sliding | 403.01 | 256.50 | 259.60 | 1.328 | 2.112 | 2.107 | 1.57 |
| verify 16 · full | 515.04 | 338.20 | 338.20 | 1.364 | 2.089 | 2.065 | 1.52 |
| verify 32 · sliding | 574.77 | 327.80 | 329.50 | 1.297 | 2.247 | 2.253 | 1.75 |
| verify 32 · full | 802.93 | 491.50 | 492.70 | 1.375 | 2.210 | 2.215 | 1.63 |

SGLang column differs per job (judge drift), compare the time columns with that in mind.

## MoE: Madhavan113/kerneloptimizer
Public file as is (`ref_madhavan_moe.py`): job 9b10a32ed95b4477, 5/5 tests, **2.311x** (CI 2.277-2.346). Header-only candidate (`sub_pub1.py`): job d97cc04bbb7745ed, 5/5 tests, **2.347x** (CI 2.311-2.382). Our board file: sub_v1.py (job e65993b3c77c4052, 1.480x).
Our board path is slower in all 6 cases (ours/ref 1.3-1.7 in time), so no dispatch to our code (MK v2/v3 not mixed in: they are slower than sub_v1's class of numbers at these shapes and unverified against these).

| case | board ours us | public ref us | candidate us | board speedup | ref speedup | cand speedup | ours/ref (time) |
|---|---|---|---|---|---|---|---|
| decode 1 | 42.54 | 25.30 | 25.20 | 1.809 | 3.049 | 3.057 | 1.68 |
| decode 16 | 131.26 | 93.80 | 95.30 | 1.424 | 2.004 | 2.035 | 1.40 |
| decode 32 | 186.50 | 128.20 | 126.10 | 1.093 | 1.606 | 1.631 | 1.45 |
| verify 1 · 8 tokens | 97.34 | 66.90 | 63.20 | 1.480 | 2.195 | 2.248 | 1.46 |
| verify 16 · 128 tokens | 294.78 | 186.00 | 183.30 | 1.564 | 2.534 | 2.576 | 1.58 |
| verify 32 · 256 tokens | 386.23 | 221.50 | 217.90 | 1.613 | 2.793 | 2.843 | 1.74 |

## Step 4: how the public kernels store weights / read KV, and could they run in SGLang?

### MoE (ref-madhavan/muse_moe_w4a16/submission.py, from vjeux's; one persistent kernel `moe_fused`, grid = 132 SMs)
Weights in. prepare(weights) takes the judge's RAW tensors: w13 / w2 uint8 [E, rows, K/2] (two FP4 e2m1 nibbles per byte), w13_sf / w2_sf float8_e4m3 [E, rows, K/16],
w13_gs / w2_gs **float32** per expert (asserted float32), post_norm_w bf16. For the 32 local experts it only permutes bits/bytes (never requantizes):
- codes: per 16-row block (rows interleaved so that gate/up pairs are adjacent: `rowsA`), `_pack_codes` scatters the 4 bits of each nibble over 4 uint32 "registers" per
  lane (bit positions `_POS`, DQ7 variant) so the dequant to bf16 needs a few shifts/LOP3 (+ PRMT for the e4m3 scale -> bf16 via a 2^64-scaled LUT; the global scale is applied once, in fp32, in the epilogue). Natural-k variant `_pack_codes_nat` for the wgmma path.
- scales: e4m3 bytes reordered `[rb][w][g][t][c][h]`, stored next to the codes in the same unit.
- one TMA bulk unit = 16 rows x K=4096 = 32 KB codes + 4 KB scales = 36,864 B (skinny builds M <= 16); WGS build (M >= 17, RS-wgmma, 4 consumer warpgroups) uses
  `_wgs_copy`: [256-row unit][16 stages of K=256][4 warpgroups][9,216 B]. `state["w"]` holds **up to four repacked copies** of each GEMM's weights (plain, fat order, unit order, WGS) plus the LUT.
Data flow: route() builds per-expert token lists on device inside the kernel (reads topk_ids / topk_weights, so it ignores SGLang's own sorted/aligned routing), GEMM1+SwiGLU -> bf16 `inter` through L2,
GEMM2 + sum of squares, grid barrier, final norm (post_norm_w) + routing-weight sum -> bf16 [M, 4096]. M tiers: <=4 TINY, 5-8 TINY2, 9-16 MID (SPLIT), >=17 WGS; chunks of 384 tokens.
Could it run inside SGLang on the Marlin tensors? **Not directly.**
1. SGLang's `prepare_moe_nvfp4_layer_for_marlin` (serve/sglang/python/sglang/srt/layers/quantization/marlin_utils_fp4.py:468) REPLACES layer.w13_weight / w2_weight with the Marlin int32 tiles and the scales with processed e4m3 (`nvfp4_marlin_process_scales`), and
   casts the global scale to the param dtype (bf16, x 2^119) with `.to(param_dtype)`: the original fp32 global scale and the raw nibble layout are gone after loading. Their kernel wants float32 gs and raw-order codes/scales.
2. Codes could be recovered from the Marlin tiles (our decode, CLAUDE.md "Marlin NVFP4 layout", is validated: the 4-bit values are a permutation) and rebuilt into their layout, but the fp32 global scale cannot be recovered exactly from bf16 (8-bit mantissa); the correct patch is to run their repack in the loader from the checkpoint tensors (hook in process_weights_after_loading, before/instead of the Marlin prep).
3. Memory: per layer 32 experts x 12,288 rows x 4,096 K / 2 = ~805 MB codes + ~100 MB scales (derived from the shapes in the file). Keeping Marlin AND 3-4 copies is impossible for a multi-layer model on one H200; a serving patch would need a single layout
   (M >= 17 needs WGS; M <= 16 needs unit order = two copies at least) and must drop the Marlin copy. Needs a measurement of the layer count before deciding.
4. Interface: they fuse routing, SwiGLU and the post-expert RMSNorm and take topk_ids/topk_weights as given, matching the problem, not SGLang's fused-MoE runner signature; the runner/dispatch (EP8, local_start) would have to be wrapped. Also SGLang's Marlin stage dtype choices (bf16 gs) mean numerics vs stock must be re-measured (serving requires agreement with stock as good as stock-vs-stock; their parity vs Marlin has NOT been measured by us).

### Attention (ref-vjeux/muse_attention_decode.py)
KV cache read. K and V are FP8 E4M3 caches addressed by token slot, `cache_loc` (int64, slot of each new token) and `page_table` (the SGLang FA3-style table = req_to_token[req_pool_indices, :max_seq_len_k]; with page_size 1 the entry is the cache row);
`seq_lens` per request. A cache row is [slot][kv_head][64] bytes, so a KV-head pair is one 128-byte row (`ROWB = 128`). Each CTA = (request, KV-head pair, split) streams 64-key tiles through shared memory with cp.async (3-4 stages; flat full-layer kernel adds a producer warp), QK^T on FP8 tensor-core MMA (queries rounded to E4M3, as FA3 does),
PV with FP16 MMA (P in FP16, V exact in FP16), except q_len 8 on sm_90 where PV is an FP8 wgmma with P in E4M3 (as SGLang's FA3 FP8 path). k_scale / v_scale are read from the device scalars. The kernel also does QK-RMSNorm, RoPE (sliding layers), FP8 quantization and the cache write of the block's new K/V, then merges splits with fp32 atomics and applies rmsnorm * sigmoid(gate).
Single-request page-size-1 builds speculate `seq = page_table.shape[1] - q_len` (SGLang builds exactly that width) and re-check against `seq_lens` on device before any K/V address is used; a miss re-runs with the real length.
Could it run inside SGLang? **Likely yes, with a thin adapter, because it already uses SGLang's own cache format**: FP8 E4M3 K/V cache, per-tensor k_scale/v_scale, cache_loc, page_table, cache_seqlens and q_len in {1, 8} (the DFlash verify length) are exactly the FA3-backend inputs; no weights are involved, so the Marlin tensors are irrelevant to it.
Open points (not verified, would need work on serve/sglang): head dim fixed at 64 with 16 query heads per KV pair (hkv even, tp8 shapes) and q_len only 1 or 8 (others use `_fallback`, plain PyTorch); the kernel fuses the QKV post-processing (norm, RoPE, cache write) so the SGLang layer's own qk-norm/rope/cache-set calls must be skipped for the same call (double write is harmless but double RoPE is not); the cluster/cooperative launches and MAX_BATCH 512 / 32-bit indexing limits (INDEX_LIMIT 2^31) need the real serving pool size (80k ctx x 32 requests fits? compute from the pool, not verified); the FP8 query rounding / FP8 P in the q_len-8 path must agree with stock FA3-FP8 numerics (the task needs outputs as close to stock as stock to itself: only the kernel test tolerance has been checked, 16/16).

## Step 4b: MoE kernels in detail (Oct 30; sources: ref-vjeux/muse_moe_w4a16.py, ref-madhavan/muse_moe_w4a16/submission.py (derived from it), ref-madhavan/notes/pub_moe_design.md with line refs; copies/ring sizes re-checked in the code)
Madhavan's design memo is a third-party read of vjeux's kernel; I re-checked the weight copies (`prepare`, lines ~3244-3290) and the ring constants (`OKC_NS=3`, 36,864 B units) in the code. The instruction tallies below are the memo's, not counted from SASS by us.

**Weights / memory.** `prepare` builds bit-permuted copies of the RAW checkpoint tensors (not the Marlin ones): plain, fat order, unit order, and for M >= 17 a WGS copy, for both w13 and w2 (8 tensors in `state["w"]`).
Per layer and GPU that is 32 experts x (805 MB codes + ~100 MB scales) per copy of the two GEMMs together, i.e. up to four copies = ~3.6 GB per layer (from shapes; layer count not measured). The fp32 global scale is applied in the epilogue (x 2^62). Not usable inside SGLang as is (see section 4 above): it needs the raw nibbles and fp32 global scale.
Codes: one 32-bit word = 8 E2M1 codes = one lane's m16n8k16 A fragment for one k16 step; the K order is permuted so that a lane's 4 k16 steps in a chunk share one 16-K scale block (0.5 PRMT + 0.5 IMAD of scale decode per word). Bits of each code are spread so that dequant is AND + a few shifts.

**Orientation.** Weights are the MMA A operand (m16 rows), tokens are N (n8 tiles) with x fragments held in REGISTERS (no ldmatrix). Ours: weights are the B operand, tokens are M (16 per group), x via ldmatrix from smem.

**Work split.** One persistent launch (grid = 132 SMs, 1 CTA/SM, no queue). Expert with > 8 tokens = ceil(tot/64) "fat" jobs of <= 64 tokens (so a 201-token expert dequantizes its weights 4 times, ours 13 times); 1-8 tokens = 1 skinny job; MID (9-16 tokens) splits an expert into jobs of <= 8.
Tiny builds (M <= 16): unit = (job, 16-row block) x full K (36,864 B), static equal slices of the unit list per CTA, split-K INSIDE the CTA (8 warps x 512 K, smem reduce by a helper warp). M >= 17: RS-wgmma m64nNk16 (N = 8..64 tokens), unit = 256 rows x 16 stages of K=256, stream-K across CTAs with cost-weighted stage ranges and a global fp32 fixup (red.add).

**Pipeline.** TMA bulk copies (cp.async.bulk + mbarrier, evict_first on weights), ring of 3 x 36,864 B slots (~108 KB in flight per SM). First two unit TMAs are issued BEFORE routing finishes. Wgmma build: separate weight-TMA producer warp and x-TMA producer warp, 3-slot x ring of 32 KB, empty barriers released as soon as the stage is in registers.

**Dequant (per the memo).** `dq4` DQ7: 7 ALU + 2 IMAD per 8 weights (AND with 0x81C081C0 on the word, on q*16, on rotl 10, and a 2-term combination for the fourth), then 4 HMUL2 with the scale (bf16x2 of scale x 2^64, built by PRMT + IMAD); the e2m1 x 2^-126 trick is exact. Total ~14 instructions per 8 weights + the MMA. Our Marlin-layout `mk_dq` path is ~15 (dq) + 4 HMUL2 + ~2 PRMT + ~3.5 (scale decode) = ~25 per 8 weights (counted from the source in mk_moe_v2.py, not from SASS).

**Epilogue.** Split-K partials are summed in smem (tiny) or via global fixup (wgmma). GEMM1's SwiGLU is formed in the accumulator registers (gate and up rows interleaved inside one m16 tile) and, in the wgmma build, written straight into GEMM2's wgmma B layout in smem/L2 with per-stage ready counters (no separate activation kernel); GEMM2 writes fp32 y plus a per-pair sum of squares; one grid barrier, then norm + routing-weight sum. Everything (route, both GEMMs, norm) is ONE kernel launch; ours is 7 launches.
Numerics: fp32 accumulate, y kept in fp32 (no bf16 rounding between GEMM and norm), global scale applied in fp32. This is the kernel-judge tolerance, NOT stock-Marlin parity, so it cannot be dropped in for serving without breaking our v2 write-out rounding.
