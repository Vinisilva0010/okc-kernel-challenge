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
