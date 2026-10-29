# muse_decode_layer plan (Oct 29, 15:30): from 1.438x (board, sub_v3.py, rank 576) toward 1.54x

Source: judge job b8797228d4e54e03 (sub_v3.py, 13/13 tests, overall 1.4375, CI 1.416-1.459). Load time = (46 x sliding + 16 x full) / 62 per layer; the problem score is the geometric mean over six loads of SGLang/ours.
The load formula reproduces the board (1.4376). All numbers below are derived from that job's table and kernel profile (kernels grouped by name; "unlisted" = the profile's "... N more" rows).

## 1. Per-load numbers (62-layer weighted average, us) and per-microsecond value
| load | SGLang | ours | speedup | sensitivity: overall score per us saved in this load |
|---|---|---|---|---|
| decode 1 | 173.7 | 109.5 | 1.586 | +0.152% |
| decode 16 | 314.4 | 210.1 | 1.497 | +0.079% |
| decode 32 | 361.7 | 281.1 | 1.287 | +0.059% |
| verify 1 | 265.7 | 163.0 | 1.630 | +0.102% |
| verify 16 | 578.5 | 431.9 | 1.339 | +0.039% |
| verify 32 | 838.2 | 633.6 | 1.323 | +0.026% |
Because the score is a geometric mean, one microsecond of the average layer is worth ~6x more at decode 1 than at verify 32 (value = 100 / (6 x our time) %). 1.54x needs +7.1% overall = e.g. -4% time on every load (-4.4 / -8.4 / -11 / -6.5 / -17 / -25 us).

## 2. Kernel-time split per case (us, GPU time from the kernel profile; groups: moe = Marlin/w4 kernels + align/act + epilogue; proj = cuBLAS + Triton GEMV + unpooler; norms = the two boundary kernels)
| case | total | moe | attention | projections | router | norms | unlisted |
|---|---|---|---|---|---|---|---|
| decode 1 sliding | 111.9 | 35.4 | 5.2 | 45.9 | 10.0 | 8.4 | 7.0 |
| decode 1 full | 118.5 | 35.3 | 15.0 | 45.3 | 7.3 | 7.9 | 7.7 |
| decode 16 sliding | 200.2 | 106.2 | 13.2 | 66.9 | 2.8 | 7.5 | 3.6 |
| decode 16 full | 280.4 | 92.4 | 106.6 | 67.5 | 2.6 | 7.5 | 3.8 |
| decode 32 sliding | 234.0 | 133.5 | 16.1 | 66.9 | 2.8 | 8.4 | 6.3 |
| decode 32 full | 459.0 | 178.6 | 196.5 | 67.0 | 2.7 | 7.8 | 6.4 |
| verify 1 sliding | 182.1 | 99.0 | 8.5 | 57.5 | 2.9 | 8.6 | 5.6 |
| verify 1 full | 179.5 | 85.0 | 20.7 | 57.2 | 2.8 | 8.4 | 5.4 |
| verify 16 sliding | 401.5 | 282.4 | 17.4 | 83.2 | 0 | 9.6 | 8.9 |
| verify 16 full | 536.1 | 298.1 | 139.5 | 80.6 | 0 | 4.9 | 13.0 |
| verify 32 sliding | 550.1 | 398.4 | 26.4 | 103.6 | 0 | 15.6 | 6.1 |
| verify 32 full | 816.1 | 421.1 | 262.9 | 104.6 | 0 | 16.4 | 11.1 |
(The router runs on a side stream and overlaps the pooler; measured case times are 3-5% below these sums.) The MoE is 32% (decode 1) to 72% (verify 32 sliding) of the GPU time; projections are 42% of decode 1 and a constant ~45-105 us elsewhere;
attention matters only in the full layers (decode 16/32: 107/197 us, verify 16/32: 140/263 us); norms/glue cost 8-16 us per case.

## 3. Pieces we already have, and where each is (or is not) faster
- MoE paths in sub_v3: T <= 16 the MoE-v4 small-M CUDA kernels (own repacked weights); T = 17..32 **Marlin** (incl. the scored T = 32); T = 33..256 the MoE-v4 large-M CUDA kernels. The MoE problem's board file (sub_v1 = sonnet_v5 "moe5", one template for every M <= 256) is +3.1% over v4 there (1.480x vs 1.436x), and v1 beats our MK path (mk_moe v2) at every M in the MoE problem
  (decode1 1.83 vs 1.58-1.70, verify1 1.49 vs 1.31-1.37, decode16 1.43 vs 1.25-1.29, decode32 1.10 vs 1.08-1.10, verify16 1.57 vs 1.47-1.52, verify32 1.57 vs 1.47-1.51).
  => mk_moe v2 is NOT a win for the layer at any M except as a replacement for Marlin at T = 32 (tie with v1/moe5 there); the layer's own v4 kernels are faster than v2-MK elsewhere. The only MK-side win would be the v3 hot-expert kernel at verify16/32 (validation starts 21:00 on the MoE problem's checks).
- Attention: the layer's `_attn_kernel`/`_full_cat_kernel`/`_ws_full_verify_kernel`/`_full_kernel` are sub_v3's copies of attention sub_v7..v10 features (header: v2 = Gluon verify-full, v3 = tail-only mask + 8-warp producer = attention sub_v10). Same kernels, already in.
- Boundary/glue: G2 pre-norm, G3 sliced unpooler, G4 epilogue grid are in (v5c).

## 4. Levers ranked by expected overall gain (kill line = judge noise +-1.5% on the changed loads)
1. **T = 32 MoE: Marlin -> the large-M CUDA path (LARGEM_MIN_TOKENS 33 -> 32), later the moe5 template.** MoE at decode 32 is 133 / 179 us of 234 / 459; the MoE problem shows ~8-9% MoE saving over Marlin at 32 tokens. Expect -12..-16 us on the weighted average => decode 32 load +4..6% => overall +0.25..0.35% per... computed: 14 us x 0.059 = +0.8%. One-line change; check 1.
2. **Port moe5 (MoE board file v1/v5) to replace v4's small-M and large-M kernels in the layer.** +3% MoE-wide: decode 1 -1.1 us (+0.17%), decode 16 -3 us (+0.24%), verify 1 -3 us (+0.3%), verify 16/32 -9/-12 us (+0.35/+0.3%) => overall about +1.3%. Effort: the moe5 extension (~500 lines of CUDA) must read local_start from the device tensor with the layer's wrap rule like v3's largem route does. Medium risk; self-check against the layer's Marlin experts exists.
3. **MoE v3 hot-expert kernel for T = 128 / 256** (needs the MK route with a device `local_start` plus the v3 kernel): MoE problem estimate verify32 -14..-24%, verify16 -4..-9% => 60-100 us / 15-25 us on the weighted layer => +1.5-2.6% / +0.6-1.0%. Only after the 21:00 validation; also only wins if it beats the layer's v4 large-M kernels, not Marlin.
4. **Projections at small T** (45-67 us per case, ~30-40% of decode/verify 1/16): qkv/o via cuBLAS nvjet at ~2.0-3.0 TB/s, pooler Triton GEMV 3.3 TB/s, router/unpool latency-bound (7.3 / 6.1 us for 8 MB). Floor for the 138 MB of per-layer projection weights is ~29 us at 4.8 TB/s vs 46 measured at decode 1. Candidate: Triton GEMV (already used for the pooler) for the qkv/o projections at T <= 16 where cuBLAS reaches only ~2 TB/s; expected -3..-6 us at decode 1/verify 1 (+0.5..0.9%). Needs a config sweep (one measure check).
5. **Boundary/glue fusion** (two norm-residual kernels 8 us, epilogue 4 us, routing 2.7 us, unlisted 4-13 us): -2..-4 us per case => +0.3..0.5% at decode 1 / verify 1 alone. High effort per us.
6. **Full-layer attention** (decode 32 full 197 us vs floor 126; verify 32 full 263 vs floor 127): -10% of the full kernels = decode 32 -5 us, verify 32 -6 us => +0.3% / +0.15%.
Reaching 1.54x needs most of 1-4 together; 1.505-1.52x is a realistic target for today.

## 5. Check plan (<= 15 layer checks; each logged in EXPERIMENTS.md; keep only wins with 13/13 tests)
1. sub_v4 = sub_v3 + LARGEM_MIN_TOKENS 32 (lever 1).   2. If the large-M path wins at 32: also try the Marlin knobs sweep for 17..31 is irrelevant (unscored) - skip.
3-6. Port moe5 (lever 2) in stages: small-M first (T <= 16), then the unified large-M, each with the layer's self-check.   7-9. Projection GEMV sweep (lever 4).   10+. Hot-kernel integration after 21:00 validation (lever 3), glue fusion, attention.
Adoption: overall >= 1.438 x 1.01 = 1.4525 with 13/13 -> READY_TO_SUBMIT.md (file, job, per-case table, rules checklist). Nothing is submitted.
