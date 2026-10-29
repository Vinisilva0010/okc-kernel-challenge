# READY_TO_SUBMIT (written Oct 29 evening; NOTHING has been submitted: no `okc submit` was run)

Candidates are the unchanged code of public Apache-2.0 entries plus an attribution header (credit: vjeux/okc-2026, and Madhavan113/kerneloptimizer for the layer and MoE).
Per-case time comparison showed none of our own paths is faster, so there is no shape dispatch to our code (details in PUBLIC_REFS.md). Each is NOT byte-identical to its public source (header lines added),
but the kernel code is identical: the judge's duplicate rule and the organizers' hand review should be considered by the human before submitting; an honest header is in each file.
All numbers below are from the `okc check` logs listed (profiles/sub_pub1.<job>.*).

## Summary
| problem | file | check job | tests | candidate | our board | gain |
|---|---|---|---|---|---|---|
| muse_attention_decode | muse_attention_decode/sub_pub1.py | 00884744e0ab4bbe | 16/16 | 2.602x | 1.892x (sub_v10) | +0.710x |
| muse_bf16_proj | muse_bf16_proj/sub_pub1.py | 65a6e80c32b04437 | 9/9 | 1.499x | 1.171x (sub_v1) | +0.328x |
| muse_decode_layer | muse_decode_layer/sub_pub1.py | 8e11ba740b75421d | 13/13 | 2.153x | 1.438x (sub_v3) | +0.715x |
| muse_moe_w4a16 | muse_moe_w4a16/sub_pub1.py | NOT CHECKED (0 checks left until 21:00 Brasilia) | - | - | 1.480x (sub_v1) | pending |

MoE: not ready. Next steps at 21:00: check ref_madhavan_moe.py, then sub_pub1.py, then fill this file.

## muse_attention_decode: `muse_attention_decode/sub_pub1.py`
Job 00884744e0ab4bbe, 16/16 tests, 2.602x (95% CI 2.563-2.660). Board file: sub_v10.py (job a095ad7c9ccb4136, 1.892x).

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

## muse_bf16_proj: `muse_bf16_proj/sub_pub1.py`
Job 65a6e80c32b04437, 9/9 tests, 1.499x (95% CI 1.476-1.521). Board file: sub_v1.py (job d4d1752631d84ec3, 1.171x).

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

## muse_decode_layer: `muse_decode_layer/sub_pub1.py`
Job 8e11ba740b75421d, 13/13 tests, 2.153x (95% CI 2.120-2.185). Board file: sub_v3.py (job b8797228d4e54e03, 1.438x).

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

## Rules checklist (CLAUDE.md hard rules). "scan" = grep of the file for .item/.cpu/.tolist/synchronize/print/caches, then reading the hits; it is not a full audit of 3-19k lines.
| rule | attention | bf16_proj | decode_layer |
|---|---|---|---|
| CUDA-graph safe kernel_fn (judge captures graphs; check passed) | yes (scan: no host sync in kernel_fn) | yes (`.item()` only in prepare helpers: build_patch_table, sk2_table) | yes (scan: no hits) |
| no caching of outputs / inputs | module caches hold device props, RoPE freqs, scratch workspace zeroed by the kernel (per its docstring) | `_autotune` in prepare picks per-(layer, m) kernels by timing on SYNTHETIC randn activations (weights only); kernel_fn only looks the choice up | scratch/workspace only (scan) |
| prepare only transforms weights | n/a (no prepare) | repacks + lossless 12-bit compression; BUT prepare also runs a timing autotune and a selftest, and prints `[okc] tune ...` / `selftest` lines to stderr. CLAUDE.md allows measurement-only studies in CHECK runs and says not to leave them in a submission: HUMAN DECISION whether to strip `_autotune` / prints (stripping changes the file, then re-check) | not scanned in depth |
| precision (NVFP4/BF16/FP8 kept, FP32 accumulate) | FP8 KV kept; queries rounded to E4M3 for QK^T (as FA3 does) and P in FP16 (E4M3 for the q_len 8 wgmma PV): matches SGLang's FA3 FP8 path per its docstring, tests 16/16 pass; NOT verified against SGLang's numerics beyond the judge tests | weights losslessly compressed (bit-exact), tests 9/9 | inherits attention + proj + MoE components; tests 13/13 |
| attention attends to every key; MoE 8 experts, norms kept | window/ctx range streamed fully per its docstring; sliding layers use the window | n/a | yes (same attention) |
| no submit/withdraw/serve run | yes | yes | yes |
| shape dispatch only (no input-value dispatch) | `_TUNED` keyed by (B, L, window, pairs); plan from shapes. Also: single-request builds read `page_table.shape[1]` as a speculation of seq len, checked on device against seq_lens (shape-derived, verified) | choice keyed by (layer, m) | shape-keyed (scan) |
Other notes for the human: (a) code identical to public entries rank ~93-141 now, so an identical-score entry will tie them; originality is by header only. (b) judge noise between same-code runs was 0-3% (PUBLIC_REFS.md).
