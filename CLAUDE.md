# OKC kernel challenge: working rules for the agent

You are working on the Muse Spark Open Kernel Challenge (Meta + Paradigm + Andromeda, openkernelchallenge.dev) for Vinicius
(GitHub Vinisilva0010). The goal is to WIN: prizes come only from the serving track (phase 2, opens Nov 5 noon PT), where we patch
SGLang. Phase 1 (kernel problems) closes Nov 4 11:59 PM PT. Work done on the MoE kernel counts for both: the phase-1 board and the
phase-2 SGLang patch.

## Hard rules (breaking any of these can disqualify the whole entry; organizers read top submissions by hand)
- kernel_fn must be CUDA-graph safe: no host syncs, no .item(), .cpu(), .tolist(), print or Python control flow on tensor values inside kernel_fn.
- No caching of outputs or input-dependent results across calls. prepare(weights) may only transform WEIGHTS (repack, layouts). Studies
  inside prepare() that only print measurements are fine in CHECK runs; never leave them in a file meant for submission.
- Precision: NVFP4 weights stay NVFP4, BF16 activations stay BF16 wherever SGLang keeps them BF16, FP8 KV cache stays FP8. Accumulate in FP32.
- Attention must attend to every key (no top-k / block skipping). MoE: 8 routed experts per token, every norm kept.
- Never run `okc submit`, `okc withdraw` or any `okc serve` command. If a file beats the board, write READY_TO_SUBMIT.md (see below).
- Never fabricate a number. Every number you write must come from a judge log you ran or that is in profiles/.

## How to run things
- PATH: `okc` lives in ~/.okc/bin. If `okc` is not found, run `export PATH="$HOME/.okc/bin:$PATH"` in the same command.
- `okc check <problem_dir>/<file>.py` (run it from ~/okc/<problem_dir>) runs tests + timing on the judge's H200, off the board.
  It takes 4-6 minutes: ALWAYS give the Bash tool timeout 600000. Output: speedup table + profiles/<file>.<job>.{txt,json,log}.
  The log keeps only the last ~16 KB of stdout: keep study prints compact and prefixed (e.g. `MK ...`), then grep them.
- Daily limits per problem: 20 checks and 10 runs; they reset at 00:00 UTC (21:00 Brasilia). `okc problems` shows what is left.
- Judge image: Python 3.12, PyTorch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0, H200 (sm_90). torch.utils.cpp_extension.load_inline works
  (nvcc present). Gluon (Triton experimental) works. No internet, no pip.

## Discipline (learned the hard way; follow it)
1. Before every check, write in EXPERIMENTS.md: the question it answers, the expected result range, and the decision each outcome leads to.
   After it, write the job id and the measured numbers. A check that cannot change a decision is not run.
2. Patch files with a Python script that asserts every anchor matches exactly once and exits non-zero otherwise, and CHAIN everything with
   `&&` (patch && syntax check && okc check) so a failed anchor never burns a check on an unmodified file.
3. Check registers and spills (cudaFuncGetAttributes: numRegs, localSizeBytes) for every CUDA variant; spills invalidate a timing.
4. "Twin" variants that remove work are only valid if the compiler cannot delete the remaining work (fold results into a stored value)
   and if the toolchain keeps the same load path (Triton drops cp.async when no dot consumes the tile: that twin was invalid).
5. Measure the real input distribution before estimating a gain (we wasted a day on an assumed MoE routing).
6. If two consecutive checks do not move the target metric, stop that line, write down why, move to the next task.
7. Commit after each experiment (English messages, `git commit -m "<type>(<scope>): <summary>" -m "<details>"`), push to origin main.
   Never force-push, never reset --hard.

## Current board (Oct 28 night): overall ~537th
| problem | board file | score |
|---|---|---|
| muse_attention_decode | muse_attention_decode/sub_v10.py | 1.892x |
| muse_decode_layer | muse_decode_layer/sub_v3.py | 1.438x |
| muse_moe_w4a16 | muse_moe_w4a16/sub_v1.py (infraax sonnet_v5, Apache-2.0) | 1.480x |
| muse_bf16_proj | muse_bf16_proj/sub_v1.py | 1.171x |

## MoE facts you need (all measured)
- Real local routing (tokens per active expert, EP8, 32 local experts): decode1 2-3 x 1; verify1 (8 tok) 5-8 experts max 6;
  decode16 10-14 experts max 5-10; decode32 16-21 max 5-14; verify16 (128 tok) 6-24 max 45-97; verify32 (256 tok) 23-25, hot expert 110-201.
- Phase 2 serves with DFlash speculative decoding, 8 draft tokens: every MoE call is a verify with 8 x concurrency tokens
  (concurrency 1/4/16/32 -> 8/32/128/256 tokens). Stock SGLang runs Marlin W4A16: ~1.1 TB/s on verify32 (574 us for ~650 MB).
- v1 (sub_v1.py) GEMM profile: verify32 GEMM1 263 us (1.65 TB/s) + GEMM2 190 us (1.14 TB/s); verify16 202 + 144; decode32 147 + 113.
- infraax (ref-infraax/) tried 3 warp-specialised builds (1 bulk-copy producer + 12 mma.sync consumers, mbarrier ring, 1 CTA/SM):
  ring floor ~2.4 TB/s, all slower than v5. Their WSTREAM pure-stream test reached 4.0 TB/s (layout is not the cap).
- Tuned Marlin config alone (diag_marlin_v1, job a2e50f73): 1.30x at verify1, ~1.01x at verify16/32. Config tuning is not the answer.

## Marlin NVFP4 layout (decoded and VALIDATED: rel L2 2.0e-3 vs a torch dequant of the raw weights)
In the phase-1 file, `state["banks"][(0, 32)]` (built by `_prepare_sglang` with SGLang's prepare_moe_nvfp4_layer_for_marlin) holds:
- `w13`: int32 [32, K/16, 2N] (GEMM1: K = 4096 hidden, N = 8192 = gate rows then up rows). `w2`: int32 [32, I/16, 2H] (GEMM2: K = 4096, N = 4096).
- Tile (k-tile kt of 16, n-tile nt of 64) = 128 ints = 512 contiguous bytes at int offset kt * 2N + nt * 128. Lane t loading the tile as
  uint4 at `nt*32 + t` (uint4 units) gets words w = 0..3; word w covers n-blocks j = 2w (n = 16w + t/4) and j = 2w+1 (n = 16w + 8 + t/4),
  as mma.sync m16n8k16 B fragments (k = 2(t%4)+{0,1} and +8). Block 2w comes from `dq(word << 8)`, block 2w+1 from `dq(word)`.
- `w13_s`: float8 [32, K/16, N] (processed). Lane t reads 8 bytes (uint2) at byte nt*64 + (t>>2)*8; byte order [j0,j2,j1,j3,j4,j6,j5,j7].
  `dqs(uint2.x)` -> fb[0] = (s_j0, s_j1), fb[1] = (s_j2, s_j3); `dqs(uint2.y)` -> (s_j4, s_j5), (s_j6, s_j7). The bf16 scale = e4m3 * 2^7.
- `w13_gs`: bf16 [32] = global * 2^119. Skip-flop dequant (no 2^126 multiply, as Marlin itself does): out = acc * float(gs_p). Validated.
- dq / dqs bit conversions: copied from sglang kernels/jit/csrc/gemm/marlin/dequant.h (Apache-2.0), see measure_mk_v2.py `mk_dq`, `mk_dqs`.

## MK study state (muse_moe_w4a16/measure_mk_v2.py, job 9eeae71b): GEMM1 only, real distributions, TB/s on weight bytes
| variant | regs | verify32 | verify16 | decode32 | verify1 | decode1 |
|---|---|---|---|---|---|---|
| REG_NT1D4 (register prefetch) | 145 | 1.23 | 1.73 | 2.00 | 1.62 | 1.53 |
| RING8 (per-warp cp.async ring, 8 stages, skip-flop) | 80 | 1.47 | 2.04 | 2.37 | 1.93 | 1.75 |
| RING4 | 78 | 1.46 | 2.04 | 2.37 | 1.97 | 1.81 |
| RING8_BIAS | 80 | 1.36 | 1.90 | 2.21 | 1.84 | 1.65 |
Reading: ring depth 4 = 8 -> no longer latency-bound; the instruction chain (dequant + scales + mma per 512 B) is the limit.
verify32 lags because the 201-token expert is dequantized 13 times (one 16-token group per warp item).

## Serving judge facts (from https://openkernelchallenge.dev/llms.txt, read Oct 29)
- "Your outputs have to agree with stock SGLang's nearly as well as it agrees with itself, and every request has to return exactly 220 tokens."
  So numerics must stay CLOSE to stock (tolerance = stock's own run-to-run agreement), not just "correct within the kernel tolerance".
- TTFT at most 2x stock at each load; compile time counts toward the 90-min run limit but not the score.
- Prizes on the serving board only: 1st $50k + signed H200, 2nd $30k, 3rd $10k, 4th $3k, 5th $2k, 6th-10th $1k.
- Kernel final review reruns top entries on the scored shapes AND nearby ones.
