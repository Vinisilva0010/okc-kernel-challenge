# Muse Spark Open Kernel Challenge: kernels and serving patch (team Vinisilva0010)

Work for the Muse Spark Open Kernel Challenge (Meta, Paradigm and Andromeda, openkernelchallenge.dev): make Meta's Muse Spark Open
(816B-parameter MoE, 256 experts, NVFP4 expert weights) run faster on NVIDIA H200, first as isolated kernels, then inside SGLang on a
full 8xH200 node.

## Tracks
- **Kernel track** (Oct 22 - Nov 4, 2026): five problems, each one Python file timed against SGLang's own kernel on one H200.
- **Serving track** (Nov 10 - Nov 19, 2026): a patch to SGLang at commit a55ab7d5, scored by throughput at stock interactivity on
  multi-turn agentic coding conversations at concurrency 1, 4, 16 and 32. Outputs must agree with stock SGLang nearly as well as
  stock agrees with itself, every request returns 220 tokens, and TTFT may be at most 2x stock.

## Kernel track: final board (Nov 5, 2026)
Overall rank 456 of about 1,000 teams.

| problem | speedup vs SGLang | source |
|---|---|---|
| muse_attention_decode | 2.580x | vjeux/okc-2026 (Apache-2.0), unchanged kernel code |
| muse_decode_layer | 2.136x | Madhavan113/kerneloptimizer (Apache-2.0), unchanged kernel code |
| muse_moe_w4a16 | 2.339x | Madhavan113/kerneloptimizer (Apache-2.0), unchanged kernel code |
| muse_bf16_proj | 1.500x | vjeux/okc-2026 (Apache-2.0), unchanged kernel code |

Our own kernels from that track (Gluon warp-specialized attention verify at 1.892x, infraax-based MoE and layer files, measurement
harnesses) are in the problem folders with their judge logs in `profiles/` and every experiment in `EXPERIMENTS.md`.

## Serving track: strategy
The serving judge decodes with DFlash speculative decoding (8 draft tokens), so every target step is a verify block of
8 x concurrency tokens (8, 32, 128, 256). Three independent levers, each behind its own switch with a fallback to stock SGLang
on any exception:

1. **MoE on SGLang's own Marlin tensors (`mk_moe`).** The routed experts are the largest share of a step from 16 users up.
   Kernels that repack NVFP4 weights into their own layout need a second copy of the experts (about 57 GB per GPU), which does
   not fit next to the KV cache, and prefill still needs Marlin. Our kernel reads the Marlin weight tiles and scales exactly as
   stored (layout decoded and validated against a torch reference), with a per-warp cp.async ring, deterministic split-K and
   Marlin's own write-out rounding, so it stays within 1e-4..1e-3 of stock while running 1.6x (8 tokens) to 2.0x (256 tokens)
   faster than stock Marlin in SGLang's call path. Prefill (more than 256 tokens) keeps stock Marlin.
2. **Public BF16 projection kernels (`proj_mk`)** for qkv, o, pooler and router at decode/verify sizes (vjeux/okc-2026,
   Apache-2.0): losslessly compressed copies verified bit for bit at load, a fixed per-shape kernel table for deterministic
   output, about 6.8 GB extra per GPU, within 1.5e-4 of SGLang's GEMMs.
3. **Prompt-lookup drafting merged into DFlash (`hybrid_draft`).** Agentic coding copies a lot of context. On the GPU, without
   host syncs, each request's longest suffix match in its own conversation replaces DFlash's draft tokens when the match is at
   least 4 tokens. Target verification is unchanged, so outputs do not change; only accepted tokens per step do. The rules allow
   drafting from the requests themselves (prompt lookup and n-grams).

Rejected with measurements: public and own attention kernels differ from SGLang's FA3 FP8 path by 3-13 % relative, too far for
the agreement rule; a shared-dequant hot-expert kernel and an INT-to-FMA instruction remix were slower.

## Method
- Measure before building: real routing distributions, per-kernel profiles and bandwidth limits drove every decision.
- Numerics parity as a first-class target: each serving component is compared with the exact SGLang path it replaces
  (relative L2, share of differing bf16 values, run-to-run determinism).
- One source of truth per component: the same module file is embedded in phase-1 check files and copied into the SGLang patch.
- Preflight inside the judge image: the patched SGLang classes were exercised with real weights, CUDA-graph capture and the
  fallbacks before the serving track opened.

## Layout
- `muse_*/`: kernel-track files, measurement scripts and judge profiles per problem.
- `muse_moe_w4a16/mk_moe/`, `muse_bf16_proj/proj_mk/`: serving modules (single source of truth).
- `serve/`: pinned SGLang checkout (branch `my-patch`) and patch variants (`patch_full.diff`, `patch_full_hybrid.diff`,
  `patch_moe_only.diff`).
- `tools/`: generators, patch-rule checker, preflight builders.
- `EXPERIMENTS.md`, `SERVE_PATCH_REPORT.md`, `PREFLIGHT_REPORT.md`, `HYBRID_DRAFT_PLAN.md`: logs and reports.

## Credits and licenses
All challenge submissions are licensed Apache-2.0. Code from vjeux/okc-2026, Madhavan113/kerneloptimizer and
infraax/okc-kernel-challenge is used under Apache-2.0 with attribution in each file header. Marlin dequantization bit tricks
come from SGLang (Apache-2.0). SGLang is Apache-2.0.
