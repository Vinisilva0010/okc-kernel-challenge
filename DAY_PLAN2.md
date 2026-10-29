# Plan (Oct 29, afternoon): the MK MoE path inside SGLang = serving patch v0

Context: sub_mk2.py (job 5ea1d8aa3b604660) runs the full MoE on SGLang's Marlin tensors at 1.48x vs stock on verify16 / verify32,
1.33x verify1, 1.66x decode1, graph safe. Serving checks open Nov 5 noon PT (3 checks + 1 scored run per day), so everything must be
validated through phase-1 checks before then. Budget today: muse_moe_w4a16 has 6 checks until 21:00 Brasilia; use at most 4.
All rules in CLAUDE.md apply. Do not run any `okc serve` command (they are not open and are denied anyway).

## Step 1: one source of truth for the MK MoE
Create muse_moe_w4a16/mk_moe/ with:
- mk_moe.py: the CUDA source, the host functions and a Python class MkMoE with
  build(device) (compile once per process), alloc(max_tokens, top_k, hidden, inter, device) (fixed buffers),
  forward(hidden_states, topk_ids, topk_weights, local_offset, local_count, w13, w13_s, w13_gs, w2, w2_s, w2_gs,
          norm_weight, norm_eps, weight_first) -> bf16 [M, hidden] (route sum after the post-expert norm).
- Generalize routing to ANY local_offset / local_count (ids outside [offset, offset+count) are not local), as each EP rank has a
  different expert range in serving. The phase-1 harness only uses local_start 0, so add a prepare()-time self-check (CHECK files only)
  that compares offset 0 and offset 96 (re-labelled ids, same weights) against each other, printing `MKSELF ...`.
- Compile: first find how the Marlin MoE kernel in serve/sglang (python/sglang/kernels/jit/..., moe_wna16_marlin) is JIT-built and
  loaded, and use the same mechanism for our .cuh/.cu source if it can take a new source file; if not, use torch's load_inline and write
  down why. Record the decision in SERVE_PATCH_REPORT.md.

## Step 2: phase-1 check file built from that module
muse_moe_w4a16/sub_mk3.py = a generator script (tools/build_sub_mk3.py) that pastes mk_moe.py into a copy of sub_mk2.py's skeleton, with
prepare() reduced to what serving has: only _prepare_sglang's Marlin banks + norm_w + MkMoE.build/alloc (no v1 own-layout copies, no v1
extensions). kernel_fn: MK path for M <= 256, Marlin (v1's _kernel_sglang) above. Check it: expect 5/5 tests and per-case speedups
within ~3 % of sub_mk2 (1.377x overall), plus MKSELF lines showing offset 0 == offset 96 bit for bit.

## Step 3: the SGLang patch (serve/sglang, branch my-patch)
- Add python/sglang/srt/layers/moe/muse_mk_moe.py (generated from mk_moe.py by the same tool, never edited by hand).
- Read layers/muse_spark_fp4_moe.py (_forward_marlin, _marlin_expert_outputs, finalize / deferred-output logic, set_post_expert_norm)
  and layers/quantization/muse_spark_nvfp4_moe.py (_process_weights_for_marlin). Hook: after the Marlin repack, build MkMoE and keep a
  reference on the layer (one shared buffer set per device, not per layer). In _forward_marlin, when num_tokens <= 256 and the MK path is
  ready, return exactly what the stock path returns for that call (same dtype, shape and post-norm / deferred-gather contract), else stock.
  Use the layer's own _post_expert_norm_weight, eps and weight_first; the global scales are the layer's w13_weight_scale_2 / w2_weight_scale_2.
- Patch rules (serve/RULES.md): only files under python/sglang/, not model_loader/, source files only, no added line over 400 chars,
  <= 720 KB. Produce the diff with: cd /home/vnspo/okc && /home/vnspo/.okc/bin/okc serve diff serve  (allowed: it only writes patch.diff;
  if it is denied, use git -C serve/sglang diff > serve/patch.diff and say so). Check every rule on the diff with a small script.
- Python-compile every changed file (python3 -m py_compile). Nothing in SGLang can be run here (no GPU): say so in the report.

## Output
SERVE_PATCH_REPORT.md: what was hooked where (file + function), the contract checks, the JIT decision, the phase-1 check numbers,
the rule checks on patch.diff, and the open risks to test on Nov 5. Log every check in EXPERIMENTS.md. Commit + push after each step.
Do NOT submit anything.
