# Preflight report (Nov 3): the serving patch inside the judge image

Question: does the patched SGLang code (MK MoE + public bf16 projection kernels) work when it meets the real SGLang classes, before
the serving track opens on Nov 5? The phase-1 judge runs in the serving image (torch 2.14.1+cu130, H200), so it was used as a smoke test.
Every number below is from a judge log in `muse_moe_w4a16/profiles/` or `muse_bf16_proj/profiles/` (jobs listed). Nothing was submitted
(no `okc submit` / `withdraw` / `serve`).

## How the test works (and the one thing that could not be done as asked)

`tools/build_preflight.py` writes two CHECK-only files (never to be submitted): `muse_moe_w4a16/preflight_moe1.py` (= `sub_mk11.py` plus
the embedded patched texts) and `muse_bf16_proj/preflight_proj1.py` (= `sub_projmk1.py` plus the embedded patched texts). Their
`prepare()` prints compact `PREFLIGHT ...` lines. The patched texts are taken from `serve/sglang` (zlib+base64 inside the file).

**Same-name loading over the installed modules is rejected by the judge.** The first submission got `rejected: edits loaded modules`
(a static rule in TASK.md; no job was created). I did not look for a way around it. The final files exec the patched text in FRESH
namespaces: nothing already loaded is replaced or modified (no module-table writes, no attribute writes on modules or `Envs`, no stubs of
server globals). Two kinds of lines in the embedded copies are redirected at generation time, nothing else:
cross-file imports of the new modules (`muse_mk_moe`, `muse_spark_nvfp4_moe`, `muse_proj_mk`) become reads from the other namespace, and the
two `envs.SGLANG_MUSE_*.get()` reads become an environment read with the same default (the installed `Envs` class lacks those fields). So
what ran is the patched code, not byte-identical to the patch on those few lines; the `environ.py` field definitions themselves were only
compile-checked.

## What was tested, and the result

Image check: installed `environ.py`, `muse_spark_fp4_moe.py`, `muse_spark_nvfp4_moe.py`, `unquant.py`, `muse_spark.py` are byte-equal (sha1)
to the a55ab7d5 blobs the patch is built on; `muse_mk_moe.py` / `muse_proj_mk.py` are absent as expected. Both variants also apply
cleanly (`git apply --check`) to a pristine export of a55ab7d5.

### MoE (jobs 2ce6bc2c5e2247c0, 5a2a67517c03411a, bb7c6a13c2ea474e; 3 of 4 budgeted checks used)
- Patched `muse_mk_moe.py`, `muse_spark_nvfp4_moe.py`, `muse_spark_fp4_moe.py` run. A `MuseSpark1xFp4Experts` built without `__init__`
  (that needs the distributed groups) with native HF-layout tensors goes through the REAL
  `MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading` (Marlin repack, then the new `layer.bind_mk_moe()`): bound, no stubs needed.
  A second layer shares the one `MkMoE` (scratch 160 MB per GPU in total).
- Patched `_forward_marlin` vs the same layer's stock Marlin path (run 3, Zipf-like routing, about M/1.4 local routes): rel L2 0 / 2.9e-5 /
  5.7e-5 / 1.5e-4 / 3.3e-4 at M = 1 / 8 / 32 / 128 / 256; M = 257 uses stock (exact). Stress routing (every route local) 3.9e-4 / 3.8e-4
  at M = 128 / 256. Standard norm form (weight_first False) 1.4e-4. A layer with up/gate global-scale ratio 1.7 (SGLang logs its own
  "must match" warning; the post-norm cancels the gain) 0 / 3.2e-4. Deterministic and bitwise equal to the phase-1 MK output everywhere.
- CUDA-graph capture of `_forward_marlin` at M = 8/32/128/256: capture ok, replay bitwise equal to eager.
- Fallbacks: `SGLANG_MUSE_MK_MOE=0` -> not bound; forced build failure -> warning "MK MoE unavailable, using the stock Marlin MoE", no
  exception, stock stays.
- Graph-replay time in the SGLang call path, patched vs stock Marlin (us): M=1 31.5 vs 75.3; M=8 75.4 vs 119.0; M=32 144.4 vs 165.0;
  M=128 253.1 vs 439.2; M=256 351.9 vs 721.6 (same inputs, L2 not flushed, a synthetic route mix, one layer: indicative, not a score).

### Projections (job 08b226916ad2469e; 1 of 2 budgeted checks used)
- Patched `muse_proj_mk.py`, `unquant.py` and `muse_spark.py` all exec as whole modules. `UnquantizedLinearMethod.apply` with
  qkv_gate [2304,8192] and o [8192,1024]: routed buckets vs `F.linear` rel 5e-6..1.4e-4 (0.02-0.24 % of bf16 values differ); qkv m <= 128
  and o m <= 32 take MK, everything else is exactly stock; 3-D input and a non-16-byte-aligned view return None (stock); a result survives
  the next call (no aliasing).
- Router `forward_split2` vs stock split2: rel 1.0-1.8e-6 for m = 16..256 (fp32 low bits differ), deterministic; m = 8 and 257 are stock.
  Pooler hook vs `F.linear`: rel 8e-5..1.6e-4 for m = 8..128 (m = 33 exact), m = 256 stock. Unpooler is not bound by default.
- The real `_bind_proj_mk` on a fake 2-layer model (patched `MuseSpark1xMoE` / `MuseSpark1xRouter` classes): qkv, o, pooler, router bound,
  unpooler not; "8 weights, 0.22 GB extra per GPU" for 2 layers = 0.11 GB per layer, 6.8 GB over 62 layers. Forced failure -> warning and
  stock, no exception. `SGLANG_MUSE_PROJ_MK=0` -> nothing bound.
- CUDA-graph capture of qkv + o + pooler + router at M = 8/32/128/256: capture ok, replay bitwise equal to eager.
- Side-stream race: the router on a side stream next to pooler and qkv on the main stream (as `MuseSpark1xMoE.forward` does), 200 rounds:
  0 mismatches vs serial.
- Eager timings in that log are launch-bound and not a performance claim.

## Not testable outside a server (stated, not faked)
- `MuseSpark1xFp4Experts.__init__` (distributed groups, backend resolution) and the real weight-loading order around `post_load_weights`.
- The MoE block's call sites in `muse_spark.py` (`_forward_experts`: pooler hook; `_reduce_and_unpool`: unpooler hook; the `gemm_ag` /
  `gemm_local` branches): they need `forward_batch`, the comm object and tensor-parallel state. Only compile-checked.
- Eight ranks compiling the two `load_inline` extensions at once, and the wall time of `bind` over 62 layers.
- Capture inside SGLang's own graph runner (multi-stream fork/join with the real streams), DFlash speculative decoding, hierarchical cache,
  and the real output-agreement test against stock.

## Open risks for Nov 5, most likely first
1. **Output agreement with stock.** MK MoE differs from Marlin by about 1e-4..4e-4 rel per layer, the projections by 1e-5..2e-4, over 62
   layers, and the judge asks for "nearly as well as stock agrees with itself". Stock Marlin is not deterministic itself (M = 257 run to
   run: det False; stock vs phase-1 Marlin 1.4e-4 in check 2). Nothing here says whether the accumulated difference passes. Flags default
   on and the judge cannot set env vars, so the fallback is the other patch file.
2. **Multi-rank compile / bind time.** Eight ranks build two extensions at load; a lock wait or compile failure is caught (stock fallback,
   warning), but a hang is not. Compile time counts toward the 90-minute limit, not the score. The bind-time selftest runs every kernel
   variant once per weight and layer; its total time over 62 layers is unmeasured.
3. **Memory.** 6.8 GB (projections) + 0.16 GB (MK scratch) per GPU, allocated in `post_load_weights`, come out of the KV pool at
   `--mem-fraction-static 0.85` with the hierarchical cache. Effect on KV capacity and TTFT at concurrency 32 is unknown; the log line
   "Muse projection kernels bound ... GB extra per GPU" gives the real number.
4. **Skewed routing.** MK is slower than Marlin when many tokens share the same experts (check 2, all tokens on the same 8 experts:
   M = 128 680 vs 487 us, M = 256 1323 vs 763 us). Phase-1 routings (hot expert 110-201 tokens of 256) still gave 1.49x, but agentic
   coding traffic across 62 layers may have worse layers. No guard exists.
5. **Untested call sites** (section above): a wrong shape or stream assumption at the pooler / unpooler hooks would show only in a server.
6. **Small gain from the projections.** From the earlier PROJSRV graph-replay study (cold L2, job 8672b928): o_proj about 1.08x, qkv 1.2-1.4x,
   pooler 1.1-1.3x, router up to 2x; phase-1 total 1.443x; the per-step saving (0.2-1.3 ms) is derived, not measured. The 6.8 GB is only
   worth it if the MoE side passes.

## Patch variants (same tree, both rule-checked: RULES OK, apply cleanly to a55ab7d5, py_compile OK)
- `serve/patch_full.diff` (= `serve/patch.diff`, 216,082 bytes, 7 files): MoE + projections, both flags default on.
- `serve/patch_moe_only.diff` (216,129 bytes, 7 files): identical except `SGLANG_MUSE_PROJ_MK` defaults to off. The bind returns before
  importing `muse_proj_mk`, so the projection hooks are inert and no projection memory is allocated.
- Built by `tools/build_patch_variants.py` (anchored; restores the nested checkout byte for byte).

No patch bug was found by the preflight, so `serve/patch.diff` is unchanged. Unused preflight budget: 1 MoE and 1 projection check.

## Hybrid drafting (added Nov 3; plan in HYBRID_DRAFT_PLAN.md, file `serve/patch_full_hybrid.diff`)
`serve/patch_full_hybrid.diff` = MoE + projections + hybrid drafting (233,158 bytes, 9 files, RULES OK, applies cleanly to a55ab7d5, py_compile OK; built by `tools/build_hybrid_variant.py`). `patch.diff`, `patch_full.diff`, `patch_moe_only.diff` are unchanged. `SGLANG_MUSE_HYBRID_DRAFT=0` makes the hybrid inert but the judge cannot set env vars: the fallback is the other patch file.
Open risks of the hybrid (most likely first):
1. **No server run exists.** The unit check (job 3b892751442f4602) covers the kernels, the history writer, counters, graph capture and a synthetic loop, not the worker call sites. Unverified assumptions: `req.kv.req_pool_idx`, `req.rid`, `req.full_untruncated_fill_ids`, `batch.prefix_lens / extend_lens` hold what the code reads in the prefill branch; `batch.seq_lens` at decode is the committed prefix (bonus not included) under overlap scheduling; `req_to_token_pool.req_to_token.shape` = (rows, context). A wrong assumption makes the history stale, which costs acceptance only (verification decides correctness) or raises in a hook (caught, hybrid off with a warning); an async CUDA fault from a bad index would not be caught.
2. **Gain unknown, loss possible.** The lookup overrides DFlash tokens at positions 1..c when a >= 4-token suffix match exists. If DFlash is already accepting most of those positions, or short matches are common but wrong, tokens per step can drop. Real hit rate on the judge's multi-turn coding traffic is unmeasured. Evidence in a run: the `HYBRID_DRAFT` log line (tokens per step with hit vs miss).
3. **Output agreement.** Greedy outputs do not depend on the drafts in exact arithmetic, but the verify batch contents change (different accepted lengths, different KV write patterns), so batch-shape-dependent numerics can differ from stock a little more often; non-greedy runs draw different random streams. Same class of risk as item 1 above.
4. **Prefill cost.** Each new request writes its whole fill_ids (also the radix / hierarchical-cache prefix) with one H2D: a numpy conversion of the token list plus an index_put, not measured at 100k-token prompts; it sits in the TTFT path (limit 2x stock).
5. **Per-step cost.** merge measured 54.6 us per step at bs 32 (graph replay, 60 000-token histories) plus begin_step / record (a few tiny launches, unmeasured in a server); a decode step is milliseconds, so about 1 % if nothing else interferes.
6. Startup: the Triton kernels compile on each rank in `init_cuda_graphs` (seconds, counts toward the 90 minutes, not the score); failure disables the hybrid.
7. Memory: 17 MB per GPU (33 pool rows x 131 088 x 4 B) at the judge's flags.
