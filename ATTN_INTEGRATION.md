# Attention integration: our phase-1 kernel inside SGLang's MuseSpark1xAttention (read-only analysis, Oct 29)

Scope: `MuseSpark1xAttention.gated_attention` on Hopper with the judge's flags, versus `muse_attention_decode/sub_v10.py` (board 1.892x).
Nothing was run (no GPU here) and no code was changed. File:line references are into `serve/sglang/python/sglang/` (base a55ab7d5, branch my-patch)
and into `muse_attention_decode/`. Statements I could not verify from source are marked **[unverified]**.

## 0. Summary

- With the judge's flags the backend is **FA3** (`FlashAttentionBackend`), **page size 1**, **FP8 E4M3 KV cache**, hybrid full/sliding-window KV pool (`SWAKVPool`, not the unified pool).
- Under DFLASH the target model's attention hot path is **target_verify** (8 tokens per request, causal chain, topk 1). Plain decode (q_len 1) only appears when a step runs without speculation.
  Both go through CUDA graphs (bs buckets <= 32); the backend fills persistent metadata buffers **outside** the graph (`init_forward_metadata_out_graph`), the captured kernels only read them.
- Every input our kernel needs exists in the backend's metadata or in the layer/pool, with **no host sync**: only two scalars differ in convention (`seq_lens` = cached tokens, so decode needs `cache_seqlens - 1`; scales are `None` in SGLang and must become a ones tensor).
- The hook point is the top of `gated_attention` (muse_spark.py:453): one call replaces QK-norm + RoPE + KV store + FA3 + output norm/gate for decode / target_verify, everything else stays stock.
- The real work is not plumbing but **(a)** numerics parity with FA3's FP8 path (the serving rule in CLAUDE.md) and **(b)** making the split plan robust to a page table 131072 columns wide (phase-1 sized it from contexts near 80K).
  Estimate: 6 steps, about 8-10 phase-1 attention checks (17 of 20 left today), plus a patch whose in-SGLang behaviour cannot be tested until Nov 5.

## 1. Path of a target_verify and a decode step (judge flags)

### 1.1 Which backend, pool, page size
- Judge flags: no `--attention-backend`, no `--page-size`, `--speculative-algorithm DFLASH --speculative-num-draft-tokens 8`, `--context-length 131072`, `--cuda-graph-max-bs-decode 32`, `--max-running-requests 32`.
- Backend: `get_default_attn_backend` (`srt/arg_groups/model_override_base.py:276-311`): MHA model on Hopper with CUDA >= 12.3 and "no spec or topk 1" returns `"fa3"`. DFLASH forces `speculative_eagle_topk == 1`
  (`arg_groups/speculative_hook.py:342`). So `FlashAttentionBackend` (`layers/attention/flashattention_backend.py:130`) with `topk == 1`.
- KV dtype: `_muse_spark_overrides` (`arg_groups/overrides.py:316-332`) turns `kv_cache_dtype auto` into `fp8_e4m3`. `RadixAttention.k_scale/v_scale` stay `None` (`layers/radix_attention.py:136-137`; the model only loads scales if the checkpoint has them, `models/muse_spark.py:1646`).
- Pool: hybrid SWA => `SWAKVPool` (`mem_cache/swa_memory_pool.py:21`): two `MHATokenToKVPool`s (full layers, sliding layers), `layers_mapping` per layer, `full_to_swa_index_mapping` for index translation (`:241-244`).
  `enable_unified_memory` defaults to False (`arg_groups/fields/memory.py:78-85`), so the unified pool / `kv_index_translator.is_translating` path is off in the judge's run.
- Layer mix (config: 62 layers, 128 q heads, 16 kv heads, head_dim 64, window 2048, rope theta per layer): under TP8 each GPU has `tp_q_head_num` 16 and `tp_k_head_num` 2, matching `muse_attention_decode`'s shapes.
  Sliding layers: `sliding_window_size = config.sliding_window - 1 = 2047` (`models/muse_spark.py:300-304`) and RoPE; full layers: no window, `layer_rope_theta == 0` so no RoPE (`:322`).
  Softmax scale `layer.scaling = 64**-0.5 * qk_scale_factor` (5.473), and q is **not** multiplied by the factor (`:327-332`).

### 1.2 Model side: `gated_attention` (`models/muse_spark.py:453-483`)
1. `qkv.split([q_size, kv_size, kv_size, q_size])` => q, k, v, o_gate (`:459-461`). Layout per token: q (16*64), k (2*64), v (2*64), gate (16*64) = 2304 bf16, the same layout as `Inputs.qkv` in phase 1.
2. `FusedQKNorm(qkv, positions, cos_sin_cache, kv_store=...)` (`:463-476`). `kv_store` comes from `_fused_kv_store` (`:411-439`), which returns `None` unless the backend is `TRTLLMHAAttnBackend` (Blackwell). **On FA3, `kv_store is None`**:
   the fused op only norms (+RoPE) and returns bf16 q and k (`kernels/ops/muse_spark_v12/fused_qk_norm.py`: fp32 norm with `tl.rsqrt`, line 82-83; RoPE reads the bf16-rounded row, line 87; output stored as bf16).
3. `self.attn(q, k, v, forward_batch, save_kv_cache=True)` (`:481`) => `RadixAttention.forward` (`layers/radix_attention.py:~165-245`) allocates `output = empty_like(q)` and calls `get_attn_backend().forward(...)`, which dispatches on the mode
   (`layers/attention/base_attn_backend.py:262-306`): `forward_decode` for decode, `forward_extend` for target_verify (and prefill).
4. `self._fused_o_norm_gate(output, o_gate)` (`:482`): per-head RMSNorm (eps 1e-5, no weight) x sigmoid(gate), JIT CUDA kernel `rms_norm_gate` (Triton twin at `kernels/ops/muse_spark_v12/fused_norm_gate.py:44-94`).

### 1.3 Backend side, decode step (`forward_mode.is_decode()`, `spec_info is None`)
- Eager metadata `init_forward_metadata` (`flashattention_backend.py:839`, normal-decode branch ~`:942-960`) and, under graphs, `_init_full_cg_decode_metadata` (`:566`) -> `_apply_cuda_graph_metadata` (`:2880`, branch at `:2988-3040`):
  `cache_seqlens_int32 = seq_lens` (the length **including** the token being decoded), `cu_seqlens_q = arange(bs+1)`, `page_table` = `req_to_token` rows, `swa_page_table` = translated rows,
  all written by `_set_decode_page_metadata` (`:2840-2878`) -> `normal_decode_set_metadata(..., swa_page_table, token_to_kv_pool, skip_page_table=translated)`.
- `forward_decode` (`:1958`): writes k/v with `token_to_kv_pool.set_kv_buffer(layer, KVWriteLoc.for_layer(forward_batch, layer, swa_loc=self.forward_metadata.swa_out_cache_loc), k, v, layer.k_scale, layer.v_scale)` (`:1980-1996`);
  `prepare_paged_mha_query` (`:1357-1388`) casts q to fp8 (`q.to(self.kv_cache_dtype)`, no q descale; k/v descale only if scales exist); page table for SWA layers = `metadata.swa_page_table` (`:2133-2146`);
  `flash_attn_with_kvcache(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=1, softmax_scale=layer.scaling, causal=True, window_size=(2047, 0) | (-1,-1), num_splits=..., scheduler_metadata=<full layers only>)` (`:2160-2190`).
  `scheduler_metadata` is precomputed once per step for non-SWA layers (`:2163-2171`); sliding layers pass `None`.

### 1.4 Backend side, target_verify (DFLASH, 8 draft tokens, topk 1)
- Eager: `init_forward_metadata` target_verify branch (`:984-1013`): `cache_seqlens = seq_lens + num_draft`, `max_seq_len_q = num_draft`, `cu_seqlens_q = arange(0, bs*8+1, 8)`, `page_table = req_to_token[req_pool_indices, :max_seq_len_k]`.
- Ragged variant: if `spec_info.ragged_verify_layout` is not None (`:986-1001`, graph version `:3056-3067`) the per-request verify lengths vary (`build_ragged_target_verify_geometry`, `speculative/ragged_verify.py:235-253`).
  The default `SGLANG_RAGGED_VERIFY_MODE` is `"static"` (`environ.py:1419`), i.e. the fixed 8-per-request geometry; the judge sets no env for it. **[unverified]** whether "static" ever creates a layout object; the hook must fall back to stock whenever `ragged_verify_layout is not None`.
- Graph: `_init_full_cg_decode_metadata(in_capture=...)` (`:566`) handles decode / idle / target_verify / draft_extend. Buffers are bound once at capture by `_bind_metadata_buffers` (`:2605`; target_verify at `:2710-2730`):
  `cache_seqlens`, `cu_seqlens_q`, `cu_seqlens_k`, `page_table [max_bs, max_num_pages]`, `swa_page_table [max_bs, max_num_pages]` (allocated `:2478-2500`) and `swa_out_cache_loc = cuda_graph_swa_out_cache_loc[:num_tokens]`
  (refilled before every replay, `:580-594`). Each replay refills them on the device before `graph.replay()` (`_apply_cuda_graph_metadata`, target_verify branch `:3053-3073` via `_set_decode_page_metadata(plan, metadata, req_pool_indices, verify_lens, verify_delta=num_draft)`).
  Padding rows (bs bucket > real bs) carry `seq_len` fill value 1 (`get_cuda_graph_seq_len_fill_value`, `:3276`) and the pool's reserved slot 0 as write location.
- `forward_extend` (`:1390`) serves target_verify: stores k/v (`:1427-1463`), then the generic FA3 call with `causal=True`, `cu_seqlens_q = metadata.cu_seqlens_q`, `cache_seqlens = metadata.cache_seqlens_int32`, `max_seqlen_q = 8`;
  SWA layers use `metadata.swa_page_table` (`:1563-1570`) and window `(2047, 0)` (`:1470-1485`). Causality inside the block is plain bottom-right-aligned causal (no custom mask for topk 1), identical to `muse_attention_decode`'s "each token sees the cache and the block up to itself".
- Where speculation matters: with DFLASH on, the target runs target_verify steps; the draft model has its own (non-MuseSpark1xAttention) layers, so only `bs*8` verify tokens per step hit this module. Normal decode runs only for steps without a draft.

### 1.5 "page_sisliding layers' swa_page_table / index translation" (as asked)
- Full layers read `metadata.page_table` = `req_to_token[req_pool_indices, :]` (token slot ids in the **full** pool; with page size 1 a page id is the slot id).
- Sliding layers read `metadata.swa_page_table` = the same rows translated by `token_to_kv_pool.full_to_swa_index_mapping` (`swa_memory_pool.py:241`); the pool for those layers is `swa_kv_pool`, a different (smaller) buffer.
  In graphs the table is built on the device in `normal_decode_set_metadata` (`:2840-2878`), in eager mode it may be `None` and `forward_decode` translates on the fly (`:2139-2146`) => the hook must treat `swa_page_table is None` as "fall back".
- Write location for sliding layers is the translated `swa_out_cache_loc` (`:1980-1996`, `KVWriteLoc` at `mem_cache/memory_pool.py:1869`, `for_layer` `:1919`), for full layers `forward_batch.out_cache_loc`.
- Old, out-of-window slots may be freed and map to dummy values; the table is only read inside the window (FA3 by `window_size`, ours by `lo = seq - window + 1`), so no problem as long as we keep the same window arithmetic.

## 2. What our phase-1 kernel needs, and where each input comes from

`sub_v10.py:kernel_fn` (line 686) reads: `qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale, v_scale` and `config` (it ignores `cache_seqlens`, `cu_seqlens_q`, `scheduler_metadata`).
It does QK-norm (eps 1e-5, `_load_norm_rope` :133, `_prep_q` :147), interleaved RoPE (`:105-145`, own sincos), stores k/v as FP8 at `cache_loc` (`_own_head` :338), attends (sliding: `_attn_kernel` :197; full: `_full_kernel` :372 / `_full_cat_kernel` :480 /
Gluon warp-specialized verify `_ws_full_verify_kernel` :1229 for page 1, batch >= 16, 64 rows, sm_90), reduces the splits with the block's own keys and applies output RMSNorm x sigmoid(gate) (`_reduce_kernel` :578-625), returns bf16 `[B, L, Hq*64]`.

| kernel input (phase 1) | SGLang source at decode / target_verify | no-host-sync in a graph? | note |
|---|---|---|---|
| `qkv` bf16 [B,L,2304] | `gated_attention(positions, qkv, forward_batch)` argument, `[T, 2304]`, same split order q,k,v,gate | yes | `view(bs, L, -1)`; T = bs*L |
| `positions` int64 [B*L] | `gated_attention` argument (`forward_batch.positions`) | yes | decode: `seq_lens-1`; verify: `seq_lens+i` |
| `cache_loc` int64 [B*L] | full layers `forward_batch.out_cache_loc`; sliding layers `backend.forward_metadata.swa_out_cache_loc` (via `KVWriteLoc.for_layer`) | yes (persistent buffer) | graph padding rows write slot 0 (reserved dummy), harmless; our kernel writes it like the stock store does |
| `seq_lens` int32 [B] (cached tokens) | verify: `forward_batch.seq_lens.to(int32)` or `metadata.cache_seqlens_int32 - 8`; **decode: `metadata.cache_seqlens_int32 - 1`** | yes | one device-side elementwise op, or add a `seq_delta` constexpr to the kernel and read `cache_seqlens_int32` directly |
| `page_table` int32 [B, max_pages] | full: `metadata.page_table`; sliding: `metadata.swa_page_table` | yes (persistent buffer, `[:bs, :]` view) | page size 1 => entries are slot ids; width `max_num_pages` = 131072; `stride(0)` is constant per graph |
| `k_cache`, `v_cache` fp8 [pages, page_size, Hkv, 64] | `backend.get_paged_mha_kv_cache(layer)` (`:1322-1355`) = `pool.get_kv_buffer(layer_id)` through `paged_kv_view` (`mem_cache/layout/paged_view.py:28`) | yes | `SWAKVPool.get_kv_buffer` returns the full or the SWA sub-pool buffer per layer (`swa_memory_pool.py:225`) |
| `k_scale`, `v_scale` fp32 [] | `layer.k_scale/v_scale` are `None` | yes | allocate one persistent `ones(())` per device; semantics "scale 1" equals what SGLang does (`cache_k.div_` only runs when scale is not None, `memory_pool.py:2824-2830`) |
| `config.batch`, `q_len` | `forward_batch.batch_size` (the graph bucket) and `1` or `spec_info.draft_token_num`/`speculative_num_draft_tokens` (8) | host ints known at capture | bucket sizes beyond 1/16/32 are unmeasured (see 5) |
| `num_q_heads`, `num_kv_heads`, `head_dim` | `layer.tp_q_head_num` (16), `layer.tp_k_head_num` (2), `layer.head_dim` (64) | yes | |
| `window` | `layer.sliding_window_size + 1` if `> -1` else 0 (2048 / 0) | yes | `window_size=(2047,0)` in FA3 == "token and the 2047 before it" |
| `rope_theta` | `config.layer_rope_theta[layer_idx]` (0 on full layers) | yes | `self.rotary_emb.cos_sin_cache` is an fp32 table built as `t * inv_freq` then cos/sin (`rotary_embedding/base.py:193-206`), same argument rounding as our on-the-fly sincos |
| `sm_scale`, `qk_scale_factor` | `layer.scaling` already contains 5.473/8 and q is not scaled | yes | pass `qk_scale_factor=1.0`, `sm_scale=layer.scaling` (see 4.1) |
| RMS eps | `self.norm_eps = config.rms_norm_eps = 1e-5` (`configs/muse_spark.py:38`; the 1e-6 in the file belongs to the audio tower) | yes | our kernels hardcode 1e-5 => matches; add a guard `norm_eps == 1e-5` in the hook |
| gate | the 4th slice of `qkv` | yes | inside `qkv`, no extra input |

Output: bf16 `[T, Hq*64]` flattened, exactly what `_fused_o_norm_gate` returns, so `gated_attention` can return our tensor directly and `o_proj` is unchanged.

Graph-safety of `sub_v10.kernel_fn` itself, by reading it: all host work depends on shapes only (`page_table.shape[1]`, `triton.next_power_of_2`, `_plan*`, `torch.cuda.get_device_capability` which is cached by `_device_mode`),
lengths/positions/cache rows/scales are read on the device inside the kernels, no `.item()/.cpu()/.tolist()` anywhere; partial buffers are `torch.empty` (allowed during capture, lives in the graph pool; a preallocated shared workspace would be cleaner).
Triton/Gluon JIT compiles on the first launch: the decode runner does a warm-up forward before `capture_one` (`model_executor/runner/decode_cuda_graph_runner.py:1204-1289`), so compilation happens outside capture; the first eager launch must therefore also happen in warm-up (it does: `run_once` runs the model).
Things the phase-1 harness never exercised and a graph run will:
1. **Page table 131072 wide.** `span = page_table.shape[1] * page` for full layers (`kernel_fn` :703-705, `_plan_full` :677). The phase-1 judge builds tables sized to the 80K contexts; serving has 131072 columns and real contexts from a few hundred tokens to 100K+.
   Splits beyond `seq_len` exit early, but with a short real context a batch can end up with almost all programs idle and a single long split => slower than FA3 (whose `num_splits`/`scheduler_metadata` also use the static bound under graphs but FA3 splits on actual lengths inside the kernel). Needs a plan driven by the device-side `seq_lens` (e.g. in-kernel `split_len = ceil(seq/ns)`), see step 4.
2. **Padded graph rows** (bucket > real bs): `seq_lens = 1 - 1 = 0` after our decode delta, `page_table` rows stale/zero, `cache_loc = 0`. The kernel must produce finite garbage and not read out of bounds (all reads are guarded by `seq`/`end` in the code I read; the own-keys program still loads `qkv` rows of the padded tokens).
3. `use_ws` (the Gluon verify path) needs `batch >= 16`, `page == 1`, `rows == 64`: the graph buckets 24 and 32 (and any 17-31 bucket) were never timed in phase 1 (only 16 and 32); buckets < 16 use the Triton `_full_kernel`.
4. Only fixed `q_len` per call: ragged verify (`ragged_verify_layout`) must fall back to stock.

## 3. Where to hook

Replace the body of `MuseSpark1xAttention.gated_attention` for **decode and target_verify only** with a guarded fast path, same pattern as the MK MoE (bind once, never take the server down, stock fallback):

```
gated_attention(self, positions, qkv, forward_batch):
    if self._mk_attn is not None and self._mk_attn.eligible(self, forward_batch):   # host-only checks
        try:
            return self._mk_attn.forward(self, positions, qkv, forward_batch)       # returns bf16 [T, Hq*64]
        except Exception:                                                           # first-launch compile/shape errors only
            logger.warning("MK attention failed, using stock", exc_info=True); self._mk_attn = None
    ... existing body unchanged ...
```
- Bind (once per layer, at the end of `MuseSpark1xAttention.__init__` or from the first warm-up call): create/lookup the shared module and per-layer constants (window, theta, scaling, `ones` scales, eps check). A shared workspace per device (partials) like `mk_shared`.
- `eligible()` (all host-side, static per graph): env flag on (`SGLANG_MUSE_MK_ATTN`, new, default on), `forward_batch.forward_mode.is_decode() or is_target_verify()` (not idle, not extend/draft_extend), `backend` is `FlashAttentionBackend` with `kv_cache_dtype fp8_e4m3`,
  `pool.dtype == float8_e4m3fn`, `layer.k_scale is None and layer.v_scale is None`, `page_size == 1` (first version; add page > 1 later), `topk == 1`, `spec_info.ragged_verify_layout is None`, `head_dim == 64`, `tp_q_head_num/tp_k_head_num == 8`,
  `not is_cp_active(forward_batch)`, `forward_batch.dcp_kv_mask is None`, `not backend.kv_index_translator.is_translating`, not prefill-aware-SWA, and for sliding layers `metadata.swa_page_table is not None and metadata.swa_out_cache_loc is not None`.
  `forward_batch.forward_mode` and the backend attributes are Python objects: no tensor value is read, so the decision is identical at capture and at replay.
- Access to backend state: `get_attn_backend()` / `get_token_to_kv_pool()` (`model_executor/forward_context.py:66-70`) are already used by `_fused_kv_store`; `backend.forward_metadata` is the persistent `FlashAttentionMetadata` (decode and target_verify objects are bound at capture, `:2605`).
  A cleaner variant is a tiny backend method `get_mk_attn_inputs(layer, forward_batch)` returning (page_table, cache_loc, cache_seqlens, k_cache, v_cache) so the model file stays free of backend internals; the patch may touch `flashattention_backend.py` (it is under `python/sglang/`).
- Skipped entirely in the hook: `self.attn(...)` (RadixAttention) and `_fused_qk_norm`/`_fused_o_norm_gate`, because our kernel does norm, RoPE, KV store, attention and the gate itself. The KV pool contents after the call must be identical in layout to what the stock store writes (fp8 bytes at `loc`, k normed+roped, v raw) -
  the DFlash accept/reject path and `move_kv_cache` rely on that (`swa_memory_pool.py:325-330`).
- Prefill / extend, draft_extend, idle batches, mixed batches, any unsupported config: untouched stock path.

## 4. Numerics parity vs FA3 under the serving rule

Rule (CLAUDE.md, serving judge facts): outputs must agree with stock "nearly as well as it agrees with itself" and every request returns exactly 220 tokens. For MoE we found stock is bitwise reproducible with itself, and that matching its internal rounding points
(not just being more accurate) was what closed the gap (5.7e-3 -> 1e-4..1e-3 after reproducing Marlin's write-out). The same lesson applies; ordered by expected impact:

1. **FP8 P in FA3 [unverified].** FA3's FP8 forward converts the softmax probabilities to e4m3 before the PV product (that is how its FP8 kernel feeds wgmma). If so, stock's attention output carries a P-quantization error (relative ~2^-4 per probability, random) that our fp16-P kernel does not have.
   Our output is then *closer to the exact result* but different from stock by stock's own error, the same situation as Marlin. Cannot be fixed by tuning: it needs either emulating the e4m3 rounding of P inside our PV step (2 conversions per probability, the verify-full kernel is memory-bound so it may be affordable) or accepting the gap.
   First measure it: run baseline FA3 and an fp32 reference on the same inputs and print FA3-vs-fp32 and ours-vs-fp32 and ours-vs-FA3 rel L2 (design in 5, step 1).
2. **Query scaling and FP8 rounding of q.** Stock quantizes the *unscaled* normed q (bf16 -> e4m3, `prepare_paged_mha_query` :1357) and applies 5.473/8 as the softmax scale. Phase 1's kernel multiplies q by 5.473 *before* the FP8 cast (`_prep_q` :147-181, `scale` argument), which rounds a different set of values.
   Free fix: call the kernel with `qk_scale_factor=1.0` and `sm_scale=layer.scaling` (the kernel takes both as arguments, `kernel_fn` :719-760); the q.k products are then the same e4m3 numbers as FA3's.
3. **bf16 intermediates.** Stock rounds normed q and k to bf16 *before* RoPE (fused_qk_norm.py:87), rotates in fp32 from the rounded values, rounds to bf16 again, then q -> e4m3 and k -> e4m3 (store). Ours goes norm -> RoPE -> e4m3 in fp32 without those two roundings.
   Add a constexpr `BF16_STAGES` to `_load_norm_rope`/`_prep_q`/`_own_head` (round after norm, round after RoPE). Non-RoPE (full) layers: one bf16 rounding after the norm. v is bf16 already (`v_in` straight to e4m3) - identical.
4. **Attention output rounding.** FA3 writes bf16 `o`, the norm-gate kernel reads that. Our reducer normalizes the fp32 attention output (`_reduce_kernel` :621-624). Add a bf16 round of `o` before the RMS (cheap, in the reducer).
   Also the norm-gate formula: stock `rstd = 1/sqrt(var+eps)` (`fused_norm_gate.py:76`, the JIT CUDA twin is the one actually used) vs ours `tl.rsqrt`; sigmoid via `tl.sigmoid` vs `1/(1+exp(-g))`: differences ~1e-7 relative before the final bf16 rounding.
5. **Accumulation order / splits.** FA3 decode uses split-KV with its own heuristic and combines in fp32; ours uses other split counts. These differ at fp32-roundoff level (1e-6..1e-7) and are invisible next to items 1-4.
6. **RoPE tables.** Stock reads an fp32 table computed on the GPU as `cos(t * inv_freq)` in fp32 (`rotary_embedding/base.py:193-206`); ours evaluates sincos of the same fp32 product with a polynomial reduction: <= 1 ulp differences, relevant only where an e4m3 rounding boundary is hit (rare).
7. **Determinism.** Ours has no atomics in attention (splits are reduced in a fixed order by `_reduce_kernel`), so it should be run-to-run bitwise like FA3 (FA3 split-KV is also deterministic for fixed shapes). Confirm with the same two-calls-equal test used for the MoE.
8. **Padded graph rows / slot 0**: stock's fused path skips `loc == 0` (`fused_qk_norm.py` KV store mask); the backend's `store_cache` call uses `reserved_skip_index`. Writing slot 0 is harmless only if slot 0 is never read as a real key - it is the reserved padding slot, so keep that invariant and test it.

How to measure parity before any SGLang run (all on the judge via phase-1 checks): a CHECK-only measure file whose `kernel_fn` wrapper, on each shape's first call, clones the caches, runs `baseline.kernel_fn` (that is exactly the stock sequence:
FusedQKNorm -> `store_cache` -> `q.to(FP8)` -> `flash_attn_with_kvcache` -> `rms_norm_gate`; `muse_attention_decode/baseline.py:62-140`) on the clones, and prints `ATTNUM` lines: rel L2 and max |diff| of our output vs FA3, FA3 vs an fp32 reference, FA3 vs FA3 (two calls), ours vs ours,
fraction of differing bf16 elements, and the byte-difference fraction of the freshly written cache rows (k and v). Same structure as the MKNUM study. The submitted file contains none of it.

## 5. Work estimate, steps and the phase-1 checks that validate each

| step | work | validates with (muse_attention_decode, 17 of 20 checks left today) |
|---|---|---|
| 1. Parity study | measure file with `ATTNUM` (above): ours (sub_v10) vs FA3 vs fp32 reference on the judge's inputs for decode and verify, sliding and full | 1 check. Decides how much of 4.1-4.4 matters and whether FA3's P is fp8 |
| 2. Parity flags in the kernel | `qk_scale_factor=1.0 / sm_scale=layer.scaling` mode (already supported by arguments), `BF16_STAGES` rounding in norm/RoPE/own keys, bf16 round of `o` in the reducer; optional e4m3 round-trip of P in the Gluon PV step | 2-3 checks (flags first, P emulation second); each must keep the 16 tests passing and speed within ~2%; `ATTNUM` shows the gap closing |
| 3. Shared module + raw-argument entry | Move the kernels into `muse_attention_decode/attn_mk/attn_mk.py` (single source of truth like `mk_moe`), entry `forward(qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale, v_scale, cfg)` without the `data` dict, preallocated partial workspace, `seq_delta` for decode, fixed output buffer option; generator script emits `sub_v11.py` | 1-2 checks: 16/16 tests, per-case speed equal to sub_v10 (1.892x) |
| 4. Graph-realistic geometry | (a) page table 131072 wide with real lengths 1K-100K, ragged across the batch: in-kernel split length from device `seq_lens` (or a plan keyed on `max_context_len` with a cheap early-exit), (b) padded rows (seq 0/1, cache_loc 0), (c) buckets 2, 4, 8, 12, 24, (d) page size 1 only | 2 checks with a measure file that feeds the judge's `generate_input` shapes but pads the page table to 131072 and pads batches; compare time against FA3 at those shapes (`baseline.kernel_fn` in the same run) |
| 5. SGLang patch | `gated_attention` hook + eligibility + `get_mk_attn_inputs`, shared workspace, env flag, new `muse_mk_attn.py` generated from the module, patch-rule checker, py_compile | none possible on the judge; static checks only (rules script, reverse apply). In-SGLang behaviour is untestable until a serving check (Nov 5) |
| 6. Serving-side validation plan | what to read in the first serving check: graph capture succeeded (no fallback warning), 220-token requests, answer agreement vs stock, per-load speedup with and without `SGLANG_MUSE_MK_ATTN` (cannot toggle on the judge: decide by diffing two checks) | serving checks only (3/day) |

Rough effort: steps 1-3 about one working session each of the earlier MoE cycles (each ends in a check and a log entry), step 4 the riskiest (kernel plan change in Triton/Gluon code that took many checks to tune), step 5 about half a session.
Total 8-10 phase-1 attention checks, spread over 2 days at the 20/day limit. Gate before step 5: step 2 must show ours-vs-FA3 at the level of FA3-vs-FA3 across the verify and decode shapes, otherwise the exact-answer risk dominates any speed gain.

Expected value, honestly: phase-1 attention is 1.89x over FA3 on the isolated layer, but in serving attention is one slice of a step (MoE, the big projections, the norms and the draft model are the rest), and the sliding layers (46 of 62) are already cheap, so the step-level gain is a fraction of that;
it should be estimated from a serving profile (not available here) before investing in steps 4-6. The MoE patch is the larger and further-along lever.

## 6. Open questions that source reading could not settle
- Does FA3 (`sgl_kernel.flash_attn`) quantize P to e4m3 on the FP8 path, and with what scale? (decides 4.1)
- Does `SGLANG_RAGGED_VERIFY_MODE=static` ever produce a layout object, and what is the verify length distribution under DFLASH when drafts are rejected early? (decides whether the ragged fallback fires often)
- The real `page_table` width in the decode/verify graphs (`max_num_pages` at `init_cuda_graph_state`, `flashattention_backend.py:2310-2360`) given `--context-length 131072` and `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`; and the real context-length distribution of the hidden multi-turn agentic workload (drives step 4).
- Whether `swa_out_cache_loc`/`swa_page_table` are always populated for every graph bucket on this pool (they are bound in `_bind_metadata_buffers` for decode/target_verify, `:2640`, `:2724`) - expected yes since `use_sliding_window_kv_pool` is true for the hybrid pool.
