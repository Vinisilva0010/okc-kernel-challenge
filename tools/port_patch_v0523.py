"""Port the MK MoE + projection + hybrid-draft hooks onto SGLang v0.5.23 (297e54fd) in serve2/sglang.

usage: port_patch_v0523.py <full_hybrid|full|moe_only|moe_hybrid>
Every anchor must match exactly once or the script exits non-zero before anything is written.
Our modules are copied byte-identical from serve/sglang-a55ab7d5 (hash-checked against the old patch blobs).
"""
import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SG = ROOT / "serve2" / "sglang" / "python" / "sglang" / "srt"
OLD = ROOT / "serve" / "sglang-a55ab7d5" / "python" / "sglang" / "srt"
variant = sys.argv[1]
assert variant in ("full_hybrid", "full", "moe_only", "moe_hybrid")
PROJ_OFF = variant in ("moe_only", "moe_hybrid")
HYBRID = variant in ("full_hybrid", "moe_hybrid")

MODULES = {  # path -> git blob hash (identical in all three old diffs)
    "layers/moe/muse_mk_moe.py": "296aee8ebce3e366416b18249cf3d906292951a5",
    "layers/muse_proj_mk.py": "946fa58783d963f5881609569e1aa347e0f4f481",
    "speculative/hybrid_draft.py": "b6bb41f83250c3886341de044d3a90370daf7e57",
}
edits = {}  # path -> list of (old, new)


def edit(path, old, new):
    edits.setdefault(path, []).append((old, new))


# ---- environ.py
env_old = "    SGLANG_ENABLE_MUSE_SPARK_AG_NORM_FUSION = EnvBool(True)\n"
env_new = env_old + (
    "    # Serve routed experts of up to 256 tokens from the Marlin tensors with the MK kernels (moe/muse_mk_moe.py); off = stock Marlin MoE.\n"
    "    SGLANG_MUSE_MK_MOE = EnvBool(True)\n"
    "    # Serve the dense BF16 projections (qkv_proj, o_proj, pooler, router) of up to 256 tokens with the public compressed-weight kernels (layers/muse_proj_mk.py); off = stock.\n"
    "    SGLANG_MUSE_PROJ_MK = EnvBool(%s)%s\n"
) % (("False", "  # OFF by default in the moe-only variant") if PROJ_OFF else ("True", ""))
if HYBRID:
    env_new += (
        "    # Prompt-lookup drafts from the request's own tokens, merged into the DFlash block when the suffix match is long enough; off = DFlash drafts only.\n"
        "    SGLANG_MUSE_HYBRID_DRAFT = EnvBool(True)\n"
        "    # Shortest suffix match (tokens) that lets the lookup continuation replace the DFlash draft.\n"
        "    SGLANG_MUSE_HYBRID_MIN_MATCH = EnvInt(4)\n"
    )
edit("environ.py", env_old, env_new)

# ---- layers/muse_spark/muse_spark_fp4_moe.py (MK MoE bind + _forward_marlin path)
F = "layers/muse_spark/muse_spark_fp4_moe.py"
edit(F, "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING, Optional\n",
     "from __future__ import annotations\n\nimport logging\nfrom typing import TYPE_CHECKING, Optional\n")
edit(F, "from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig\n",
     "from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig\n"
     "from sglang.srt.layers.moe.muse_mk_moe import (\n    MK_EXPERTS,\n    MK_HIDDEN,\n    MK_INTER,\n    MK_MAX_TOKENS,\n    MK_TOP_K,\n    mk_shared,\n)\n")
edit(F, "from sglang.srt.layers.moe import MoeRunnerBackend, get_moe_runner_backend\n", "from sglang.srt.environ import envs\nfrom sglang.srt.layers.moe import MoeRunnerBackend, get_moe_runner_backend\n")
edit(F, "# 1 / (fp8_e4m3_max * 6)\n", "logger = logging.getLogger(__name__)\n\n# 1 / (fp8_e4m3_max * 6)\n")
edit(F, "        self.dispatcher = _DirectDispatcher()\n", "        self.dispatcher = _DirectDispatcher()\n        self._mk_moe = None\n        self._mk_layer = None\n")
edit(F, "    def prepare_input(\n",
     '''    @torch.no_grad()
    def bind_mk_moe(self) -> None:
        """Serve up to MK_MAX_TOKENS tokens from the repacked Marlin tensors with the MK kernels; stock Marlin serves the rest."""
        if not (envs.SGLANG_MUSE_MK_MOE.get() and self.use_marlin):
            return
        if (
            self.num_local_experts != MK_EXPERTS
            or self.hidden_size != MK_HIDDEN
            or self.intermediate_size_per_partition != MK_INTER
            or self.top_k != MK_TOP_K
        ):
            return
        try:
            mk_moe = mk_shared(self.w13_weight.device)
            mk_layer = mk_moe.bind_layer(
                self.w13_weight.data,
                self.w13_weight_scale.data,
                self.w13_weight_scale_2.data,
                self.w2_weight.data,
                self.w2_weight_scale.data,
                self.w2_weight_scale_2.data,
            )
        except Exception:
            # A failed JIT build or an unexpected tensor layout must not take the server down; stock Marlin serves.
            logger.warning("MK MoE unavailable, using the stock Marlin MoE", exc_info=True)
            return
        self._mk_moe = mk_moe
        self._mk_layer = mk_layer

    def prepare_input(
''')
edit(F, '''        if pre_expert_norm is not None:
            hidden_states = pre_expert_norm(hidden_states)
        expert_outputs, local_topk_ids = self._marlin_expert_outputs(
            hidden_states, topk_scores, topk_ids
        )
        if self._post_expert_norm is not post_expert_norm:
            self.set_post_expert_norm(post_expert_norm)
        if self._post_expert_norm_weight is not None:
''', '''        if pre_expert_norm is not None:
            hidden_states = pre_expert_norm(hidden_states)
        if self._post_expert_norm is not post_expert_norm:
            self.set_post_expert_norm(post_expert_norm)
        if (
            self._mk_moe is not None
            and self._post_expert_norm_weight is not None
            and hidden_states.dtype == torch.bfloat16
            and 0 < hidden_states.shape[0] <= MK_MAX_TOKENS
        ):
            return self._mk_moe.forward(
                hidden_states,
                topk_ids,
                topk_scores,
                self.local_expert_offset,
                self.num_local_experts,
                self._mk_layer,
                self._post_expert_norm_weight,
                self._post_expert_norm_eps,
                self._post_expert_norm_weight_first,
            )
        expert_outputs, local_topk_ids = self._marlin_expert_outputs(
            hidden_states, topk_scores, topk_ids
        )
        if self._post_expert_norm_weight is not None:
''')

# ---- Marlin weight prep hook
edit("layers/muse_spark/muse_spark_nvfp4_moe.py",
     "                delattr(layer, native)\n        return result\n",
     "                delattr(layer, native)\n        layer.bind_mk_moe()\n        return result\n")

# ---- linear method hook
edit("layers/quantization/unquant.py",
     "        bias: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:\n        if use_intel_amx_backend(layer):\n            x_shapes = x.shape\n",
     "        bias: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:\n"
     "        proj_mk = getattr(layer, \"_proj_mk\", None)\n        if proj_mk is not None and bias is None:\n"
     "            out = proj_mk.run(x)\n            if out is not None:\n                return out\n"
     "        if use_intel_amx_backend(layer):\n            x_shapes = x.shape\n")

# ---- models/muse_spark.py: router / pooler / unpooler / bind
M = "models/muse_spark.py"
edit(M, "        self._split2_key = None\n\n    @torch.no_grad()\n    def _build_split2",
     "        self._split2_key = None\n        self._proj_mk = None\n\n    @torch.no_grad()\n    def _build_split2")
edit(M, "        self._build_split2()\n        x = hidden_states.to(torch.bfloat16)\n        acc = torch.mm(x, self._split2_hi",
     "        x = hidden_states.to(torch.bfloat16)\n        if self._proj_mk is not None:\n            out = self._proj_mk.run(x)\n            if out is not None:\n                return out\n"
     "        self._build_split2()\n        acc = torch.mm(x, self._split2_hi")
edit(M, "            self.hidden_size, self.num_experts, dtype=router_dtype\n        )\n\n        if self.expert_hidden_size != self.hidden_size:",
     "            self.hidden_size, self.num_experts, dtype=router_dtype\n        )\n        self._proj_pool = None\n        self._proj_unpool = None\n\n        if self.expert_hidden_size != self.hidden_size:")
edit(M, "            output = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)\n        return output\n",
     "            unpooled = (\n                None if self._proj_unpool is None else self._proj_unpool.run(output)\n            )\n"
     "            if unpooled is None:\n                unpooled = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)\n            output = unpooled\n        return output\n")
edit(M, "            else:\n                expert_input = bf16_gemm_dispatch(\n                    hidden_states, self.pre_expert_proj.weight, None\n                )\n",
     "            else:\n                expert_input = (\n                    None if self._proj_pool is None else self._proj_pool.run(hidden_states)\n                )\n"
     "                if expert_input is None:\n                    expert_input = bf16_gemm_dispatch(\n                        hidden_states, self.pre_expert_proj.weight, None\n                    )\n")
edit(M, "                module.experts.prepare_trtllm_weights(module.experts.post_expert_norm)\n\n\n__all__ = [",
     '''                module.experts.prepare_trtllm_weights(module.experts.post_expert_norm)
        self._bind_proj_mk()

    @torch.no_grad()
    def _bind_proj_mk(self) -> None:
        """Bind the public compressed-weight projection kernels once per layer, after loading (never during CUDA-graph capture).

        Stock weights are kept for prefill and for every shape the kernels do not serve; any failure leaves the stock path."""
        if not (envs.SGLANG_MUSE_PROJ_MK.get() and is_cuda() and not _is_cpu):
            return
        try:
            from sglang.srt.layers.muse_proj_mk import bind_projection

            handles = []

            def bind(name, weight):
                handle = bind_projection(name, weight.data)
                if handle is not None:
                    handles.append(handle)
                return handle

            for layer in self.model.language_model.layers:
                attn = layer.self_attn
                attn.qkv_proj._proj_mk = bind("qkv_gate", attn.qkv_proj.weight)
                attn.o_proj._proj_mk = bind("o", attn.o_proj.weight)
                mlp = layer.mlp
                if isinstance(mlp, MuseSpark1xMoE):
                    if mlp.pre_expert_proj is not None:
                        mlp._proj_pool = bind("pooler", mlp.pre_expert_proj.weight)
                        mlp._proj_unpool = bind("unpooler", mlp.post_expert_proj.weight)
                    mlp.gate._proj_mk = bind("router", mlp.gate.weight)
            torch.cuda.empty_cache()
            logger.info(
                "Muse projection kernels bound: %d weights, %.2f GB extra per GPU",
                len(handles),
                sum(h.extra_bytes for h in handles) / 2**30,
            )
        except Exception:
            logger.warning("Muse projection kernels unavailable, using the stock path", exc_info=True)


__all__ = [''')

# ---- DFlash worker (hybrid variant only)
if HYBRID:
    W = "speculative/dflash_worker_v2.py"
    edit(W, "from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2\n",
         "from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2\nfrom sglang.srt.speculative.hybrid_draft import build_hybrid_drafter\n")
    edit(W, "        self._target_logits_processor = self._defer_target_verify_gather()\n\n    def _defer_target_verify_gather",
         "        self._target_logits_processor = self._defer_target_verify_gather()\n        self._hybrid_drafter = None\n\n    def _defer_target_verify_gather")
    edit(W, "                self._warm_eager_selector()\n\n    def _warm_eager_selector(self) -> None:\n",
         "                self._warm_eager_selector()\n        self._init_hybrid_draft()\n\n"
         + '''    def _init_hybrid_draft(self) -> None:
        # Draft sampler q rows (selector / lilicorr) and simulated acceptance need the DFlash candidates untouched.
        if (
            not envs.SGLANG_MUSE_HYBRID_DRAFT.get()
            or self.selector is not None
            or self.lilicorr is not None
            or SIMULATE_ACC_LEN > 0
        ):
            return
        self._hybrid_drafter = build_hybrid_drafter(
            model_runner=self.model_runner,
            block_size=int(self.block_size),
            device=self.device,
            min_match=envs.SGLANG_MUSE_HYBRID_MIN_MATCH.get(),
        )

    def _disable_hybrid_draft(self, exc: Exception) -> None:
        logger.warning("hybrid draft disabled, using DFlash drafts only: %s", exc)
        self._hybrid_drafter = None

    def _hybrid_observe_prefill(self, *, batch: ScheduleBatch, next_token_ids) -> None:
        if self._hybrid_drafter is None:
            return
        try:
            self._hybrid_drafter.observe_prefill(
                reqs=batch.reqs,
                prefix_lens=batch.prefix_lens,
                extend_lens=batch.extend_lens,
                slots_gpu=batch.req_pool_indices,
                seq_lens_gpu=batch.seq_lens,
                next_token_ids=next_token_ids,
            )
        except Exception as e:  # noqa: BLE001
            self._disable_hybrid_draft(e)

    def _hybrid_merge(self, *, batch: ScheduleBatch, draft_tokens, prefix_lens) -> None:
        if self._hybrid_drafter is None:
            return
        try:
            self._hybrid_drafter.begin_step(len(batch.seq_lens))
            if self._selector_sample is None:
                self._hybrid_drafter.merge(
                    draft_tokens=draft_tokens,
                    slots=batch.req_pool_indices,
                    prefix_lens=prefix_lens,
                )
        except Exception as e:  # noqa: BLE001
            self._disable_hybrid_draft(e)

    def _hybrid_record(self, *, batch: ScheduleBatch, out_tokens, commit_lens, prefix_lens) -> None:
        if self._hybrid_drafter is None:
            return
        try:
            self._hybrid_drafter.record(
                out_tokens=out_tokens,
                commit_lens=commit_lens,
                slots=batch.req_pool_indices,
                prefix_lens=prefix_lens,
            )
            self._hybrid_drafter.maybe_log(is_rank0=get_parallel().tp_rank == 0)
        except Exception as e:  # noqa: BLE001
            self._disable_hybrid_draft(e)

    def _warm_eager_selector(self) -> None:
''')
    edit(W, "                seq_lens=new_seq_lens,\n            )\n            return batch_output\n\n        # Decode / target-verify stage.",
         "                seq_lens=new_seq_lens,\n            )\n            self._hybrid_observe_prefill(batch=batch, next_token_ids=next_token_ids)\n            return batch_output\n\n        # Decode / target-verify stage.")
    edit(W, "        draft_tokens[:, 1:].copy_(draft_next)\n",
         "        draft_tokens[:, 1:].copy_(draft_next)\n        self._hybrid_merge(batch=batch, draft_tokens=draft_tokens, prefix_lens=prefix_lens)\n")
    edit(W, "        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(\n            sampling_info,\n            next_token_logits=logits_output.next_token_logits,\n            draft_input=draft_input,\n            draft_token_num=int(self.block_size),",
         "        self._hybrid_record(\n            batch=batch, out_tokens=out_tokens, commit_lens=commit_lens, prefix_lens=prefix_lens\n        )\n"
         "        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(\n            sampling_info,\n            next_token_logits=logits_output.next_token_logits,\n            draft_input=draft_input,\n            draft_token_num=int(self.block_size),")

# ---- phase 1: verify every anchor, no writes
edits = {k: [e for e in v if e is not None] for k, v in edits.items()}
bad = False
new_text = {}
for path, lst in edits.items():
    text = (SG / path).read_text()
    for old, new in lst:
        n = text.count(old)
        if n != 1:
            print(f"ANCHOR FAIL ({n} matches) in {path}: {old[:90]!r}")
            bad = True
        else:
            text = text.replace(old, new)
    new_text[path] = text
need = ["layers/moe/muse_mk_moe.py", "layers/muse_proj_mk.py"] + (["speculative/hybrid_draft.py"] if HYBRID else [])
for m in need:
    data = (OLD / m).read_bytes()
    h = subprocess.run(["git", "hash-object", "--stdin"], input=data, capture_output=True, check=True).stdout.decode().strip()
    if h != MODULES[m]:
        print("MODULE HASH FAIL", m, h)
        bad = True
    if (SG / m).exists():
        print("module already exists", m)
        bad = True
if bad:
    sys.exit(1)
# ---- phase 2: write
for path, text in new_text.items():
    (SG / path).write_text(text)
for m in need:
    (SG / m).write_bytes((OLD / m).read_bytes())
print("ported", variant, "files:", len(new_text), "modules:", len(need))
