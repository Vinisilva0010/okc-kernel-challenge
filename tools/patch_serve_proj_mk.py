"""Anchored patch of the serve/sglang checkout: bind the public projection kernels (muse_proj_mk.py) next to the MK MoE.
Each anchor must match exactly once; the script exits non-zero otherwise. Idempotence is not attempted: run it on the base + MoE tree once."""
R = "/home/vnspo/okc/serve/sglang/python/sglang/srt/"


def patch(path, edits):
    s = open(R + path).read()
    for old, new in edits:
        assert s.count(old) == 1, f"{path}: anchor count {s.count(old)}: {old[:80]!r}"
        s = s.replace(old, new)
    open(R + path, "w").write(s)


patch("environ.py", [(
    "    SGLANG_MUSE_MK_MOE = EnvBool(True)\n",
    "    SGLANG_MUSE_MK_MOE = EnvBool(True)\n"
    "    # Serve the dense BF16 projections (qkv_proj, o_proj, pooler, router) of up to 256 tokens with the public compressed-weight kernels (layers/muse_proj_mk.py); off = stock.\n"
    "    SGLANG_MUSE_PROJ_MK = EnvBool(True)\n")])

patch("layers/quantization/unquant.py", [(
    "        bias: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:\n        if use_intel_amx_backend(layer):\n            x_shapes = x.shape\n",
    "        bias: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:\n"
    "        proj_mk = getattr(layer, \"_proj_mk\", None)\n"
    "        if proj_mk is not None and bias is None:\n"
    "            out = proj_mk.run(x)\n"
    "            if out is not None:\n"
    "                return out\n"
    "        if use_intel_amx_backend(layer):\n            x_shapes = x.shape\n")])

patch("models/muse_spark.py", [
    # router: hook before _build_split2 so the stock hi/lo copies are not built for the shapes the kernels serve
    ("        self._split2_hi = None\n        self._split2_lo = None\n        self._split2_key = None\n\n    @torch.no_grad()\n    def _build_split2",
     "        self._split2_hi = None\n        self._split2_lo = None\n        self._split2_key = None\n        self._proj_mk = None\n\n    @torch.no_grad()\n    def _build_split2"),
    ("    def forward_split2(self, hidden_states: torch.Tensor) -> torch.Tensor:\n        self._build_split2()\n        x = hidden_states.to(torch.bfloat16)\n",
     "    def forward_split2(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
     "        x = hidden_states.to(torch.bfloat16)\n"
     "        if self._proj_mk is not None:\n"
     "            out = self._proj_mk.run(x)\n"
     "            if out is not None:\n"
     "                return out\n"
     "        self._build_split2()\n"),
    # pooler / unpooler call sites
    ("        self.gate = MuseSpark1xRouter(\n            self.hidden_size, self.num_experts, dtype=router_dtype\n        )\n",
     "        self.gate = MuseSpark1xRouter(\n            self.hidden_size, self.num_experts, dtype=router_dtype\n        )\n"
     "        self._proj_pool = None\n        self._proj_unpool = None\n"),
    ("            else:\n                expert_input = bf16_gemm_dispatch(\n                    hidden_states, self.pre_expert_proj.weight, None\n                )\n",
     "            else:\n"
     "                expert_input = (\n"
     "                    None if self._proj_pool is None else self._proj_pool.run(hidden_states)\n"
     "                )\n"
     "                if expert_input is None:\n"
     "                    expert_input = bf16_gemm_dispatch(\n"
     "                        hidden_states, self.pre_expert_proj.weight, None\n"
     "                    )\n"),
    ("            output = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)\n        return output\n",
     "            unpooled = (\n                None if self._proj_unpool is None else self._proj_unpool.run(output)\n            )\n"
     "            if unpooled is None:\n"
     "                unpooled = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)\n"
     "            output = unpooled\n        return output\n"),
    # bind after the weights are loaded (same place as the other derived-weight setup)
    ("                module.experts.prepare_trtllm_weights(module.experts.post_expert_norm)\n",
     "                module.experts.prepare_trtllm_weights(module.experts.post_expert_norm)\n"
     "        self._bind_proj_mk()\n\n"
     "    @torch.no_grad()\n"
     "    def _bind_proj_mk(self) -> None:\n"
     "        \"\"\"Bind the public compressed-weight projection kernels once per layer, after loading (never during CUDA-graph capture).\n\n"
     "        Stock weights are kept for prefill and for every shape the kernels do not serve; any failure leaves the stock path.\"\"\"\n"
     "        if not (envs.SGLANG_MUSE_PROJ_MK.get() and is_cuda() and not _is_cpu):\n"
     "            return\n"
     "        try:\n"
     "            from sglang.srt.layers.muse_proj_mk import bind_projection\n\n"
     "            handles = []\n\n"
     "            def bind(name, weight):\n"
     "                handle = bind_projection(name, weight.data)\n"
     "                if handle is not None:\n"
     "                    handles.append(handle)\n"
     "                return handle\n\n"
     "            for layer in self.model.language_model.layers:\n"
     "                attn = layer.self_attn\n"
     "                attn.qkv_proj._proj_mk = bind(\"qkv_gate\", attn.qkv_proj.weight)\n"
     "                attn.o_proj._proj_mk = bind(\"o\", attn.o_proj.weight)\n"
     "                mlp = layer.mlp\n"
     "                if isinstance(mlp, MuseSpark1xMoE):\n"
     "                    if mlp.pre_expert_proj is not None:\n"
     "                        mlp._proj_pool = bind(\"pooler\", mlp.pre_expert_proj.weight)\n"
     "                        mlp._proj_unpool = bind(\"unpooler\", mlp.post_expert_proj.weight)\n"
     "                    mlp.gate._proj_mk = bind(\"router\", mlp.gate.weight)\n"
     "            torch.cuda.empty_cache()\n"
     "            logger.info(\n"
     "                \"Muse projection kernels bound: %d weights, %.2f GB extra per GPU\",\n"
     "                len(handles),\n"
     "                sum(h.extra_bytes for h in handles) / 2**30,\n"
     "            )\n"
     "        except Exception:\n"
     "            logger.warning(\"Muse projection kernels unavailable, using the stock path\", exc_info=True)\n"),
])
print("serve patched")
