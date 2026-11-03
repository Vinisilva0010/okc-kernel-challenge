"""Anchored patch of dflash_worker_v2.py for hybrid drafting (every anchor must match exactly once)."""
import pathlib, sys
p = pathlib.Path("/home/vnspo/okc/serve/sglang/python/sglang/srt/speculative/dflash_worker_v2.py")
t = p.read_text()
def once(old, new):
    global t
    n = t.count(old)
    if n != 1:
        sys.exit(f"ANCHOR x{n}: {old[:70]!r}")
    t = t.replace(old, new)
once("from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2\n",
     "from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2\nfrom sglang.srt.speculative.hybrid_draft import build_hybrid_drafter\n")
once("        self._target_logits_processor = self._defer_target_verify_gather()\n",
     "        self._target_logits_processor = self._defer_target_verify_gather()\n        self._hybrid_drafter = None\n")
once("""            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )
""", """            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )
        self._init_hybrid_draft()

    def _init_hybrid_draft(self) -> None:
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
""")
once("""                bonus_tokens=next_token_ids,
                seq_lens=new_seq_lens,
            )
            return batch_output
""", """                bonus_tokens=next_token_ids,
                seq_lens=new_seq_lens,
            )
            self._hybrid_observe_prefill(batch=batch, next_token_ids=next_token_ids)
            return batch_output
""")
once("        draft_tokens[:, 1:].copy_(draft_next)\n",
     "        draft_tokens[:, 1:].copy_(draft_next)\n        self._hybrid_merge(batch=batch, draft_tokens=draft_tokens, prefix_lens=prefix_lens)\n")
once("        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(\n",
     "        self._hybrid_record(\n            batch=batch, out_tokens=out_tokens, commit_lens=commit_lens, prefix_lens=prefix_lens\n        )\n        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(\n")
p.write_text(t)
print("patched")
