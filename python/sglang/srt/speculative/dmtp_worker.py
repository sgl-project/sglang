"""DMTP extension of the upstream DFlash verifier and commit pipeline."""

import logging
from copy import deepcopy

from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dmtp_runtime import scoped_config, validate_recipe

logger = logging.getLogger(__name__)


class DMTPWorkerV2(DFlashWorkerV2):
    def __init__(self, *args, **kwargs):
        parallel = get_parallel()
        if (
            parallel.pp_size != 1
            or parallel.tp_size != 4
            or parallel.attn_dp_enabled
            or parallel.attn_cp_size != 1
        ):
            raise ValueError("DMTP currently requires PP1/TP4 without attention DP/CP")
        validate_recipe()
        super().__init__(*args, **kwargs)
        self._dflash2_selector = self.selector is not None
        if (
            not hasattr(self.draft_model, "mtp_prefix_conditioner")
            or self.lilicorr is not None
            or self._is_domino
            or self.block_size != 8
            or self.use_compact_draft_cache
        ):
            raise ValueError(
                "DMTP requires a T1-residual checkpoint, seven proposals and shared-index DF KV"
            )
        import os

        cfg = self.draft_model.config.dflash_config
        if (
            cfg.get("mtp_injection_mode") != "t1_residual"
            or cfg.get("known_prefix_tokens") != 2
            or cfg.get("mtp_drop_anchor", False)
        ):
            raise ValueError("Only double-anchor residual at T1 is supported")
        if os.getenv("SGLANG_MTP_PREFIX_TARGET_ONLY", "0") != "0":
            raise ValueError("T1 residual requires native sampled-q verification")
        if os.getenv("SGLANG_MTP_PREFIX_HEAD_GRAPH", "0") != "0":
            raise ValueError("Legacy greedy head graphs are not supported")
        if os.getenv("SGLANG_MTP_PREFIX_HEAD_CONDITIONER_GRAPH", "0") not in ("0", "1"):
            raise ValueError("Sampled head-conditioner graph switch must be 0 or 1")
        self._single_anchor = False
        slots = 8
        self.draft_model.set_block_size(slots)
        from sglang.srt.speculative.draft_worker_common import (
            make_draft_block_spec_info,
        )

        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=slots, device=self.device
        )
        self._dflash_native_mtp_target_kv = False
        from sglang.srt.speculative.mtp_first_controller import MTPFirstController

        self._mtp_first = MTPFirstController(self)
        self._dmtp_controller = self._mtp_first
        logger.info(
            "DMTP upstream adapter: DF slots=%d, Target slots=8, proposals=7", slots
        )

    @property
    def tp_rank(self):
        return get_parallel().tp_rank

    def weight_update_runners(self):
        return [
            *super().weight_update_runners(),
            ("dmtp_mtp", self._mtp_first.native.worker.model_runner),
        ]

    @property
    def spec_v2_attn_backends(self):
        return (
            *super().spec_v2_attn_backends,
            self._mtp_first.native.worker.model_runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        super().alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )
        if (
            self.model_runner.runner_ids.is_translating
            or self.draft_model_runner.runner_ids.is_translating
        ):
            raise ValueError(
                "DMTP unified/translated KV pools are not validated; use the static pool"
            )
        self._mtp_first.native.alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )
        self._mtp_first.alloc_hidden_cache()

    def init_attention_backends(self):
        super().init_attention_backends()
        backend = self.draft_model_runner.attn_backend
        backend.speculative_num_draft_tokens = self._mtp_first.draft_slots
        self._mtp_first.configure_attention(backend)
        self._mtp_first.native.init_attention_backends()

    def _maybe_build_draft_sampler(self):
        # The native sampler would treat T1 as a DF proposal. DMTP's controller
        # keeps the explicit MTP T1 and samples only the six MASK hidden rows.
        return None

    def init_cuda_graphs(self):
        config = deepcopy(get_exec().graph.cuda_graph_config)
        config.decode.bs = list(range(1, get_schedule().max_running_requests + 1))
        with scoped_config(
            speculative_num_draft_tokens=self._mtp_first.draft_slots,
            cuda_graph_config=config,
        ):
            super().init_cuda_graphs()
        runner = self.draft_model_runner.decode_cuda_graph_runner
        if runner is not None:
            widths = (
                runner.captured_req_width,
                self._draft_block_spec_info.draft_token_num,
                self._draft_block_spec_info.num_tokens_per_req,
                self._draft_block_spec_info.num_tokens_for_logprob_per_req,
            )
            if any(width != self._mtp_first.draft_slots for width in widths):
                raise RuntimeError(f"DMTP DF capture/metadata width mismatch: {widths}")
        backend = self.draft_model_runner.attn_backend
        if (
            self.draft_model_runner.decode_cuda_graph_runner is not None
            and backend.mtp_anchor_fa3.capture_calls < 1
        ):
            raise RuntimeError("DMTP anchor-relative FA3 adapter was not captured")
        self._mtp_first.native.init_cuda_graphs()
        self._mtp_first.init_graphs()

    def _validate_phase1_sampling_support(self, batch):
        if not batch.forward_mode.is_idle():
            self._mtp_first.validate_sampling(batch)
        if batch.return_logprob or batch.has_grammar:
            raise ValueError("DMTP logprobs/grammar are not validated")
