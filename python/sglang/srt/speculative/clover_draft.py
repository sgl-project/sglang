# SPDX-License-Identifier: Apache-2.0
"""Clover-only preview attention and CUDA graphs over shared target weights."""

from functools import partial

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
    fast_prefill_plan,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.clover_backend import CloverGDNAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.speculative.clover_info import CloverPreviewInput


class CloverPreviewFlashInferBackend(FlashInferAttnBackend):
    def __init__(self, runner, block_size):
        super().__init__(runner)
        # A private preview backend config, never published to ServerArgs.
        self.dllm_config = DllmConfig(
            "Clover",
            {},
            block_size,
            runner.model_config.hf_config.mask_token_id,
            runner.max_running_requests,
            requires_separate_context_encoding=True,
        )
        self.is_dllm_model = True
        # Paged mode lets one plan cover prefix and block with host metadata.
        self.use_paged = True

    def init_forward_metadata(self, forward_batch):
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata(forward_batch)

        if not isinstance(forward_batch.spec_info, CloverPreviewInput):
            raise TypeError("Clover preview requires CloverPreviewInput metadata")
        self._prepare_dequant_workspace_metadata_for_extend(forward_batch, False)
        self.indices_updater_prefill.update(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_cpu,
            forward_batch.seq_lens_sum,
            prefix_lens=forward_batch.extend_prefix_lens,
            prefill_wrappers=self.prefill_wrappers_paged,
            use_ragged=False,
            encoder_lens=forward_batch.encoder_lens,
            spec_info=forward_batch.spec_info,
            fixed_split_size=self.prefill_split_tile_size,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            custom_kv_indices=self.dq_page_table,
            plan=forward_batch.kv_loc_plan,
        )
        self.forward_metadata = PrefillMetadata(
            self.prefill_wrappers_paged,
            False,
            not any(forward_batch.extend_prefix_lens_cpu),
        )

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata_out_graph(forward_batch, in_capture)

        spec_info = forward_batch.spec_info
        if not isinstance(spec_info, CloverPreviewInput):
            raise TypeError("Clover preview requires CloverPreviewInput metadata")
        bs = forward_batch.batch_size
        if in_capture:
            self._prepare_cuda_graph_metadata(
                bs,
                forward_batch.positions.numel(),
                forward_batch.forward_mode,
                spec_info,
            )
        self.indices_updater_prefill.update(
            forward_batch.req_pool_indices[:bs],
            forward_batch.seq_lens[:bs],
            (
                forward_batch.seq_lens_cpu[:bs]
                if forward_batch.seq_lens_cpu is not None
                else None
            ),
            forward_batch.seq_lens_sum,
            prefix_lens=forward_batch.seq_lens - self.dllm_config.block_size,
            prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
            use_ragged=False,
            encoder_lens=(
                forward_batch.encoder_lens[:bs]
                if forward_batch.encoder_lens is not None
                else None
            ),
            spec_info=spec_info,
            plan=forward_batch.kv_loc_plan,
        )
        if (
            in_capture
            and forward_batch.spec_info.custom_mask is None
            and self.use_paged
            and self.prefill_backend == "fa2"
            and self.dispatch_reason is None
        ):
            # The first plan populates _cached_module, which the replay planner reads.
            for wrapper in self.prefill_cuda_graph_metadata[forward_batch.batch_size]:
                wrapper.begin_forward = partial(fast_prefill_plan, wrapper)

    def _prepare_cuda_graph_metadata(self, bs, num_tokens, forward_mode, spec_info):
        super()._prepare_cuda_graph_metadata(bs, num_tokens, forward_mode, spec_info)
        if forward_mode.is_dllm_extend():
            self.forward_metadata.use_ragged = False

    def get_cuda_graph_seq_len_fill_value(self):
        return self.dllm_config.block_size


class CloverPreviewBackend(HybridLinearAttnBackend):
    def __init__(self, runner, block_size):
        self.block_size = block_size
        super().__init__(
            CloverPreviewFlashInferBackend(runner, block_size),
            CloverGDNAttnBackend(runner),
            runner.model_config.hf_config.full_attention_layer_ids,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.block_size


class CloverDraftCudaGraphRunner(DecodeCudaGraphRunner):
    """Use native graph buffers/buckets, with Clover preview capture dispatch."""

    def capture(self):
        # Set before the base constructor sizes buffers and dummy batches.
        self.capture_forward_mode = ForwardMode.DLLM_EXTEND
        self.is_dllm = True
        return super().capture()

    def get_spec_info(self, num_tokens):
        return CloverPreviewInput(self.captured_req_width)
