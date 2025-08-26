from typing import TYPE_CHECKING

from sglang.srt.layers.attention.mha_chunk_prefix.tuned import tuned_dispatch_mha_chunk
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_bool_env_var, get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class MhaChunkHelper:
    def __init__(
        self,
        model_runner: "ModelRunner",
    ):
        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = get_int_env_var(
            "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD", 8192
        )

        self.chunked_prefix_cache_use_tuned = get_bool_env_var(
            "SGL_CHUNKED_PREFIX_CACHE_USE_TUNED", "false"
        )

        self.disable_chunked_prefix_cache = global_server_args_dict[
            "disable_chunked_prefix_cache"
        ]
        self.attention_backend_str = global_server_args_dict[
            "prefill_attention_backend"
        ]

        num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_local_heads = num_local_heads

        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        prefill_attn_backend = (
            model_runner.attn_backend.prefill_backend
            if isinstance(model_runner.attn_backend, HybridAttnBackend)
            else model_runner.attn_backend
        )
        self.is_mha_chunk_supported = (
            hasattr(prefill_attn_backend, "is_mha_chunk_supported")
            and prefill_attn_backend.is_mha_chunk_supported()
        )

    def dispatch_mha_chunk(self, forward_batch: ForwardBatch):
        if (
            not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            return False
        if not self.is_mha_chunk_supported:
            return False

        if forward_batch.extend_prefix_lens_cpu is None:
            return False
        sum_extend_prefix_lens = sum(forward_batch.extend_prefix_lens_cpu)
        if sum_extend_prefix_lens != 0 and self.disable_chunked_prefix_cache:
            return False
        if self.chunked_prefix_cache_use_tuned:
            rst = tuned_dispatch_mha_chunk(
                self.attention_backend_str,
                self.num_local_heads,
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.seq_lens_cpu,
            )
            if rst is not None:
                return rst
        return (
            sum_extend_prefix_lens == 0
            or sum_extend_prefix_lens >= self.chunked_prefix_cache_threshold
        )
