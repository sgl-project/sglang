from sglang.srt.layers.attention.mha_chunk_prefix.tuned import tuned_dispatch_mha_chunk
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_bool_env_var, get_int_env_var


class MhaChunkHelper:
    def __init__(
        self,
        num_local_heads: int,
    ):
        self.num_local_heads = num_local_heads
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

        self.attention_backend = global_server_args_dict["prefill_attention_backend"]

    def dispatch_mha_chunk(self, forward_batch: ForwardBatch):
        if (
            not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            return False
        if (
            not hasattr(forward_batch.attn_backend, "is_mha_chunk_supported")
            or not forward_batch.attn_backend.is_mha_chunk_supported()
        ):
            return False

        if forward_batch.extend_prefix_lens_cpu is None:
            return False
        sum_extend_prefix_lens = sum(forward_batch.extend_prefix_lens_cpu)
        if sum_extend_prefix_lens != 0 and self.disable_chunked_prefix_cache:
            return False
        if self.chunked_prefix_cache_use_tuned:
            rst = tuned_dispatch_mha_chunk(
                self.attention_backend,
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
