"""Physical storage accounting for the DSV4 PA_ND pools."""

from sglang.srt.hardware_backend.npu.dsv4.dsv4_kv_layout import (
    NPUDSV4KVLayout,
    get_a3_c2_state_bank_count,
    pa_nd_num_pages,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DSV41_INDEX_PAGE_SIZE
from sglang.srt.model_executor.pool_configurator import DSV4PagedPoolLayout
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils.common import ceil_align


class NPUDSV4PagedPoolLayout(DSV4PagedPoolLayout):
    def __init__(self, kvc):
        # A3 SMLA consumes native BF16 KV/index rows; A5 keeps its existing
        # mixed-quant pricing. V4 C4/C128 keep the existing budget on both.
        self.index_layout = None
        if any(
            r in NPUDSV4KVLayout.COMPRESSED_RATIOS
            for r in kvc.model_config.compress_ratios
        ):
            self.index_layout = (
                NPUDSV4KVLayout.V41_CMP_FP4
                if is_npu_arch35()
                else NPUDSV4KVLayout.V41_BF16
            )
        if (
            self.index_layout is NPUDSV4KVLayout.V41_BF16
            and kvc.spec_algorithm.is_dspark()
            and not get_exec().features.enable_encoder_swa_bounded_replay
        ):
            # Every draft stage adds only a paged SWA, not another compressed pool.
            aux = kvc.spec_aux_config
            self.draft_swa_layers = int(
                aux.dspark_num_stages or aux.dflash_draft_num_layers or 0
            )
            assert self.draft_swa_layers > 0, "DSpark draft layer count is required"

    def low_ratio_index_row_bytes(self, head_dim):
        if self.index_layout is None:
            return super().low_ratio_index_row_bytes(head_dim)
        k, s, kd, sd = self.index_layout.indexer_kv_info(head_dim)
        return k * kd.itemsize + s * sd.itemsize

    def kv_row_bytes(self, pool, compress_ratio):
        if self.index_layout is NPUDSV4KVLayout.V41_BF16 and compress_ratio in (
            0,
            1,
            2,
        ):
            width, dtype = NPUDSV4KVLayout.V41_BF16.kv_info(
                pool.qk_nope_head_dim, pool.qk_rope_head_dim
            )
            return width * dtype.itemsize
        return pool.kv_bytes

    def low_ratio_padding_bytes(self, pool, full_tokens):
        if self.index_layout is not NPUDSV4KVLayout.V41_BF16:
            return 0
        row_bytes = self.kv_row_bytes(pool, 0)
        # SWA and compressed main KV each reserve one physical page.
        padding = (
            pool.page_size * row_bytes * (pool.num_layers_total + self.draft_swa_layers)
        )
        for ratio in (1, 2):
            size = (full_tokens + pool.page_size) // ratio
            index_slots = (
                pa_nd_num_pages(size, DSV41_INDEX_PAGE_SIZE) * DSV41_INDEX_PAGE_SIZE
            )
            padding += pool.num_layers(ratio) * (
                (pool.page_size // ratio) * row_bytes
                + (index_slots - full_tokens // ratio) * pool.low_ratio_index_bytes
            )
        return padding

    def pair_state_rows(self, num_slots, ring_size):
        if self.index_layout is not NPUDSV4KVLayout.V41_BF16:
            return super().pair_state_rows(num_slots, ring_size)
        # CompressorV2's 3-D state cache requires a full ring per padded TP bank.
        banks = get_a3_c2_state_bank_count(num_slots)
        return ceil_align(banks * ring_size + ring_size + 1, ring_size)
