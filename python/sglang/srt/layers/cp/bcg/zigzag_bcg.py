"""Zigzag-specific capture geometry for prefill context parallel CUDA graphs."""

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.bcg.bcg import PrefillCPBCGInput


class ZigzagCPBCGInput(PrefillCPBCGInput):
    @staticmethod
    def supports(server_args) -> bool:
        # Zigzag replay has not been adapted to pipeline stages.
        return resolving_view(server_args).pp_size == 1

    @staticmethod
    def min_capture_tokens(cp_size: int) -> int:
        return cp_size * 2

    def _max_local_tokens(self, extend_seq_lens) -> int:
        cp_size = get_cp_strategy().cp_size
        cp_segment_num = cp_size * 2
        per_rank_logical_tokens = [0] * cp_size
        for raw_length in extend_seq_lens:
            base, remainder = divmod(int(raw_length), cp_segment_num)
            for rank in range(cp_size):
                opposite_rank = cp_segment_num - 1 - rank
                per_rank_logical_tokens[rank] += (
                    base * 2 + int(rank < remainder) + int(opposite_rank < remainder)
                )
        return max(per_rank_logical_tokens)
