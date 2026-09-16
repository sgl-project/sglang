"""Small CPU tensors; production CP slicing/gather, mocked collective transport."""

from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.model_executor.forward_batch_info import ForwardMode

CP = "sglang.srt.layers.cp"


@contextmanager
def cp_context(size, rank, lengths=(3, 6), prefix_lengths=(7, 13)):
    """Keep real interleave indexing/padding; replace only runtime context."""
    strategy = InterleaveCPStrategy(size)
    parallel = NS(attn_cp_size=size, attn_cp_rank=rank, attn_cp_group=None)
    batch = NS(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.arange(1, sum(lengths) + 1),
        positions=torch.cat(
            [
                torch.arange(prefix, prefix + length)
                for prefix, length in zip(prefix_lengths, lengths)
            ]
        ),
        extend_seq_lens_cpu=list(lengths),
        extend_prefix_lens_cpu=list(prefix_lengths),
        mm_inputs=None,
        spec_info=None,
    )
    batch.attn_cp_metadata = strategy.build_metadata(
        sum(lengths), [p + n for p, n in zip(prefix_lengths, lengths)], list(lengths)
    )
    with ExitStack() as stack:
        for module in ("base", "utils", "padding", "interleave"):
            stack.enter_context(
                patch(CP + "." + module + ".get_parallel", return_value=parallel)
            )
        stack.enter_context(patch(CP + ".utils.get_cp_strategy", return_value=strategy))
        stack.enter_context(
            patch(CP + ".padding.get_cp_padding_align_size", return_value=size)
        )
        stack.enter_context(
            patch(
                CP + ".utils.get_moe_a2a_backend", return_value=NS(is_none=lambda: True)
            )
        )
        pad_logical_token_to_physical(batch.attn_cp_metadata)
        yield strategy, batch


@contextmanager
def simulated_collective(strategy, batch, global_tensor):
    """Inject peer buffers into all-gather; retain production unpadding/reordering."""
    physical = max(batch.attn_cp_metadata.per_rank_actual_token)
    buffers = []
    for rank in range(strategy.cp_size):
        buf = global_tensor.new_zeros((physical, *global_tensor.shape[1:]))
        local = global_tensor[rank :: strategy.cp_size]
        buf[: len(local)] = local
        buffers.append(buf)

    def gather(output, local):
        torch.testing.assert_close(local, buffers[strategy.cp_rank], rtol=0, atol=0)
        output.copy_(torch.cat(buffers))

    with (
        patch(
            CP + ".interleave.use_symmetric_memory",
            side_effect=lambda *a, **k: nullcontext(),
        ),
        patch(CP + ".interleave.is_allocation_symmetric", return_value=False),
        patch(CP + ".interleave.attn_cp_all_gather_into_tensor", side_effect=gather),
    ):
        yield
