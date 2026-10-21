"""Exercise compact bounded-replay output gathering using simulated CP ranks."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    LateLayerTail,
)
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
BACKEND = "sglang.srt.layers.attention.deepseek_v4_backend"
MODEL = "sglang.srt.models.deepseek_v4"
INTERLEAVE = "sglang.srt.layers.cp.interleave"
CP = "sglang.srt.layers.cp.utils"


class TestDSV41CPTail(CustomTestCase):
    def test_tail_gather_preserves_dspark_global_indices_and_empty_ranks(self):
        cp_size = 8
        token_indices = torch.tensor([0, 1, 16, 17, 18, 19])
        lengths = torch.tensor([2, 3, 1])
        # These are global logical tail rows. The collective below packs them
        # rank-major; the real gather_index must restore request order.
        global_outputs = [
            torch.stack(
                [token_indices.float() + k, token_indices.float() + k + 0.5], -1
            )
            for k in range(3)
        ]
        global_outputs.append(token_indices)
        for rank in range(cp_size):
            with self.subTest(rank=rank):
                full_cp = object()
                batch = SimpleNamespace(
                    batch_size=3,
                    forward_mode=ForwardMode.EXTEND,
                    positions=torch.arange(20),
                    input_ids=torch.arange(20) + 100,
                    attn_cp_metadata=full_cp,
                    return_logprob=False,
                )
                parallel = SimpleNamespace(
                    attn_cp_rank=rank, attn_cp_size=cp_size, attn_cp_group=None
                )
                backend = object.__new__(DeepseekV4AttnBackend)
                with patch(f"{BACKEND}.get_parallel", return_value=parallel):
                    layout = backend._late_layer_tail_cp_layout(
                        batch, token_indices, lengths
                    )
                meta = layout["cp_metadata"]
                tail = LateLayerTail(
                    token_indices=layout["local_token_indices"],
                    positions=layout["local_positions"],
                    extend_seq_lens=lengths,
                    extend_seq_lens_cpu=lengths.tolist(),
                    swa_out_cache_loc=torch.empty(0),
                    pad_rows=layout["pad_rows"],
                    cp_metadata=meta,
                    local_lens_cpu=layout["local_lens_cpu"],
                )
                strategy = InterleaveCPStrategy(cp_size=cp_size)
                owned = token_indices % cp_size == rank
                physical = meta.per_rank_actual_token[rank]

                def local_tensor(value, owned=owned, physical=physical):
                    local = value[owned]
                    pad = local.new_zeros((physical - local.shape[0], *local.shape[1:]))
                    return torch.cat([local, pad])

                peer_outputs = iter(global_outputs)

                def gather(out, local):
                    value = next(peer_outputs)
                    chunks = []
                    for r in range(cp_size):
                        chunk = value[token_indices % cp_size == r]
                        padding = chunk.new_zeros(
                            (physical - len(chunk), *chunk.shape[1:])
                        )
                        chunks.append(torch.cat([chunk, padding]))
                    chunks[rank] = local
                    out.copy_(torch.cat(chunks))

                def logits(input_ids, hidden, lm_head, metadata, aux, **kwargs):
                    return SimpleNamespace(
                        next_token_logits=hidden @ lm_head, hidden_states=aux
                    )

                head = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                model = SimpleNamespace(
                    capture_aux_hidden_states=True,
                    model=SimpleNamespace(late_layer_start=21),
                    logits_processor=logits,
                    lm_head=head,
                )
                with (
                    patch(
                        f"{MODEL}.get_attn_backend",
                        return_value=SimpleNamespace(
                            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
                        ),
                    ),
                    patch(f"{MODEL}.get_parallel", return_value=parallel),
                    patch(
                        f"{MODEL}.LogitsMetadata",
                        SimpleNamespace(from_forward_batch=lambda _: SimpleNamespace()),
                    ),
                    patch("torch.cuda.current_stream", return_value=None),
                    patch(f"{CP}.is_cp_active", return_value=True),
                    patch(f"{CP}.get_cp_strategy", return_value=strategy),
                    # cp_rank is inherited from the base strategy and reads
                    # its module's runtime context, not an instance attribute.
                    patch(
                        "sglang.srt.layers.cp.base.get_parallel", return_value=parallel
                    ),
                    patch(f"{INTERLEAVE}.get_parallel", return_value=parallel),
                    patch(
                        f"{INTERLEAVE}.use_symmetric_memory", return_value=nullcontext()
                    ),
                    patch(f"{INTERLEAVE}.is_allocation_symmetric", return_value=False),
                    patch(
                        f"{INTERLEAVE}.attn_cp_all_gather_into_tensor",
                        side_effect=gather,
                    ),
                ):
                    out = DeepseekV4ForCausalLM.logits_from_cp_hidden_states(
                        model,
                        (
                            local_tensor(global_outputs[0]),
                            local_tensor(global_outputs[1]),
                        ),
                        local_tensor(global_outputs[2]),
                        batch,
                    )
                torch.testing.assert_close(
                    out.hidden_states_token_indices, token_indices
                )
                torch.testing.assert_close(
                    out.next_token_logits, global_outputs[0] @ head
                )
                torch.testing.assert_close(out.hidden_states, global_outputs[2])


if __name__ == "__main__":
    unittest.main()
