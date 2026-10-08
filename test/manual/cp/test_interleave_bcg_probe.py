"""Guard the deliberately narrow interleave BCG performance experiment."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.bcg import PrefillCPBCGInput
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.layers.cp.interleave_bcg import (
    InterleaveCPBCGInput,
    cp_interleave_indexer_rows,
    supports_interleave_bcg,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase


class TestInterleaveBCGProbe(CustomTestCase):
    def test_single_rank_keeps_eager_fallback(self):
        """CP=1 has no strategy, so enabling its graph adapter crashes startup."""
        config = SimpleNamespace(
            enable_prefill_cp=True,
            pp_size=1,
            dp_size=1,
            ep_size=1,
            attn_cp_size=4,
            tp_size=4,
            _model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Glm5NextForConditionalGeneration"]
                )
            ),
        )
        self.assertTrue(supports_interleave_bcg(config))
        config.attn_cp_size = config.tp_size = 1
        self.assertFalse(supports_interleave_bcg(config))

    def test_shared_cp_adapter_preserves_mrope_selection(self):
        batch = SimpleNamespace(
            positions=torch.arange(16), mrope_positions=torch.zeros(3, 16)
        )
        for model in (
            SimpleNamespace(is_mrope_enabled=True),
            SimpleNamespace(language_model=SimpleNamespace(is_mrope_enabled=True)),
        ):
            runner = SimpleNamespace(
                enable_cp_bcg_capture=True,
                prefill_cp_bcg_input=PrefillCPBCGInput(torch.empty(0), torch.empty(0)),
                model_runner=SimpleNamespace(model=model),
            )
            self.assertIs(
                PrefillCudaGraphRunner._get_layer_model_positions(runner, batch),
                batch.mrope_positions,
            )

    def test_global_ids_capacity_includes_cp_padding(self):
        runner = SimpleNamespace(
            device="cpu",
            max_num_tokens=17,
            model_runner=SimpleNamespace(
                dtype=torch.float32,
                model_config=SimpleNamespace(hidden_size=2),
            ),
        )
        with get_parallel().override(attn_cp_size=4):
            adapter = InterleaveCPBCGInput.create(runner)
        self.assertGreaterEqual(adapter.input_ids_global.numel(), 32)

    def test_prepare_preserves_global_positions_and_refreshes_static_shards(self):
        runner = SimpleNamespace(
            device="cpu",
            max_num_tokens=16,
            model_runner=SimpleNamespace(
                dtype=torch.float32,
                model_config=SimpleNamespace(hidden_size=2),
                model=SimpleNamespace(
                    get_input_embeddings=lambda: (
                        lambda ids: ids[:, None].expand(-1, 2).float()
                    )
                ),
            ),
        )
        init_cp_strategy(enable_prefill_cp=True, cp_size=4, cp_strategy="interleave")
        try:
            with (
                get_parallel().override(attn_cp_size=4, attn_cp_rank=2),
                patch(
                    "sglang.srt.layers.cp.utils.get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_none=lambda: True),
                ),
            ):
                adapter = InterleaveCPBCGInput.create(runner)
                input_ptr = adapter.input_embeds.data_ptr()
                position_ptr = adapter.positions.data_ptr()
                for capture, offset in ((True, 0), (False, 100)):
                    ids = torch.arange(16) + offset
                    positions = torch.arange(16)
                    batch = SimpleNamespace(
                        input_ids=ids,
                        positions=positions,
                        forward_mode=ForwardMode.EXTEND,
                        extend_num_tokens=16,
                        extend_seq_lens_cpu=[16],
                        seq_lens_cpu=[16],
                        attn_cp_metadata=None,
                    )
                    adapter.prepare(
                        runner, batch, static_num_tokens=16, capture=capture
                    )
                    self.assertIs(batch.positions, positions)
                    torch.testing.assert_close(
                        adapter.model_positions(batch), positions[2::4]
                    )
                    torch.testing.assert_close(
                        batch.input_embeds[:, 0], ids[2::4].float()
                    )
                    torch.testing.assert_close(
                        batch.input_ids_global, ids.reshape(4, 4).T.flatten()
                    )
                    self.assertEqual(batch.input_embeds.data_ptr(), input_ptr)
                    self.assertEqual(
                        adapter.model_positions(batch).data_ptr(), position_ptr
                    )
        finally:
            init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    def test_only_exact_single_sequence_bucket_replays(self):
        adapter = InterleaveCPBCGInput(torch.empty(0), torch.empty(0))
        adapter.bucket_local_tokens = {1024: 256}
        self.assertTrue(adapter.allows_replay(1, 1024, [0], False))
        self.assertFalse(adapter.allows_replay(1, 1000, [0], False))
        self.assertFalse(adapter.allows_replay(2, 1024, [0, 0], False))
        self.assertFalse(adapter.allows_replay(1, 1024, [16], False))
        self.assertFalse(adapter.allows_replay(1, 1024, [0], True))

    def test_indexer_uses_local_physical_and_logical_rows(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_num_tokens=15,
            extend_seq_lens_cpu=[15],
            attn_cp_metadata=InterleaveContextParallelMetadata(
                total_seq_lens=15,
                per_rank_actual_token=[4] * 4,
                per_rank_logical_token=[4, 4, 4, 3],
            ),
        )
        with get_parallel().override(attn_cp_rank=3, attn_cp_size=4):
            self.assertEqual(cp_interleave_indexer_rows(batch), (4, 3))


if __name__ == "__main__":
    unittest.main()
