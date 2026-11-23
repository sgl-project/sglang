import unittest
from importlib.metadata import EntryPoint
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.arg_groups.engram_config import EngramConfig
from sglang.srt.layers import engram, engram_host_table
from sglang.srt.model_executor.cuda_graph_config import Backend, PhaseConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeBackend(engram_host_table.EngramHostTableBackend):
    def __init__(self, spec, config):
        super().__init__(spec, config)
        self._rows = tuple(config["rows"])
        self.closed = False

    @property
    def row_range(self):
        return self._rows

    def lookup(self, indices, out):
        lo, hi = self.row_range
        owned = (indices >= lo) & (indices < hi)
        out.copy_(owned.unsqueeze(-1).expand_as(out))

    def close(self):
        self.closed = True


class TestEngramHostTableBackend(unittest.TestCase):
    def setUp(self):
        parallel = SimpleNamespace(
            tp_rank=0,
            tp_size=2,
            tp_group=None,
            pp_size=1,
            dp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            dcp_size=1,
        )
        self.graph = SimpleNamespace(
            prefill=PhaseConfig(),
            decode=PhaseConfig(backend=Backend.FULL, max_bs=32),
        )
        execution = SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=self.graph),
            overlap=SimpleNamespace(enable_two_batch_overlap=False),
        )
        self.schedule = SimpleNamespace(chunked_prefill_size=16, max_running_requests=4)
        self.model = SimpleNamespace(model_path="/model", load_format="auto")
        self.enterContext(patch.object(engram, "get_parallel", return_value=parallel))
        self.enterContext(patch.object(engram, "get_exec", return_value=execution))
        self.enterContext(
            patch.object(
                engram,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            )
        )
        self.enterContext(
            patch.object(engram, "get_schedule", return_value=self.schedule)
        )
        self.enterContext(patch.object(engram, "get_model", return_value=self.model))
        self.enterContext(
            patch.object(engram, "max_prefill_buffer_tokens", return_value=16)
        )
        self.graph_align = self.enterContext(
            patch.object(
                engram, "get_cuda_graph_max_batch_size", side_effect=lambda n: n
            )
        )
        self.enterContext(
            patch.object(engram, "get_eager_max_batch_size", side_effect=lambda n: n)
        )
        self.enterContext(patch.object(engram, "is_cuda", return_value=True))
        self.enterContext(patch.object(torch.cuda, "current_device", return_value=3))
        entry = EntryPoint(
            "fake", f"{__name__}:FakeBackend", "sglang.engram_host_table_backends"
        )
        self.entry_points = self.enterContext(
            patch.object(engram_host_table, "entry_points", return_value=[entry])
        )
        self.fake_config = EngramConfig.from_dict(
            {
                "placement": "cpu",
                "host_backend": "fake",
                "host_backend_config": {"rows": [0, 4]},
            }
        )

    def embedding(self, config):
        with patch.object(engram, "resolve_engram_config", return_value=config):
            return engram.EngramEmbedding(10, 32, 1, (0, 4, 10))

    def test_uses_registered_backend_with_native_shard(self):
        embedding = self.embedding(self.fake_config)
        self.entry_points.assert_called_once_with(
            group="sglang.engram_host_table_backends", name="fake"
        )
        self.assertEqual(embedding._host_backend.spec.device, torch.device("cuda:3"))
        self.assertEqual(embedding._host_backend.spec.max_num_tokens, 32)
        self.assertEqual(embedding.weight.untyped_storage().nbytes(), 1)
        self.assertEqual(embedding.scale.untyped_storage().nbytes(), 1)
        embedding.weight.weight_loader(embedding.weight, torch.empty(0))
        out = embedding._owned_rows(torch.tensor([[0, 4]]))
        torch.testing.assert_close(
            out, torch.tensor([[[1.0] * 32, [0.0] * 32]], dtype=torch.bfloat16)
        )

    def test_prefill_graph_buckets_can_exceed_chunked_prefill(self):
        self.model.load_format = "safetensors"
        self.graph.prefill = PhaseConfig(backend=Backend.BREAKABLE, bs=[32, 4095])
        self.graph_align.side_effect = lambda n: (n + 7) // 8 * 8
        embedding = self.embedding(self.fake_config)
        self.graph_align.assert_called_once_with(4095)
        self.assertEqual(embedding._host_backend.spec.max_num_tokens, 4096)

    def test_rejects_unchunked_prefill_before_loading_provider(self):
        self.schedule.chunked_prefill_size = -1
        with self.assertRaisesRegex(ValueError, "--chunked-prefill-size > 0"):
            self.embedding(self.fake_config)
        self.entry_points.assert_not_called()

    def test_rejects_direct_checkpoint_loading_before_loading_provider(self):
        self.model.load_format = "sharded_state"
        with self.assertRaisesRegex(ValueError, "--load-format auto or safetensors"):
            self.embedding(self.fake_config)
        self.entry_points.assert_not_called()

    def test_rejects_wrong_range_and_releases_backend(self):
        for bounds in ([0, 11], [0, 5]):
            with (
                self.subTest(bounds=bounds),
                patch.object(FakeBackend, "close") as close,
            ):
                config = EngramConfig.from_dict(
                    {
                        "placement": "cpu",
                        "host_backend": "fake",
                        "host_backend_config": {"rows": bounds},
                    }
                )
                with self.assertRaisesRegex(ValueError, "expected rank 0's rows"):
                    self.embedding(config)
                close.assert_called_once()

    def test_default_uses_native_weights(self):
        embedding = self.embedding(EngramConfig.from_dict({"lookup": "all_reduce"}))
        self.entry_points.assert_not_called()
        raw = torch.arange(10, dtype=torch.float32).view(10, 1).expand(10, 32)
        weight = raw.to(torch.float8_e4m3fn)
        embedding.weight.weight_loader(embedding.weight, weight)
        self.assertIsNone(embedding._host_backend)
        embedding.begin_lookup(torch.tensor([[0, 4]]))
        torch.testing.assert_close(embedding.weight.float(), weight[:4].float())

    def test_shared_table_and_legacy_backend_do_not_publish(self):
        shared = EngramConfig.from_dict({"placement": "cpu", "replication": "per_node"})
        host = SimpleNamespace(
            replication=shared.replication,
            device_ptr=0,
            bytes=torch.zeros(330, dtype=torch.uint8),
        )
        with (
            patch.object(engram, "_HostTable", return_value=host),
            patch.object(engram, "_sorted_host_reads", return_value=False),
        ):
            embedding = self.embedding(shared)
        self.assertTrue(embedding._shared)
        old = self.embedding(self.fake_config)
        for emb in (embedding, old):
            self.assertEqual(emb.host_issue, "inline")
            self.assertFalse(emb.split_host_lookup)
            emb.begin_lookup(torch.tensor([[0, 4]]))
        with patch.object(
            old._host_backend, "lookup", wraps=old._host_backend.lookup
        ) as lookup:
            old._owned_rows(torch.tensor([[0, 4]]))
            lookup.assert_called_once()

    def test_split_provider_requires_both_hooks_and_eager_prefill(self):
        with patch.object(FakeBackend, "begin_lookup"):
            with self.assertRaisesRegex(
                ValueError, "both begin_lookup and finish_lookup"
            ):
                self.embedding(self.fake_config)
            with patch.object(FakeBackend, "finish_lookup"):
                self.graph.prefill = PhaseConfig(backend=Backend.BREAKABLE, bs=[16])
                with self.assertRaisesRegex(
                    ValueError, "prefill CUDA graphs to be disabled"
                ):
                    self.embedding(self.fake_config)

    def test_schedule_and_empty_batch(self):
        hashes = torch.arange(12).reshape(3, 2, 2)
        for mode, starts in (
            ("inline", ((), ())),
            ("early", ((1, 14), ())),
            ("relay", ((1,), (14,))),
        ):
            layers = {}
            for position, layer in enumerate((1, 14)):
                emb = self.embedding(self.fake_config)
                emb.host_issue = mode
                emb._host_backend.begin_lookup = Mock()
                emb._host_backend.finish_lookup = Mock()
                layers[layer] = SimpleNamespace(embed=emb, layer_hash_index=position)
            schedule = engram.host_lookup_schedule(layers.items())
            for after, expected in zip((None, 1), starts):
                self.assertEqual(
                    tuple(target.layer_hash_index for target in schedule.get(after, ())),
                    tuple((1, 14).index(layer) for layer in expected),
                )
                engram.begin_host_lookups(schedule, after, hashes)
            for target in layers.values():
                emb = target.embed
                begin = emb._host_backend.begin_lookup
                if mode == "inline":
                    begin.assert_not_called()
                else:
                    begin.assert_called_once()
                    actual = begin.call_args.args[0]
                    expected_slice = hashes[:, target.layer_hash_index]
                    self.assertEqual(actual.data_ptr(), expected_slice.data_ptr())
                    self.assertEqual(actual.stride(), expected_slice.stride())
                begin.reset_mock()
                empty = hashes[:0, target.layer_hash_index]
                emb.begin_lookup(empty)
                self.assertEqual(emb._lookup(empty).shape, (0, 2, 32))
                begin.assert_not_called()
                emb._host_backend.finish_lookup.assert_not_called()

    def test_backend_config_requires_cpu_per_rank_layout(self):
        with self.assertRaisesRegex(ValueError, "needs placement 'cpu'"):
            EngramConfig.from_dict({"host_backend": "fake"})
        with self.assertRaisesRegex(ValueError, "needs 'host_backend'"):
            EngramConfig.from_dict({"host_backend_config": {}})
        with self.assertRaisesRegex(ValueError, "invalid host_backend_config"):
            EngramConfig.from_dict(
                {"placement": "cpu", "host_backend": "fake", "host_backend_config": 42}
            )


if __name__ == "__main__":
    unittest.main()
