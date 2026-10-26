"""Synthetic chunk profiling must stay within request bounds and release on failure."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_components import dynamic_chunk_sizer as module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDynamicChunkSizerProfiling(CustomTestCase):
    def _profile(self, *, context, mapping, prefill, base, fail_forward=False):
        prepared = []
        live = set()
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                context_len=context, hidden_size=2, dtype=torch.float32
            ),
            get_pp_proxy_topk_size=lambda: None,
            get_pp_proxy_residual_num_blocks=lambda: None,
        )

        def make_req(**kwargs):
            return SimpleNamespace(
                rid=kwargs["rid"],
                full_untruncated_fill_ids=kwargs["origin_input_ids"],
                init_next_round_input=Mock(),
                last_node=None,
                kv=SimpleNamespace(holds_kv=False),
            )

        def make_batch(reqs, *_args):
            req = reqs[0]

            def prepare():
                prepared.append(req.extend_end)
                live.add(req.rid)
                req.kv.holds_kv = True

            return SimpleNamespace(
                prepare_for_extend=prepare,
                input_ids=torch.zeros(1, dtype=torch.int64),
                forward_mode=SimpleNamespace(is_extend=lambda: True),
            )

        def release(req, _cache, *, checkpoint):
            self.assertFalse(checkpoint)
            live.remove(req.rid)
            req.kv.holds_kv = False

        runner.forward = Mock(
            side_effect=RuntimeError("synthetic forward failed")
            if fail_forward
            else None
        )
        with (
            patch.object(
                module, "get_parallel", return_value=SimpleNamespace(pp_rank=0)
            ),
            patch.object(module, "Req", side_effect=make_req),
            patch.object(module.ScheduleBatch, "init_new", side_effect=make_batch),
            patch.object(
                module.ForwardBatch, "init_new", return_value=SimpleNamespace()
            ),
            patch.object(module, "release_kv_cache", side_effect=release),
            patch.object(module, "is_dp_attention_enabled", return_value=False),
            patch.object(module, "set_is_extend_in_batch"),
            patch.object(module, "get_device_module", return_value=Mock()),
            patch.object(module, "tqdm", side_effect=lambda items, **_kwargs: items),
        ):
            sizer = module.DynamicChunkSizer(
                model_runner=runner,
                model_config=runner.model_config,
                tree_cache=Mock(),
                req_to_token_pool=SimpleNamespace(max_context_len=mapping),
                token_to_kv_pool_allocator=Mock(),
                spec_algorithm=Mock(),
                chunked_prefill_size=base,
                max_prefill_tokens=prefill,
                page_size=1,
                device="cpu",
            )
            if fail_forward:
                with self.assertRaisesRegex(RuntimeError, "synthetic forward failed"):
                    sizer._profile_prefill_latency()
                self.assertEqual(len(prepared), 1)
            else:
                lengths, latencies = sizer._profile_prefill_latency()
                self.assertEqual(lengths, prepared)
                self.assertEqual(len(latencies), len(lengths))
                self.assertGreater(len(set(lengths)), 8)
                self.assertGreater(min(lengths), 0)
                self.assertLessEqual(max(lengths), min(context, mapping))
            self.assertEqual(
                live, set(), "profiling left a prepared request holding KV"
            )
            return prepared

    def test_samples_fit_request_capacity_and_preserve_probe_range(self):
        for context, mapping, prefill, base, expected_max in (
            (8192, 8196, 16384, 16384, 8192),
            (8192, 4096, 16384, 16384, 4096),
            (8192, 8196, 2048, 16384, 8192),
            (8192, 8196, 16384, 512, 640),
            (32768, 32772, 16384, 16384, 20480),
        ):
            with self.subTest(
                context=context, mapping=mapping, prefill=prefill, base=base
            ):
                lengths = self._profile(
                    context=context, mapping=mapping, prefill=prefill, base=base
                )
                self.assertEqual(max(lengths), expected_max)

    def test_forward_failure_releases_prepared_request(self):
        self._profile(
            context=8192, mapping=8196, prefill=16384, base=512, fail_forward=True
        )


if __name__ == "__main__":
    unittest.main()
