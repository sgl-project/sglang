"""Indexer top-k capturer backend selection tests."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.state_capturer import indexer_topk as indexer_topk_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _model_config(architecture, **attrs):
    text_config = SimpleNamespace(
        architectures=[architecture],
        index_topk=8,
        num_hidden_layers=4,
        compress_ratios=[4, 0, 4, 128],
        **attrs,
    )
    return SimpleNamespace(hf_text_config=text_config)


class TestIndexerTopkNpuSupport(CustomTestCase):
    def _enabled_exec(self):
        return SimpleNamespace(
            features=SimpleNamespace(enable_return_indexer_topk=True)
        )

    def test_npu_dsa_uses_common_capturer(self):
        config = _model_config("GlmMoeDsaForCausalLM")
        sentinel = object()
        with (
            mock.patch.object(
                indexer_topk_mod, "get_exec", return_value=self._enabled_exec()
            ),
            mock.patch.object(
                indexer_topk_mod,
                "_create_indexer_capturer_raw",
                return_value=sentinel,
            ) as create,
        ):
            result = indexer_topk_mod.create_indexer_capturer(
                model_config=config,
                num_tokens=32,
                max_running_requests=4,
                device="npu",
            )

        self.assertIs(result, sentinel)
        self.assertEqual(create.call_args.kwargs["num_indexer_layers"], 4)
        self.assertEqual(create.call_args.kwargs["index_topk"], 8)

    def test_npu_v4_uses_compressed_layer_count(self):
        config = _model_config("DeepseekV4ForCausalLM")
        sentinel = object()
        with (
            mock.patch.object(
                indexer_topk_mod, "get_exec", return_value=self._enabled_exec()
            ),
            mock.patch.object(
                indexer_topk_mod,
                "_create_indexer_capturer_raw",
                return_value=sentinel,
            ) as create,
        ):
            result = indexer_topk_mod.create_indexer_capturer(
                model_config=config,
                num_tokens=32,
                max_running_requests=4,
                device="npu",
            )

        self.assertIs(result, sentinel)
        self.assertEqual(create.call_args.kwargs["num_indexer_layers"], 2)

    def test_non_indexer_npu_does_not_create_capturer(self):
        config = _model_config("LlamaForCausalLM")
        with (
            mock.patch.object(
                indexer_topk_mod, "get_exec", return_value=self._enabled_exec()
            ),
            mock.patch.object(
                indexer_topk_mod, "_create_indexer_capturer_raw"
            ) as create,
        ):
            result = indexer_topk_mod.create_indexer_capturer(
                model_config=config,
                num_tokens=32,
                max_running_requests=4,
                device="npu",
            )

        self.assertIsNone(result)
        create.assert_not_called()

    def test_invalid_topk_disables_raw_capturer(self):
        self.assertIsNone(
            indexer_topk_mod._create_indexer_capturer_raw(
                enable=True,
                num_indexer_layers=2,
                index_topk=0,
                num_tokens=32,
                max_running_requests=4,
                device="npu",
            )
        )


if __name__ == "__main__":
    unittest.main()
