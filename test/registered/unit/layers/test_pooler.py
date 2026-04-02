import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import torch


def _load_pooler_module():
    activation_mod = types.ModuleType("sglang.srt.layers.activation")
    activation_mod.get_cross_encoder_activation_function = lambda config: (lambda x: x)

    forward_batch_mod = types.ModuleType("sglang.srt.model_executor.forward_batch_info")

    class ForwardBatch:
        pass

    forward_batch_mod.ForwardBatch = ForwardBatch

    transformers_mod = types.ModuleType("transformers")

    class PretrainedConfig:
        pass

    transformers_mod.PretrainedConfig = PretrainedConfig

    repo_root = Path(__file__).resolve().parents[4]
    pooler_path = repo_root / "python" / "sglang" / "srt" / "layers" / "pooler.py"
    spec = importlib.util.spec_from_file_location("test_pooler_isolated", pooler_path)
    module = importlib.util.module_from_spec(spec)
    patched_modules = {
        "sglang.srt.layers.activation": activation_mod,
        "sglang.srt.model_executor.forward_batch_info": forward_batch_mod,
        "transformers": transformers_mod,
        spec.name: module,
    }
    previous_modules = {
        name: sys.modules.get(name) for name in patched_modules if name in sys.modules
    }

    try:
        sys.modules.update(patched_modules)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in patched_modules:
            if name in previous_modules:
                sys.modules[name] = previous_modules[name]
            else:
                sys.modules.pop(name, None)


_POOLER_MODULE = _load_pooler_module()
Pooler = _POOLER_MODULE.Pooler
PoolingType = _POOLER_MODULE.PoolingType


def _make_forward_batch(extend_seq_lens, dimensions=None, extend_seq_lens_cpu=None):
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.int32),
        extend_seq_lens_cpu=(
            list(extend_seq_lens)
            if extend_seq_lens_cpu is None
            else extend_seq_lens_cpu
        ),
        dimensions=dimensions,
    )


class TestPooler(unittest.TestCase):
    def test_last_pooling_returns_last_token(self):
        hidden_states = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 3])

        output = Pooler(PoolingType.LAST, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0, 20.0],
                    [5.0, 50.0],
                ]
            ),
        )

    def test_cls_pooling_returns_first_token(self):
        hidden_states = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 3])

        output = Pooler(PoolingType.CLS, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [1.0, 10.0],
                    [3.0, 30.0],
                ]
            ),
        )

    def test_mean_pooling_returns_segment_means(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 3])

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0, 3.0],
                    [7.0, 8.0],
                ]
            ),
        )

    def test_mean_pooling_normalizes_output(self):
        hidden_states = torch.tensor(
            [
                [2.0, 2.0],
                [4.0, 6.0],
                [0.0, 3.0],
                [0.0, 7.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 2])

        output = Pooler(PoolingType.MEAN, normalize=True)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [0.6, 0.8],
                    [0.0, 1.0],
                ]
            ),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_mean_pooling_supports_tensor_extend_seq_lens_cpu(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ]
        )
        forward_batch = _make_forward_batch(
            [2, 3], extend_seq_lens_cpu=torch.tensor([2, 3], dtype=torch.int32)
        )

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0, 3.0],
                    [7.0, 8.0],
                ]
            ),
        )

    def test_mean_pooling_accumulates_in_float32(self):
        torch.manual_seed(2)
        hidden_states = (torch.randn(4096, 2, dtype=torch.float32) * 100).to(
            torch.bfloat16
        )
        forward_batch = _make_forward_batch([2048, 2048])

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        expected = torch.stack(
            [
                hidden_states[:2048].float().mean(dim=0),
                hidden_states[2048:].float().mean(dim=0),
            ]
        )
        self.assertEqual(output.embeddings.dtype, torch.float32)
        torch.testing.assert_close(output.embeddings, expected, atol=1e-6, rtol=0.0)

    def test_dimensions_same_truncates_tensor_output(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [6.0, 8.0, 10.0],
                [8.0, 10.0, 12.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 2], dimensions=[1, 1])

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0],
                    [7.0],
                ]
            ),
        )

    def test_dimensions_mixed_returns_list_and_normalizes_each(self):
        hidden_states = torch.tensor(
            [
                [2.0, 2.0, 0.0],
                [4.0, 6.0, 0.0],
                [0.0, 2.0, 2.0],
                [0.0, 4.0, 6.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 2], dimensions=[1, 2])

        output = Pooler(PoolingType.MEAN, normalize=True)(hidden_states, forward_batch)

        self.assertIsInstance(output.embeddings, list)
        self.assertEqual(len(output.embeddings), 2)
        torch.testing.assert_close(output.embeddings[0], torch.tensor([1.0]))
        torch.testing.assert_close(
            output.embeddings[1],
            torch.tensor([0.0, 1.0]),
            atol=1e-6,
            rtol=1e-6,
        )
