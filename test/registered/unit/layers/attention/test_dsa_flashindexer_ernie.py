"""ERNIE 5.5 FlashIndexer glue: gate attach, head selection hook, FlashIndexer tensors and padding."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer, flashindexer
from sglang.srt.layers.attention.dsa.dsa_fp4_indexer import Fp4Indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.flashindexer import (
    FLASHINDEXER_BIAS_PARAM_SUFFIX,
    FLASHINDEXER_GATE_PARAM_SUFFIX,
    attach_flashindexer_gate,
    flashindexer_select_heads,
)
from sglang.srt.models import ernie55
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _indexer(n_heads=64, hidden_size=128):
    indexer = torch.nn.Module()
    indexer.n_heads, indexer.hidden_size = n_heads, hidden_size
    return indexer


class TestAttachFlashIndexerGate(CustomTestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(flashindexer, "is_cuda", lambda: True))

    def test_builds_the_checkpoint_params(self):
        indexer = _indexer()
        attach_flashindexer_gate(indexer)
        params = dict(indexer.named_parameters())
        gate = params[FLASHINDEXER_GATE_PARAM_SUFFIX]
        bias = params[FLASHINDEXER_BIAS_PARAM_SUFFIX]
        self.assertEqual((gate.shape, gate.dtype), ((64, 128), torch.bfloat16))
        self.assertEqual((bias.shape, bias.dtype), ((64,), torch.float32))
        self.assertFalse(bias.requires_grad)

    def test_rejects_shapes_the_kernel_cannot_tile(self):
        for n_heads, hidden_size in ((4, 128), (48, 128), (64, 100)):
            with self.subTest(n_heads=n_heads, hidden_size=hidden_size):
                with self.assertRaises(ValueError):
                    attach_flashindexer_gate(_indexer(n_heads, hidden_size))

    def test_rejects_non_cuda(self):
        with mock.patch.object(flashindexer, "is_cuda", lambda: False):
            with self.assertRaises(ValueError):
                attach_flashindexer_gate(_indexer())


class TestFlashIndexerSelectHeads(CustomTestCase):
    def test_no_gate_is_a_no_op(self):
        q, weights = torch.zeros(3, 64, 128), torch.zeros(3, 64)
        # Without a gate the hidden is never read, so a ROCm tuple is fine too.
        out = flashindexer_select_heads(_indexer(), q, weights, (torch.zeros(3), None))
        self.assertIs(out[0], q)
        self.assertIs(out[1], weights)

    def test_gate_without_hidden_fails(self):
        indexer = _indexer()
        indexer.linear_gate = torch.nn.Linear(128, 64, bias=False)
        with self.assertRaises(AssertionError):
            flashindexer_select_heads(
                indexer, torch.zeros(3, 64, 128), torch.zeros(3, 64), None
            )

    def test_slices_weights_and_hidden_to_the_q_tokens(self):
        indexer = _indexer()
        indexer.linear_gate = torch.nn.Linear(128, 64, bias=False)
        indexer.flashindexer_bias = torch.zeros(64)
        q = torch.zeros(3, 64, 128)
        # Graph prefill hands over q sliced to the real tokens, the rest padded.
        weights, hidden = torch.zeros(5, 64), torch.zeros(5, 128)
        with mock.patch.object(
            flashindexer, "flashindexer_select_heads_fused", return_value=("q", "w")
        ) as fused:
            self.assertEqual(
                flashindexer_select_heads(indexer, q, weights, hidden), ("q", "w")
            )
        args = fused.call_args.args
        self.assertIs(args[0], q)
        self.assertEqual((args[1].shape[0], args[4].shape[0]), (3, 3))
        self.assertIs(args[2], indexer.linear_gate.weight)
        self.assertIs(args[3], indexer.flashindexer_bias)


class _Mla:
    def __init__(self):
        self.indexer = object()


class TestErnieAttachesGates(CustomTestCase):
    """--enable-flashindexer gives a gate to every in-stage MLA indexer, only."""

    def _attached(self, enable_flashindexer):
        layers = [
            SimpleNamespace(self_attn=_Mla()),  # out of this PP stage
            SimpleNamespace(self_attn=_Mla()),
            SimpleNamespace(self_attn=object()),  # HCA layer: no DSA indexer
            SimpleNamespace(self_attn=_Mla()),
        ]
        pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
        attach = mock.MagicMock()
        patches = {
            "get_parallel": lambda: SimpleNamespace(pp_group=pp_group),
            "make_pp_layers": lambda *args, **kwargs: (layers, 1, 4),
            "Ernie5_5MlaAttention": _Mla,
            "attach_flashindexer_gate": attach,
        }
        for name, value in patches.items():
            self.enterContext(mock.patch.object(ernie55, name, value))
        config = SimpleNamespace(num_hidden_layers=4)
        with get_context().override_server_args(
            enable_flashindexer=enable_flashindexer
        ):
            ernie55.Ernie5_5Model(config, None, "model")
        expected = [layers[1].self_attn.indexer, layers[3].self_attn.indexer]
        return [call.args[0] for call in attach.call_args_list], expected

    def test_on_attaches_in_stage_mla_indexers(self):
        attached, expected = self._attached(True)
        self.assertEqual(attached, expected)

    def test_off_by_default(self):
        attached, _ = self._attached(False)
        self.assertEqual(attached, [])


def _indexer_loader(params):
    loader = object.__new__(ernie55.Ernie5_5WeightLoader)
    loader.params = params
    loader.loaded = set()
    loader.quant_config = None
    loader.pending_indexer_wk = {}
    loader.fused_indexer_halves = {}
    return loader


class TestLoaderFlashIndexerTensors(CustomTestCase):
    PREFIX = "model.layers.0."
    INDEXER = "self_attn.core_attention.indexer."

    def _load(self, loader, field, tensor):
        loader._load_layer_tensor(self.PREFIX, 0, f"{self.INDEXER}{field}", tensor)

    def test_flashindexer_tensors_load_into_the_indexer_params(self):
        gate = torch.nn.Parameter(torch.zeros(4, 8), requires_grad=False)
        bias = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        indexer = f"{self.PREFIX}self_attn.indexer."
        loader = _indexer_loader(
            {
                f"{indexer}{FLASHINDEXER_GATE_PARAM_SUFFIX}": gate,
                f"{indexer}{FLASHINDEXER_BIAS_PARAM_SUFFIX}": bias,
            }
        )
        self._load(loader, FLASHINDEXER_GATE_PARAM_SUFFIX, torch.ones(4, 8))
        self._load(loader, FLASHINDEXER_BIAS_PARAM_SUFFIX, torch.full((4,), 2.0))
        self.assertEqual(len(loader.loaded), 2)
        torch.testing.assert_close(gate, torch.ones(4, 8))
        torch.testing.assert_close(bias, torch.full((4,), 2.0))

    def test_a_stray_non_flashindexer_tensor_still_fails(self):
        with self.assertRaises(KeyError):
            self._load(_indexer_loader({}), "k_norm.weight", torch.ones(4))


class TestRaggedLogitsHeads(CustomTestCase):
    """_mqa_logits keeps FlashIndexer's 8 heads and still pads full heads below 32 to 32."""

    TOKENS, N_HEADS = 3, 16

    def _args(self, num_heads):
        return (
            torch.zeros(self.TOKENS, num_heads, 128),
            torch.zeros(5, 128),
            torch.ones(self.TOKENS, num_heads),
            torch.zeros(self.TOKENS, dtype=torch.int32),
            torch.zeros(self.TOKENS, dtype=torch.int32),
        )

    def _fp8_heads_seen(self, num_heads):
        seen = {}

        def fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits):
            seen["heads"] = q.shape[1], weights.shape[1]

        indexer = object.__new__(Indexer)
        indexer.n_heads = self.N_HEADS
        with mock.patch.object(
            dsa_indexer, "resolve_fp8_mqa_logits_fn", lambda: fp8_mqa_logits
        ):
            indexer._mqa_logits(*self._args(num_heads))
        return seen["heads"]

    def _fp4_heads_seen(self, num_heads):
        seen = {}

        def fp8_fp4_mqa_logits(q, kv, weights, ks, ke, clean_logits):
            seen["heads"] = q[0].shape[1], q[1].shape[1], weights.shape[1]

        indexer = object.__new__(Fp4Indexer)
        indexer.n_heads = self.N_HEADS
        indexer._quantize_fp4 = lambda q: (
            torch.zeros(self.TOKENS, num_heads, 64, dtype=torch.uint8),
            torch.zeros(self.TOKENS, num_heads, dtype=torch.int32),
        )
        kv = (torch.zeros(5, 64, dtype=torch.uint8), torch.zeros(5, dtype=torch.int32))
        deep_gemm = types.SimpleNamespace(fp8_fp4_mqa_logits=fp8_fp4_mqa_logits)
        args = list(self._args(num_heads))
        args[1] = kv
        with mock.patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            indexer._mqa_logits(*args)
        return seen["heads"]

    def test_full_heads_still_pad_to_32(self):
        self.assertEqual(self._fp8_heads_seen(self.N_HEADS), (32, 32))
        self.assertEqual(self._fp4_heads_seen(self.N_HEADS), (32, 32, 32))

    def test_flashindexer_selected_heads_are_not_padded(self):
        self.assertEqual(self._fp8_heads_seen(8), (8, 8))
        self.assertEqual(self._fp4_heads_seen(8), (8, 8, 8))


if __name__ == "__main__":
    unittest.main()
