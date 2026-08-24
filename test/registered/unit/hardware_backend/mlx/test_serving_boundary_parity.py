"""Parity between the MLX region and Torch at the boundary the sampler sees.

The region's validation harness compares the exported wrapper running on MLX
against the *same wrapper* running on Torch, so anything the wrapper gets
wrong relative to the real serving path cancels out of the comparison. The
padded-vocabulary bug lived in exactly that blind spot: the wrapper returned
``hidden @ lm_head.T`` at the lm_head's padded width while every other path in
SGLang reaches ``LogitsProcessor._copy_logits_to_buffer`` and is narrowed back
to ``config.vocab_size`` first.

These tests therefore take :class:`LogitsProcessor` itself as the oracle and
compare what each path hands the sampler, and they run on a fixture whose
vocabulary is deliberately *not* a multiple of the 64-row lm_head padding --
the three models the region is validated on (Qwen3-0.6B, Qwen2.5-Coder-1.5B,
SmolLM2-360M) all have zero padding rows, so none of them can express the bug.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.mlx.export_validation import (
    ServingForwardExportWrapper,
)
from sglang.srt.hardware_backend.mlx.fx_backend import MlxFxLoweringRegistry
from sglang.srt.hardware_backend.mlx.region_runner import _PAD_SINK_SLOT
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")

_HIDDEN = 8
# 61 is not a multiple of the 64-row lm_head padding, so the fixture carries
# three pad columns -- gpt2's shape (50257 -> 50304) in miniature.
_VOCAB_SIZE = 61
_PADDED_VOCAB = 64
_NUM_TOKENS = 4


class _StubBody(torch.nn.Module):
    """A decoder body the topology resolver accepts, returning fixed hidden.

    Only the logits boundary is under test, so the body ignores its inputs;
    it carries a RadixAttention block purely so ``resolve_decoder_topology``
    can find a layer stack the way it does on a real model.
    """

    def __init__(self, hidden_states: torch.Tensor):
        super().__init__()
        self.register_buffer("hidden_states", hidden_states, persistent=False)
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        block.self_attn.attn = RadixAttention(
            2, _HIDDEN, _HIDDEN**-0.5, num_kv_heads=2, layer_id=0
        )
        self.layers = torch.nn.ModuleList([block])

    def forward(self, input_ids, positions, forward_batch):
        return self.hidden_states


def _make_model(hidden_states: torch.Tensor, lm_head_weight: torch.Tensor):
    model = torch.nn.Module()
    model.body = _StubBody(hidden_states)
    model.lm_head = torch.nn.Linear(_HIDDEN, _PADDED_VOCAB, bias=False)
    with torch.no_grad():
        model.lm_head.weight.copy_(lm_head_weight)
    model.logits_processor = LogitsProcessor(
        SimpleNamespace(vocab_size=_VOCAB_SIZE, final_logit_softcapping=None),
        skip_all_gather=True,
        logit_scale=None,
    )
    return model


def _make_model_runner(model: torch.nn.Module):
    kv = torch.zeros(4, 2, _HIDDEN)
    return SimpleNamespace(
        model=model,
        attention_layers=[object()],
        attn_backend=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros(1, 4, dtype=torch.int32)
            ),
            token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda layer_id: (kv, kv)),
        ),
    )


_SERVING_ARGS = (
    torch.zeros(_NUM_TOKENS, dtype=torch.int64),
    torch.arange(_NUM_TOKENS, dtype=torch.int64),
    torch.zeros(_NUM_TOKENS, dtype=torch.int64),
    torch.full((_NUM_TOKENS,), _NUM_TOKENS, dtype=torch.int64),
    torch.arange(_NUM_TOKENS, dtype=torch.int64),
    None,
    None,
    None,
    None,
)


def _make_wrapper(hidden_states: torch.Tensor, lm_head_weight: torch.Tensor):
    model = _make_model(hidden_states, lm_head_weight)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens_sum=_NUM_TOKENS,
        extend_num_tokens=None,
        num_token_non_padded_cpu=_NUM_TOKENS,
    )
    wrapper = ServingForwardExportWrapper(
        model, _make_model_runner(model), forward_batch
    ).eval()
    return wrapper, model


def _region_logits(hidden_states: torch.Tensor, lm_head_weight: torch.Tensor):
    """What the region hands the sampler, and the processor it bypassed."""
    wrapper, model = _make_wrapper(hidden_states, lm_head_weight)
    with torch.no_grad():
        logits = wrapper(*_SERVING_ARGS)
    return logits, model.logits_processor, model.lm_head


class _DummyLogitsMetadata:
    """The subset of LogitsMetadata a single-rank ``_get_logits`` reads."""

    gathered_buffer = None
    next_token_logits_buffer = None

    def compute_dp_attention_metadata(self):
        pass


def _torch_logits(processor, lm_head, hidden_states: torch.Tensor):
    """What the standard Torch serving path hands the sampler."""
    with torch.no_grad():
        return processor._get_logits(
            hidden_states,
            lm_head,
            _DummyLogitsMetadata(),
            use_logits_buffer=False,
        )


class TestServingBoundaryLogitsParity(CustomTestCase):
    def setUp(self):
        super().setUp()
        # LogitsProcessor reads get_parallel()/get_exec() leaves, which fail
        # closed until a config is published. Publish for real rather than
        # standing a SimpleNamespace in for server_args.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_region_and_torch_hand_the_sampler_identical_logits(self):
        torch.manual_seed(0)
        hidden_states = torch.randn(_NUM_TOKENS, _HIDDEN)
        lm_head_weight = torch.randn(_PADDED_VOCAB, _HIDDEN)
        # SGLang never loads the pad rows, so they stay at their initialized
        # zeros on a real model; keep the fixture faithful to that.
        lm_head_weight[_VOCAB_SIZE:] = 0.0

        region, processor, lm_head = _region_logits(hidden_states, lm_head_weight)
        torch_logits = _torch_logits(processor, lm_head, hidden_states)

        # Precondition: a fixture whose vocabulary became a multiple of 64
        # would leave nothing for the assertions below to catch.
        self.assertGreater(lm_head.weight.shape[0], processor.vocab_size)
        self.assertEqual(region.shape, torch_logits.shape)
        self.assertEqual(region.shape[-1], _VOCAB_SIZE)
        torch.testing.assert_close(region, torch_logits)

    def test_all_negative_row_cannot_argmax_onto_a_pad_column(self):
        # The concrete failure: pad rows are zeros, so their logit is exactly
        # 0.0, and a row whose real logits are all negative used to argmax
        # onto a pad column. Reproduced on gpt2 with "The capital of France
        # is", which emitted token 50257 and decoded to nothing.
        torch.manual_seed(0)
        hidden_states = torch.ones(_NUM_TOKENS, _HIDDEN)
        lm_head_weight = torch.zeros(_PADDED_VOCAB, _HIDDEN)
        lm_head_weight[:_VOCAB_SIZE] = -torch.rand(_VOCAB_SIZE, _HIDDEN) - 0.1

        region, processor, lm_head = _region_logits(hidden_states, lm_head_weight)

        self.assertTrue(bool((region < 0).all()))
        self.assertTrue(bool((region.argmax(dim=-1) < _VOCAB_SIZE).all()))
        torch.testing.assert_close(
            region, _torch_logits(processor, lm_head, hidden_states)
        )

    def test_narrowing_survives_export_and_the_region_can_lower_it(self):
        # The region serves the *exported* graph, not the eager wrapper, and
        # a narrowing the MLX lowering did not recognize would push every
        # padded-vocabulary model onto the eager fallback instead of fixing
        # it. Assert the exported program narrows, and that every node in it
        # resolves to a lowering.
        torch.manual_seed(0)
        wrapper, _ = _make_wrapper(
            torch.randn(_NUM_TOKENS, _HIDDEN), torch.randn(_PADDED_VOCAB, _HIDDEN)
        )
        exported = torch.export.export(wrapper, _SERVING_ARGS, strict=False)

        self.assertEqual(exported.module()(*_SERVING_ARGS).shape[-1], _VOCAB_SIZE)
        registry = MlxFxLoweringRegistry.standard_export_decoder()
        unresolved = [
            str(node.target)
            for node in exported.graph_module.graph.nodes
            if registry.resolve(node) is None
        ]
        self.assertEqual(unresolved, [])


class TestPadSinkSlotIsNotAllocatable(CustomTestCase):
    """Padded prefill K/V must land somewhere the allocator never hands out.

    ``_pad_extend_batch`` used to send pad rows to ``token_to_kv_pool.size``.
    That index is in bounds -- the pools carry one extra row for padding --
    but it is the *last allocatable* slot, so pad K/V silently overwrote a
    live request once the pool neared exhaustion. Short-prompt tests never
    reach that slot because ``alloc`` takes from the front of the free list,
    which is why this test drains the pool instead of serving traffic.
    """

    def _drain(self, size: int) -> torch.Tensor:
        """Every slot the real allocator will ever hand out, in issue order."""
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        allocator = TokenToKVPoolAllocator(
            size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocated = allocator.alloc(size)
        self.assertIsNotNone(allocated)
        return allocated

    def test_the_allocator_reserves_the_sink_slot_and_nothing_else(self):
        size = 16
        allocated = sorted(self._drain(size).tolist())

        # The whole contract in one assertion: slot 0 is withheld and every
        # other index up to and including ``size`` is handed out. The right
        # half is what condemns the old sink -- ``token_to_kv_pool.size`` is
        # a live request's slot, not spare space past the range.
        self.assertEqual(allocated, list(range(1, size + 1)))
        self.assertNotIn(_PAD_SINK_SLOT, allocated)


if __name__ == "__main__":
    unittest.main()
