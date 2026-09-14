"""Unit tests for backend-advertised decode graph capture widths.

The DSV4 attention backend may advertise narrower capture widths than the full
configured context; the decode graph runner captures one graph per width and
replays the smallest that fits. Selection lives on the backend, configuration
validation lives with the server arguments, and the runner only composes a
namespaced graph key.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.deepseek_v4_hook import (
    validate_deepseek_v4_decode_graph_widths,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DecodeGraphWidths,
    DeepseekV4AttnBackend,
    DSV4RawDecodeMetadata,
    DSV4RawVerifyMetadata,
    _resolve_decode_graph_seq_lens,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
    _compose_graph_width_variant,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

WIDTHS = (8192, 32768, 262148)


def _backend(**attrs):
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.decode_graph_widths = DecodeGraphWidths(widths=WIDTHS, label="dsv4_seq")
    backend._active_decode_graph_width = WIDTHS[-1]
    backend._full_seq_len_for_capture = WIDTHS[-1]
    for k, v in attrs.items():
        setattr(backend, k, v)
    return backend


def _runner_without_widths(**attrs):
    """A runner whose backend advertises nothing, i.e. the feature switched off."""
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.decode_graph_widths = None
    runner._active_decode_graph_width = None
    for k, v in attrs.items():
        setattr(runner, k, v)
    return runner


@contextmanager
def _capture_loop_isolated():
    """Stub the device and compile plumbing so the capture loop is observable."""

    @contextmanager
    def fake_patch_model(*_args, **_kwargs):
        yield "forward"

    mod = "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
    with patch(f"{mod}.get_available_gpu_memory", return_value=0.0), patch(
        f"{mod}.get_parallel", return_value=SimpleNamespace(tp_rank=1)
    ), patch(f"{mod}.torch_compile_decoration.patch_model", fake_patch_model), patch(
        f"{mod}._set_capture_lora_variant"
    ):
        yield


def _runner(**attrs):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.decode_graph_widths = DecodeGraphWidths(widths=WIDTHS, label="dsv4_seq")
    runner._active_decode_graph_width = None
    for k, v in attrs.items():
        setattr(runner, k, v)
    return runner


class TestResolveWidths(CustomTestCase):
    def test_environment_parses_an_integer_tuple(self):
        with envs.SGLANG_DSV4_DECODE_GRAPH_SEQ_LENS.override("32768, 8192"):
            self.assertEqual(
                envs.SGLANG_DSV4_DECODE_GRAPH_SEQ_LENS.get(), (32768, 8192)
            )

    def test_widths_are_sorted_deduplicated_and_bounded_by_full_context(self):
        self.assertEqual(
            _resolve_decode_graph_seq_lens(
                configured=(32768, 8192, 32768, 999999),
                full_seq_len=262148,
                is_xpu=True,
            ),
            WIDTHS,
        )

    def test_empty_configuration_preserves_existing_behaviour(self):
        self.assertEqual(
            _resolve_decode_graph_seq_lens(
                configured=(), full_seq_len=262148, is_xpu=True
            ),
            (),
        )

    def test_non_positive_widths_are_rejected(self):
        # int("-1") parses cleanly, so this guard is the only thing between a
        # typo and a graph captured at a nonsensical width.
        for bad in ((0,), (8192, -1)):
            with self.subTest(configured=bad):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    _resolve_decode_graph_seq_lens(
                        configured=bad, full_seq_len=262148, is_xpu=True
                    )

    def test_configured_widths_reject_non_xpu_platforms(self):
        with self.assertRaisesRegex(ValueError, "only supported on XPU"):
            _resolve_decode_graph_seq_lens(
                configured=(8192,), full_seq_len=262148, is_xpu=False
            )


class TestBackendSelectsTheWidth(CustomTestCase):
    def _batch(self, seq_lens, mode=ForwardMode.DECODE):
        return SimpleNamespace(seq_lens_cpu=seq_lens, forward_mode=mode)

    def test_smallest_fitting_width_is_chosen_at_the_boundaries(self):
        backend = _backend()
        for length, expected in (
            (1, 8192),
            (8192, 8192),
            (8193, 32768),
            (262148, 262148),
        ):
            with self.subTest(length=length):
                self.assertEqual(
                    backend.select_decode_graph_width(
                        self._batch(torch.tensor([length]))
                    ),
                    expected,
                )

    def test_missing_cpu_mirror_declines_a_graph(self):
        backend = _backend()
        self.assertIsNone(backend.select_decode_graph_width(self._batch(None)))

    def test_empty_idle_batch_takes_the_smallest_width(self):
        backend = _backend()
        self.assertEqual(
            backend.select_decode_graph_width(
                self._batch(torch.empty(0, dtype=torch.int64), ForwardMode.IDLE)
            ),
            8192,
        )

    def test_empty_non_idle_batch_declines_a_graph(self):
        backend = _backend()
        self.assertIsNone(
            backend.select_decode_graph_width(
                self._batch(torch.empty(0, dtype=torch.int64), ForwardMode.DECODE)
            )
        )

    def test_selection_does_not_disturb_the_active_width(self):
        backend = _backend()
        backend.select_decode_graph_width(self._batch(torch.tensor([1024])))
        self.assertEqual(backend.MAX_SEQ_LEN_FOR_CAPTURE, WIDTHS[-1])

    def test_active_width_controls_the_capture_bound(self):
        backend = _backend()
        backend.set_decode_graph_width(8192)
        self.assertEqual(backend.MAX_SEQ_LEN_FOR_CAPTURE, 8192)

    def test_raw_decode_conversion_uses_the_active_width(self):
        backend = _backend(
            req_to_token=torch.empty((1, 262148), dtype=torch.int32),
            token_to_kv_pool=object(),
            make_core_attn_metadata=Mock(return_value=object()),
            init_forward_metadata_indexer=Mock(return_value=None),
        )
        backend._active_decode_graph_width = 8192
        raw = DSV4RawDecodeMetadata(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([128], dtype=torch.int32),
            out_cache_loc=torch.tensor([0], dtype=torch.int64),
        )

        with patch(
            "sglang.srt.layers.attention.deepseek_v4_backend.create_paged_compressor_data",
            return_value=None,
        ):
            backend.make_forward_metadata_from_raw_decode(raw)

        self.assertEqual(
            backend.make_core_attn_metadata.call_args.kwargs["max_seq_len"], 8192
        )

    def test_raw_verify_conversion_is_bounded_by_the_full_context(self):
        # Forward guard. Today widths and speculative decoding cannot coexist
        # (validate_deepseek_v4_decode_graph_widths rejects the pair), so a
        # verify-shaped graph is never replayed at a narrowed width. Should
        # that validator ever be relaxed, the raw->full upgrade for verify
        # must still bound the page table by the full context -- a request at
        # seq_len + num_draft_tokens > width would otherwise be truncated
        # silently. Narrowing verify graphs is only valid together with a
        # draft-aware width selector; until both land, this line must not move.
        backend = _backend(
            req_to_token=torch.empty((1, 262148), dtype=torch.int32),
            token_to_kv_pool=object(),
            speculative_num_draft_tokens=2,
            make_core_attn_metadata=Mock(return_value=object()),
            init_forward_metadata_indexer=Mock(return_value=None),
            expand_extend_with_same_length=Mock(
                return_value=(
                    torch.tensor([[129, 130]], dtype=torch.int32),
                    torch.tensor([0, 0], dtype=torch.int32),
                )
            ),
        )
        backend._active_decode_graph_width = 8192
        raw = DSV4RawVerifyMetadata(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([128], dtype=torch.int32),
            out_cache_loc=torch.tensor([0, 0], dtype=torch.int64),
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
            total_verify_tokens=2,
        )

        with patch(
            "sglang.srt.layers.attention.deepseek_v4_backend.create_paged_compressor_data",
            return_value=None,
        ):
            backend.make_forward_metadata_from_raw_verify(raw)

        self.assertEqual(
            backend.make_core_attn_metadata.call_args.kwargs["max_seq_len"], 262148
        )


class TestTboWrapperDelegatesTheWidths(CustomTestCase):
    def test_the_wrapper_reports_the_primary_ladder_not_the_base_default(self):
        # Under two-batch-overlap the runner's attn_backend is TboAttnBackend,
        # which forwards unknown attributes to its primary via __getattr__.
        # A class-level default on AttentionBackend ends normal lookup first,
        # so __getattr__ never runs and the ladder would read as None -- the
        # feature silently off. TboAttnBackend therefore overrides it, the
        # same way it already overrides verify_mask.
        wrapper = TboAttnBackend.__new__(TboAttnBackend)
        wrapper.primary = _backend()

        self.assertIsNotNone(wrapper.decode_graph_widths)
        self.assertEqual(wrapper.decode_graph_widths.widths, WIDTHS)

        # Pin the delegation itself, not just the one attribute: the setter
        # reaches the primary only because the base declares no such method.
        # Adding one there would silently strand the primary at full width.
        wrapper.set_decode_graph_width(8192)
        self.assertEqual(wrapper.primary.MAX_SEQ_LEN_FOR_CAPTURE, 8192)
        self.assertEqual(wrapper.MAX_SEQ_LEN_FOR_CAPTURE, 8192)


class TestUnchangedWhenNoBackendAdvertisesWidths(CustomTestCase):
    """The feature must be inert for backends that advertise nothing.

    The helpers above always populate widths, so these three paths -- the
    capture loop, the graph key, and the eligibility check -- would otherwise
    go untested with the feature off, on every device that shares this runner.
    """

    def test_capture_widths_collapses_to_a_single_none(self):
        runner = _runner_without_widths(
            model_runner=SimpleNamespace(
                device="cpu", gpu_id=0, model=object(), tp_group=None
            ),
            capture_bs=[2, 1],
            compile_bs=set(),
            captured_req_width=1,
            attn_backend=Mock(),
            capture_one_shape=Mock(),
        )

        with _capture_loop_isolated():
            runner._capture_one_stream()

        # One capture per batch size, not one per width, and the backend is
        # never put on a width.
        self.assertEqual(runner.capture_one_shape.call_count, len(runner.capture_bs))
        runner.attn_backend.set_decode_graph_width.assert_not_called()
        self.assertIsNone(runner._active_decode_graph_width)

    def test_graph_key_is_what_it_was_before_the_feature(self):
        runner = _runner_without_widths()

        self.assertEqual(
            runner._make_graph_key(4, stream_idx=1, variant_label="lora"),
            ShapeKey(size=4, stream_idx=1, variant_label="lora"),
        )
        self.assertEqual(
            runner._make_graph_key(4), ShapeKey(size=4, stream_idx=None, variant_label=None)
        )

    def test_eligibility_never_consults_the_width_check(self):
        attn_backend = Mock()
        runner = _runner_without_widths(
            ragged_verify_mode=False,
            require_mlp_tp_gather=False,
            require_mlp_sync=False,
            is_encoder_decoder=False,
            enable_pdmux=False,
            enable_two_batch_overlap=False,
            disable_padding=False,
            max_bs=8,
            captured_req_width=1,
            attn_backend=attn_backend,
            _resolve_lora_variant=Mock(return_value=None),
            model_runner=SimpleNamespace(
                spec_algorithm=SimpleNamespace(is_ngram=lambda: False)
            ),
        )
        forward_batch = SimpleNamespace(
            replace_embeds=None, spec_info=None, batch_size=2
        )

        # Returns True, so it ran past the width branch rather than bailing
        # before it -- which is what makes the assertion below meaningful.
        self.assertTrue(runner.can_run_graph(forward_batch))
        attn_backend.select_decode_graph_width.assert_not_called()


class TestEagerFallbackUsesTheFullContext(CustomTestCase):
    def test_unknown_lengths_bound_by_full_context_not_the_active_width(self):
        # A narrowed width belongs to a graph replay. With no override and no
        # CPU mirror the real lengths are unknown, so an eager forward must be
        # bounded by the full context -- not by whatever width the last replay
        # left in force. Regressing this line is a silent under-sizing bug.
        backend = _backend(
            req_to_token=torch.empty((1, WIDTHS[-1]), dtype=torch.int32),
            page_size=256,
            swa_page_size=0,
            topk=0,
            speculative_num_steps=1,
            online_c128_mtp=Mock(prepare_forward=Mock(return_value=0)),
            init_forward_metadata_decode=Mock(return_value=object()),
        )
        backend.req_to_token_pool = SimpleNamespace(req_to_token=backend.req_to_token)
        backend.set_decode_graph_width(8192)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([128], dtype=torch.int32),
            seq_lens_cpu=None,
            out_cache_loc=torch.tensor([0], dtype=torch.int64),
        )

        mod = "sglang.srt.layers.attention.deepseek_v4_backend"
        with patch(
            f"{mod}._get_logical_forward_mode", return_value=ForwardMode.DECODE
        ), patch(f"{mod}._get_target_verify_bs", return_value=None):
            backend._build_forward_metadata(forward_batch)

        self.assertEqual(
            backend.init_forward_metadata_decode.call_args.kwargs["max_seq_len"],
            WIDTHS[-1],
        )


class TestRunnerComposesTheKey(CustomTestCase):
    def test_variant_preserves_lora_and_adds_the_backend_label(self):
        self.assertEqual(
            _compose_graph_width_variant("lora", "dsv4_seq", 8192),
            "lora|dsv4_seq=8192",
        )
        self.assertEqual(
            _compose_graph_width_variant(None, "dsv4_seq", 8192), "dsv4_seq=8192"
        )
        self.assertEqual(_compose_graph_width_variant("lora", "dsv4_seq", None), "lora")
        self.assertEqual(_compose_graph_width_variant("lora", None, 8192), "lora")

    def test_distinct_widths_produce_distinct_graph_keys(self):
        runner = _runner()
        runner._active_decode_graph_width = 8192
        low = runner._make_graph_key(4, stream_idx=1, variant_label="lora")
        runner._active_decode_graph_width = 32768
        high = runner._make_graph_key(4, stream_idx=1, variant_label="lora")

        self.assertEqual(low.variant_label, "lora|dsv4_seq=8192")
        self.assertNotEqual(low, high)

    def test_activation_restores_the_widest_width_when_none_fits(self):
        runner = _runner(attn_backend=Mock())
        runner.attn_backend.select_decode_graph_width.return_value = None

        self.assertIsNone(runner._activate_decode_graph_width(SimpleNamespace()))
        runner.attn_backend.set_decode_graph_width.assert_called_once_with(WIDTHS[-1])
        self.assertIsNone(runner._active_decode_graph_width)

    def test_activation_puts_the_backend_on_the_selected_width(self):
        runner = _runner(attn_backend=Mock())
        runner.attn_backend.select_decode_graph_width.return_value = 32768

        self.assertEqual(runner._activate_decode_graph_width(SimpleNamespace()), 32768)
        runner.attn_backend.set_decode_graph_width.assert_called_once_with(32768)
        self.assertEqual(runner._active_decode_graph_width, 32768)


class TestValidatorsLiveWithTheServerArgs(CustomTestCase):
    def _args(self, **overrides):
        base = dict(
            speculative_algorithm=None,
            dllm_algorithm=None,
            enable_pdmux=False,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=Backend.FULL)
            ),
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_unset_variable_accepts_any_configuration(self):
        with envs.SGLANG_DSV4_DECODE_GRAPH_SEQ_LENS.override(""):
            validate_deepseek_v4_decode_graph_widths(
                self._args(speculative_algorithm="EAGLE", enable_pdmux=True)
            )

    def test_supported_configuration_passes(self):
        with envs.SGLANG_DSV4_DECODE_GRAPH_SEQ_LENS.override("8192"):
            validate_deepseek_v4_decode_graph_widths(self._args())

    def test_unsupported_configurations_are_rejected(self):
        cases = (
            ("speculative decoding", {"speculative_algorithm": "EAGLE"}),
            ("diffusion decoding", {"dllm_algorithm": "LLaDA2Moe"}),
            ("PDMux", {"enable_pdmux": True}),
            (
                "full decode graph",
                {
                    "cuda_graph_config": SimpleNamespace(
                        decode=SimpleNamespace(backend=Backend.BREAKABLE)
                    )
                },
            ),
        )
        with envs.SGLANG_DSV4_DECODE_GRAPH_SEQ_LENS.override("8192"):
            for expected, overrides in cases:
                with self.subTest(case=expected):
                    with self.assertRaisesRegex(ValueError, expected):
                        validate_deepseek_v4_decode_graph_widths(
                            self._args(**overrides)
                        )


if __name__ == "__main__":
    unittest.main()
