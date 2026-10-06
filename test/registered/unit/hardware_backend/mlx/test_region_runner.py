"""Gating and lifecycle contract of the decode MLX region runner.

``_forward_raw`` admits any ``is_cuda_graph()`` mode to the decode runner,
so every ``can_run_graph`` rejection below is the only barrier between an
unsupported serving shape and a region executor that would silently compute
the wrong thing (spec tokens, encoder batches, LoRA-adapted weights the
exported graph never saw). Each case guards one such silent-failure path.
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.mlx.export_validation import (
    ServingForwardArg,
    ServingForwardExportWrapper,
    resolve_body_call_kwargs,
    resolve_decoder_topology,
    serving_forward_args,
)
from sglang.srt.hardware_backend.mlx.region_config import (
    DEFAULT_MAX_DECODE_BATCH_SIZE,
    DEFAULT_MAX_PREFILL_TOKENS,
    apply_mps_region_backends,
    fill_mps_region_ladders,
    mlx_region_decode_batch_sizes,
    mlx_region_prefill_token_buckets,
)
from sglang.srt.hardware_backend.mlx.region_runner import (
    _PAD_SINK_SLOT,
    MlxRegionRunner,
    _configured_decode_batch_sizes,
    _configured_prefill_token_buckets,
    _kernel_contract_reject_reason,
    _nontrivial_logits_reason,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _make_model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(8, 16, bias=False)
    model.logits_processor = SimpleNamespace(
        logit_scale=None, final_logit_softcapping=None
    )
    return model


def _make_runner(*, lora_enabled: bool = False) -> MlxRegionRunner:
    runner = MlxRegionRunner.__new__(MlxRegionRunner)
    k_cache = torch.zeros(4, 2, 8)
    pool = SimpleNamespace(
        start_layer=0,
        get_kv_buffer=lambda layer_id: (k_cache, k_cache),
    )
    runner.model_runner = SimpleNamespace(
        token_to_kv_pool=pool,
        hisparse_coordinator=None,
        model=_make_model(),
        device="cpu",
        attn_backend=None,
    )
    runner._lora_enabled = lora_enabled
    runner._decode_batch_sizes = tuple(
        mlx_region_decode_batch_sizes(DEFAULT_MAX_DECODE_BATCH_SIZE)
    )
    runner._prefill_token_buckets = tuple(
        mlx_region_prefill_token_buckets(DEFAULT_MAX_PREFILL_TOKENS)
    )
    runner._prefill_batch_sizes = runner._decode_batch_sizes
    runner._executors = {}
    runner._failed_batch_sizes = set()
    runner._state_token = None
    runner._constants_checked = False
    runner._model_reject_reason = None
    return runner


def _decode_batch(**overrides) -> SimpleNamespace:
    fields = dict(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        spec_info=None,
        encoder_lens=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_rejects_modes_the_region_cannot_serve():
    # Dispatch admits TARGET_VERIFY/IDLE via is_cuda_graph() and MIXED via
    # is_extend(); only this gate keeps them off the region.
    runner = _make_runner()
    for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.IDLE, ForwardMode.MIXED):
        assert not runner.can_run_graph(_decode_batch(forward_mode=mode))


def _extend_batch(**overrides):
    fields = dict(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        spec_info=None,
        encoder_lens=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_logprob=False,
        input_embeds=None,
        replace_embeds=None,
        input_ids=torch.zeros(300, dtype=torch.int64),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_extend_gating_rejects_shapes_the_export_never_saw():
    runner = _make_runner()
    assert not runner.can_run_graph(_extend_batch(batch_size=2))
    assert not runner.can_run_graph(_extend_batch(return_logprob=True))
    assert not runner.can_run_graph(_extend_batch(input_embeds=object()))
    assert not runner.can_run_graph(
        _extend_batch(input_ids=torch.zeros(4096, dtype=torch.int64))
    )


def test_extend_key_buckets_token_count():
    runner = _make_runner()
    assert runner._executor_key(_extend_batch()) == ("extend", 384)
    assert runner._executor_key(
        _extend_batch(input_ids=torch.zeros(512, dtype=torch.int64))
    ) == ("extend", 512)


def test_rejects_request_shapes_the_export_never_saw():
    runner = _make_runner()
    assert not runner.can_run_graph(_decode_batch(spec_info=object()))
    assert not runner.can_run_graph(_decode_batch(encoder_lens=torch.ones(1)))
    assert not runner.can_run_graph(
        _decode_batch(capture_hidden_mode=CaptureHiddenMode.FULL)
    )
    assert not runner.can_run_graph(
        _decode_batch(batch_size=runner._decode_batch_sizes[-1] + 1)
    )
    assert not _make_runner(lora_enabled=True).can_run_graph(_decode_batch())


def test_failed_export_blacklists_the_batch_size_instead_of_retrying():
    runner = _make_runner()
    with mock.patch.object(
        MlxRegionRunner, "_ensure_executor", side_effect=RuntimeError("boom")
    ) as ensure:
        ensure.side_effect = None
        ensure.return_value = None
        assert not runner.can_run_graph(_decode_batch())
    runner._failed_batch_sizes.add(("decode", 1))
    # A blacklisted size must short-circuit before any export attempt.
    with mock.patch.object(MlxRegionRunner, "_ensure_executor") as ensure:
        assert not runner.can_run_graph(_decode_batch())
        ensure.assert_not_called()


def _run_ensure_with_blocked_export(runner):
    with (
        mock.patch(
            "sglang.srt.hardware_backend.mlx.export_validation.build_serving_mlx_executor",
            side_effect=RuntimeError("stop before real export"),
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mlx.export_validation.serving_export_context"
        ),
        mock.patch("sglang.srt.compilation.torch_compile_decoration._to_torch"),
    ):
        runner._ensure_executor(
            _decode_batch(input_ids=torch.zeros(1, dtype=torch.int64)),
            ("decode", 1),
        )


def test_pool_reallocation_invalidates_cached_executors():
    # Executors hold zero-copy views of the pool buffers; serving a stale
    # view after a pool reallocation would read freed storage silently.
    runner = _make_runner()
    runner._state_token = runner._state_identity()
    runner._executors[("decode", 1)] = object()
    runner._failed_batch_sizes.add(("decode", 2))

    new_cache = torch.zeros(4, 2, 8)
    runner.model_runner.token_to_kv_pool.get_kv_buffer = lambda layer_id: (
        new_cache,
        new_cache,
    )
    _run_ensure_with_blocked_export(runner)
    assert runner._executors == {}
    assert runner._failed_batch_sizes == {("decode", 1)}


def test_weight_replacement_invalidates_cached_executors():
    # A weight update that REPLACES parameter storage leaves the exported
    # views aliasing the old tensors; without invalidation the region keeps
    # serving the old model silently.
    runner = _make_runner()
    runner._state_token = runner._state_identity()
    runner._executors[("decode", 1)] = object()

    runner.model_runner.model.lm_head = torch.nn.Linear(8, 16, bias=False)
    _run_ensure_with_blocked_export(runner)
    assert runner._executors == {}


def test_in_place_weight_update_keeps_executors():
    # In-place updates keep the same storage, which the views alias, so the
    # region stays correct and must not pay a re-export.
    runner = _make_runner()
    runner._state_token = runner._state_identity()
    sentinel = object()
    runner._executors[("decode", 1)] = sentinel

    with torch.no_grad():
        runner.model_runner.model.lm_head.weight.copy_(
            torch.ones_like(runner.model_runner.model.lm_head.weight)
        )
    assert runner._state_identity() == runner._state_token
    executor = runner._ensure_executor(
        _decode_batch(input_ids=torch.zeros(1, dtype=torch.int64)),
        ("decode", 1),
    )
    assert executor is sentinel


def test_nontrivial_logits_processing_disables_the_region():
    # The wrapper computes hidden @ lm_head.T; logit_scale or softcapping in
    # the real LogitsProcessor would silently diverge inside the region.
    runner = _make_runner()
    runner.model_runner.model.logits_processor.final_logit_softcapping = 30.0
    from sglang.srt.hardware_backend.mlx.region_runner import (
        _nontrivial_logits_reason,
    )

    runner._model_reject_reason = _nontrivial_logits_reason(runner.model_runner.model)
    assert runner._model_reject_reason is not None
    assert not runner.can_run_graph(_decode_batch())


def _radix_block(
    *, num_heads: int = 2, num_kv_heads: int = 2, head_dim: int = 32
) -> torch.nn.Module:
    block = torch.nn.Module()
    self_attn = torch.nn.Module()
    self_attn.attn = RadixAttention(
        num_heads,
        head_dim,
        head_dim**-0.5,
        num_kv_heads=num_kv_heads,
        layer_id=0,
    )
    block.self_attn = self_attn
    return block


class TestStartupExport(CustomTestCase):
    """Startup export must reproduce the serve-time path shape for shape.

    The executor binds its input signature (dtypes, arities) at export, so a
    synthetic batch that differs from a real serving batch would export an
    executor that the first real batch of that shape cannot use -- the
    startup pass would then cost boot time and buy nothing. The ladder is
    the other failure mode: a startup pass that silently drifts from the
    configured buckets.
    """

    def _exported_keys(self) -> list:
        runner = _make_runner()
        seen = []

        def fake_ensure(self, batch, key):
            seen.append(key)
            return object()

        with (
            mock.patch.object(MlxRegionRunner, "_ensure_executor", fake_ensure),
            mock.patch.object(MlxRegionRunner, "execute", lambda self, b: None),
        ):
            runner._export_at_startup()
        return seen

    def test_startup_exports_both_configured_ladders(self):
        # CUDA-graph parity: every shape the region can serve is captured at
        # startup; there is no level knob and no lazy path for serving shapes.
        self.assertEqual(
            self._exported_keys(),
            [
                ("decode", bs)
                for bs in mlx_region_decode_batch_sizes(DEFAULT_MAX_DECODE_BATCH_SIZE)
            ]
            + [
                ("extend", b)
                for b in mlx_region_prefill_token_buckets(DEFAULT_MAX_PREFILL_TOKENS)
            ]
            + _make_runner()._packed_prefill_keys(),
        )

    def test_synthetic_batches_match_the_serving_signature(self):
        # Serving-path dtypes, measured on the real ForwardBatch: int64 for
        # ids/positions/indices/seq_lens/cache slots, int32 for the extend
        # metadata. The reserved rows (req 0, slot 0) are the only ones a
        # dummy batch may touch.
        runner = _make_runner()
        decode = runner._synthetic_batch(("decode", 4))
        self.assertEqual(runner._executor_key(decode), ("decode", 4))
        args = serving_forward_args(decode)
        for tensor in args[:5]:
            self.assertEqual(tensor.dtype, torch.int64)
            self.assertEqual(tensor.shape, (4,))
        self.assertTrue(bool((decode.req_pool_indices == 0).all()))
        self.assertTrue(bool((decode.out_cache_loc == 0).all()))

        extend = runner._synthetic_batch(("extend", 256))
        self.assertEqual(runner._executor_key(extend), ("extend", 256))
        self.assertEqual(extend.input_ids.shape, (256,))
        for name in ("extend_seq_lens", "extend_prefix_lens", "extend_start_loc"):
            self.assertEqual(getattr(extend, name).dtype, torch.int32)
        self.assertEqual(int(extend.extend_prefix_lens.max()), 0)
        # Already bucket-sized: serving must not pad it a second time.
        self.assertIs(runner._pad_extend_batch(extend, 256), extend)

    def test_warm_up_failure_blacklists_only_that_shape(self):
        runner = _make_runner()

        def fake_ensure(self, batch, key):
            self._executors[key] = object()
            return self._executors[key]

        def fake_execute(self, batch):
            if batch.forward_mode.is_decode() and batch.batch_size == 2:
                raise RuntimeError("metal dispatch failed")

        with (
            mock.patch.object(MlxRegionRunner, "_ensure_executor", fake_ensure),
            mock.patch.object(MlxRegionRunner, "execute", fake_execute),
        ):
            runner._export_at_startup()
        self.assertEqual(runner._failed_batch_sizes, {("decode", 2)})
        self.assertNotIn(("decode", 2), runner._executors)
        self.assertIn(("decode", 16), runner._executors)


class TestDecoderTopologyResolver(CustomTestCase):
    """Layer discovery must not assume where a model class hangs its stack.

    External testing found the previous ``model.model.layers`` assumption
    crashed setup for OPT (``model.decoder.layers``) and GPT-2
    (``transformer.h``); the resolver keys on the one shared invariant (a
    single block stack containing RadixAttention) instead.
    """

    def test_resolves_the_three_shipped_stack_layouts(self):
        llama_style = torch.nn.Module()
        llama_style.model = torch.nn.Module()
        llama_style.model.layers = torch.nn.ModuleList([_radix_block()])

        gpt2_style = torch.nn.Module()
        gpt2_style.transformer = torch.nn.Module()
        gpt2_style.transformer.h = torch.nn.ModuleList([_radix_block()])

        opt_style = torch.nn.Module()
        opt_style.model = torch.nn.Module()
        opt_style.model.decoder = torch.nn.Module()
        opt_style.model.decoder.layers = torch.nn.ModuleList([_radix_block()])

        self.assertEqual(resolve_decoder_topology(llama_style).body_name, "model")
        self.assertEqual(resolve_decoder_topology(gpt2_style).body_name, "transformer")
        self.assertEqual(resolve_decoder_topology(opt_style).body_name, "model")

    def test_rejects_zero_or_ambiguous_stacks_naming_the_model_class(self):
        class HeadlessModel(torch.nn.Module):
            pass

        with self.assertRaisesRegex(RuntimeError, "HeadlessModel"):
            resolve_decoder_topology(HeadlessModel())

        two_stacks = torch.nn.Module()
        two_stacks.model = torch.nn.Module()
        two_stacks.model.layers = torch.nn.ModuleList([_radix_block()])
        two_stacks.vision = torch.nn.Module()
        two_stacks.vision.layers = torch.nn.ModuleList([_radix_block()])
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            resolve_decoder_topology(two_stacks)


class TestBodyCallBinding(CustomTestCase):
    """The wrapper must bind body arguments by name, never positionally.

    External testing hit this on Phi: its body declares ``(input_ids,
    forward_batch, positions)``, so a positional ``(input_ids, positions,
    forward_batch)`` call handed the rotary embedding a ForwardBatch and
    crashed export. OPT additionally requires ``pp_proxy_tensors``.
    """

    def test_binds_phi_style_swapped_parameters_by_role(self):
        class PhiStyleBody(torch.nn.Module):
            def forward(self, input_ids, forward_batch, positions, inputs_embeds=None):
                return input_ids

        call_kwargs = resolve_body_call_kwargs(torch.nn.Module(), PhiStyleBody())
        self.assertEqual(call_kwargs["positions"], "positions")
        self.assertEqual(call_kwargs["forward_batch"], "forward_batch")
        self.assertEqual(call_kwargs["input_ids"], "input_ids")

    def test_binds_position_ids_and_required_pp_proxy(self):
        class Gpt2StyleBody(torch.nn.Module):
            def forward(self, input_ids, position_ids, forward_batch):
                return input_ids

        class OptStyleBody(torch.nn.Module):
            def forward(self, input_ids, positions, forward_batch, pp_proxy_tensors):
                return input_ids

        gpt2_kwargs = resolve_body_call_kwargs(torch.nn.Module(), Gpt2StyleBody())
        self.assertEqual(gpt2_kwargs["position_ids"], "positions")
        opt_kwargs = resolve_body_call_kwargs(torch.nn.Module(), OptStyleBody())
        self.assertIsNone(opt_kwargs["pp_proxy_tensors"])

    def test_rejects_unbindable_bodies_naming_the_model_class(self):
        class AlienBody(torch.nn.Module):
            def forward(self, tokens, batch):
                return tokens

        class AlienModel(torch.nn.Module):
            pass

        with self.assertRaisesRegex(RuntimeError, "AlienModel"):
            resolve_body_call_kwargs(AlienModel(), AlienBody())


def _contract_model_runner(
    *,
    num_heads: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 32,
    scaling: float | None = None,
    pool_dtype: torch.dtype = torch.bfloat16,
    model_dtype: torch.dtype = torch.bfloat16,
) -> SimpleNamespace:
    layer = RadixAttention(
        num_heads,
        head_dim,
        head_dim**-0.5 if scaling is None else scaling,
        num_kv_heads=num_kv_heads,
        layer_id=0,
    )
    k_cache = torch.zeros(4, num_kv_heads, max(head_dim, 1), dtype=pool_dtype)
    pool = SimpleNamespace(
        start_layer=0,
        get_kv_buffer=lambda layer_id: (k_cache, k_cache),
    )
    return SimpleNamespace(
        model=_make_model(),
        attention_layers=[layer],
        token_to_kv_pool=pool,
        dtype=model_dtype,
    )


class TestKernelContractPreLaunch(CustomTestCase):
    """Kernel-contract violations must reject at startup, not at dispatch.

    External testing hit the runtime path on a Mistral-architecture tiny
    checkpoint (head_dim 8): the deferred Metal kernel raised mid-serving.
    The invariant is that no batch reaches device work unless the kernel
    contract is known-satisfied, so the same constraints must fail
    ``_kernel_contract_reject_reason`` before any executor exists.
    """

    def test_rejects_head_dim_off_the_simdgroup_width_before_launch(self):
        reason = _kernel_contract_reject_reason(
            _contract_model_runner(num_heads=4, num_kv_heads=2, head_dim=8)
        )
        self.assertIn("simdgroup width", reason)

        runner = _make_runner()
        runner._model_reject_reason = reason
        self.assertFalse(runner.can_run_graph(_decode_batch()))
        self.assertFalse(runner.can_run_graph(_extend_batch()))

    def test_rejects_uneven_gqa_fanout(self):
        reason = _kernel_contract_reject_reason(
            _contract_model_runner(num_heads=3, num_kv_heads=2, head_dim=32)
        )
        self.assertIn("divide evenly", reason)

    def test_rejects_non_bf16_pools_and_activations(self):
        # The kernels only accept bfloat16; an fp16 model previously crashed
        # serving at the first region execute instead of falling back.
        reason = _kernel_contract_reject_reason(
            _contract_model_runner(pool_dtype=torch.float16, model_dtype=torch.float16)
        )
        self.assertIn("bfloat16", reason)

    def test_rejects_nonstandard_attention_scale(self):
        # The exported region bakes head_dim ** -0.5; the attention custom op
        # carries no scale, so a model with a different ``layer.scaling``
        # would be silently wrong rather than slow.
        reason = _kernel_contract_reject_reason(
            _contract_model_runner(head_dim=32, scaling=1.0)
        )
        self.assertIn("head_dim ** -0.5", reason)

    def test_accepts_the_supported_geometry(self):
        self.assertIsNone(
            _kernel_contract_reject_reason(
                _contract_model_runner(num_heads=4, num_kv_heads=2, head_dim=64)
            )
        )

    def test_topology_failure_surfaces_as_reject_reason(self):
        class HeadlessModel(torch.nn.Module):
            pass

        model_runner = SimpleNamespace(model=HeadlessModel())
        reason = _kernel_contract_reject_reason(model_runner)
        self.assertIn("decoder layer stack", reason)


class TestMissingLmHeadRejection(CustomTestCase):
    """Tied-embedding models without ``lm_head`` must reject, never crash.

    Gemma-2 exposes no ``lm_head`` attribute (logits go through
    ``model.embed_tokens``); the startup probe must return a reason for such
    a model instead of raising, and the runner must then refuse every batch.
    """

    def test_model_without_lm_head_rejects_cleanly(self):
        gemma2_style = torch.nn.Module()
        gemma2_style.logits_processor = SimpleNamespace(
            logit_scale=None, final_logit_softcapping=30.0
        )

        reason = _nontrivial_logits_reason(gemma2_style)
        self.assertIn("lm_head", reason)

        runner = _make_runner()
        runner.model_runner.model = gemma2_style
        runner._model_reject_reason = reason
        self.assertFalse(runner.can_run_graph(_decode_batch()))
        self.assertFalse(runner.can_run_graph(_extend_batch()))


class TestServingForwardArgLayout(CustomTestCase):
    # The executor indexes runtime inputs by ServingForwardArg member VALUES,
    # so the tuple layout must follow values regardless of how the enum
    # declarations are ordered; these tests make the enum a genuine single
    # source of truth for the signature.

    def test_values_are_contiguous_positions(self):
        self.assertEqual(
            sorted(arg.value for arg in ServingForwardArg),
            list(range(len(ServingForwardArg))),
        )

    def test_tuple_position_matches_member_value_and_field_name(self):
        # Sampling indices are an explicit argument derived by the runner;
        # all other positions come directly from ForwardBatch fields.
        batch = SimpleNamespace(
            **{arg.name.lower(): arg.name.lower() for arg in ServingForwardArg}
        )
        args = serving_forward_args(batch, batch.sampling_indices)
        self.assertEqual(len(args), len(ServingForwardArg))
        for arg in ServingForwardArg:
            self.assertEqual(args[arg.value], arg.name.lower())


class TestRegionLadders(CustomTestCase):
    """MPS-sized default ladders: the same flags as CUDA, smaller defaults."""

    def test_defaults_reproduce_the_shipped_ladders(self):
        self.assertEqual(
            mlx_region_decode_batch_sizes(DEFAULT_MAX_DECODE_BATCH_SIZE),
            [1, 2, 4, 8, 12, 16],
        )
        self.assertEqual(
            mlx_region_prefill_token_buckets(DEFAULT_MAX_PREFILL_TOKENS),
            [128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
        )

    def test_decode_ladder_matches_the_cuda_shape_and_ends_at_the_cap(self):
        self.assertEqual(
            mlx_region_decode_batch_sizes(64),
            [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
        )
        for cap in (10, 3, 1):
            self.assertEqual(mlx_region_decode_batch_sizes(cap)[-1], cap)

    def test_prefill_ladder_grows_shrinks_and_ends_at_the_cap(self):
        self.assertEqual(
            mlx_region_prefill_token_buckets(1024),
            [128, 192, 256, 384, 512, 768, 1024],
        )
        self.assertEqual(mlx_region_prefill_token_buckets(4096)[-2:], [3072, 4096])
        # Otherwise a batch at exactly the cap exceeds the largest bucket and
        # silently falls back to eager, which is the opposite of the ask.
        for cap in (3000, 64):
            self.assertEqual(mlx_region_prefill_token_buckets(cap)[-1], cap)

    def test_rejects_nonpositive_caps(self):
        for cap in (0, -5, None):
            with self.assertRaises(ValueError):
                mlx_region_decode_batch_sizes(cap)
            with self.assertRaises(ValueError):
                mlx_region_prefill_token_buckets(cap)


class TestLaddersReadFromCudaGraphConfig(CustomTestCase):
    """The runner reads cuda_graph_config like the CUDA runners, never the
    ServerArgs instance; publish for real rather than faking server_args."""

    def _publish(self, decode=None, prefill=None):
        override = get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.FULL, **(decode or {})),
                prefill=PhaseConfig(backend=Backend.FULL, **(prefill or {})),
            )
        )
        override.install()
        self.addCleanup(override.restore)

    @staticmethod
    def _model_runner(pool_size):
        return SimpleNamespace(req_to_token_pool=SimpleNamespace(size=pool_size))

    def test_explicit_lists_are_sorted_and_deduplicated(self):
        self._publish(decode=dict(bs=[6, 2, 6]), prefill=dict(bs=[640, 160, 640]))
        self.assertEqual(_configured_decode_batch_sizes(self._model_runner(64)), (2, 6))
        self.assertEqual(_configured_prefill_token_buckets(), (160, 640))

    def test_decode_ladder_is_clamped_to_the_request_pool(self):
        # Mirrors get_batch_sizes_to_capture: a pool smaller than the ladder
        # still gets its full size as a bucket.
        self._publish(decode=dict(bs=[1, 2, 4, 8, 12, 16]))
        self.assertEqual(
            _configured_decode_batch_sizes(self._model_runner(5)), (1, 2, 4, 5)
        )
        self.assertEqual(
            _configured_decode_batch_sizes(self._model_runner(8)), (1, 2, 4, 8)
        )

    def test_unset_or_invalid_lists_fail_loudly(self):
        self._publish(decode=dict(bs=None), prefill=dict(bs=[0, 128]))
        with self.assertRaises(ValueError):
            _configured_decode_batch_sizes(self._model_runner(64))
        with self.assertRaises(ValueError):
            _configured_prefill_token_buckets()


class TestMpsRegionGraphConfig(CustomTestCase):
    """On MPS the region is cuda_graph_config's graph runner: resolution keeps
    Backend.FULL and fills MPS-sized ladders unless the user set them."""

    def test_unset_config_gets_the_mps_defaults(self):
        config = CudaGraphConfig()
        fill_mps_region_ladders(config)
        self.assertEqual(config.decode.max_bs, DEFAULT_MAX_DECODE_BATCH_SIZE)
        self.assertEqual(config.decode.bs, [1, 2, 4, 8, 12, 16])
        self.assertEqual(config.prefill.max_bs, DEFAULT_MAX_PREFILL_TOKENS)
        self.assertEqual(
            config.prefill.bs, [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
        )

    def test_explicit_lists_win_and_pin_the_cap(self):
        # --cuda-graph-bs-decode 6 2 / --cuda-graph-bs-prefill 640 160
        config = CudaGraphConfig(
            decode=PhaseConfig(bs=[6, 2]), prefill=PhaseConfig(bs=[640, 160])
        )
        fill_mps_region_ladders(config)
        self.assertEqual((config.decode.bs, config.decode.max_bs), ([6, 2], 6))
        self.assertEqual((config.prefill.bs, config.prefill.max_bs), ([640, 160], 640))

    def test_explicit_caps_get_the_mps_ladder_up_to_them(self):
        # --cuda-graph-max-bs-decode 32 / --cuda-graph-max-bs-prefill 512
        config = CudaGraphConfig(
            decode=PhaseConfig(max_bs=32), prefill=PhaseConfig(max_bs=512)
        )
        fill_mps_region_ladders(config)
        self.assertEqual(config.decode.bs, [1, 2, 4, 8, 12, 16, 24, 32])
        self.assertEqual(config.prefill.bs, [128, 192, 256, 384, 512])

    def test_default_prefill_cap_never_exceeds_max_total_tokens(self):
        config = CudaGraphConfig()
        fill_mps_region_ladders(config, max_total_tokens=300)
        self.assertEqual(config.prefill.max_bs, 300)
        self.assertEqual(config.prefill.bs, [128, 192, 256, 300])

    def test_unpinned_backends_become_full(self):
        config = CudaGraphConfig()  # decode defaults to full, prefill to tc_piecewise
        apply_mps_region_backends(config, locked=set())
        self.assertEqual(config.decode.backend, Backend.FULL)
        self.assertEqual(config.prefill.backend, Backend.FULL)

    def test_pinned_disable_is_respected_per_phase(self):
        # --cuda-graph-backend-decode=disabled
        config = CudaGraphConfig(decode=PhaseConfig(backend=Backend.DISABLED))
        apply_mps_region_backends(config, locked={(Phase.DECODE, "backend")})
        self.assertEqual(config.decode.backend, Backend.DISABLED)
        self.assertEqual(config.prefill.backend, Backend.FULL)

    def test_pinned_backends_without_an_mps_implementation_are_rejected(self):
        for backend in (Backend.TC_PIECEWISE, Backend.BREAKABLE):
            config = CudaGraphConfig(prefill=PhaseConfig(backend=backend))
            with self.assertRaises(ValueError):
                apply_mps_region_backends(config, locked={(Phase.PREFILL, "backend")})


class TestDecodeBatchBucketing(CustomTestCase):
    """Decode is padded onto an exported bucket ladder, like CUDA-graph replay."""

    def test_batch_routes_to_the_smallest_covering_bucket(self):
        runner = _make_runner()
        for batch_size, bucket in ((1, 1), (3, 4), (12, 12), (13, 16), (16, 16)):
            self.assertEqual(
                runner._executor_key(_decode_batch(batch_size=batch_size)),
                ("decode", bucket),
            )
        self.assertIsNone(runner._executor_key(_decode_batch(batch_size=17)))

    def _three_request_batch(self, runner):
        batch = runner._synthetic_batch(("decode", 3))
        return dataclasses.replace(
            batch,
            input_ids=torch.tensor([11, 12, 13]),
            positions=torch.tensor([9, 19, 29]),
            req_pool_indices=torch.tensor([5, 6, 7]),
            seq_lens=torch.tensor([10, 20, 30]),
            seq_lens_cpu=torch.tensor([10, 20, 30]),
            seq_lens_sum=60,
            out_cache_loc=torch.tensor([100, 200, 300]),
        )

    def test_pad_rows_are_the_reserved_dummy_request(self):
        runner = _make_runner()
        batch = self._three_request_batch(runner)
        padded = runner._pad_decode_batch(batch, 4)
        self.assertEqual(padded.batch_size, 4)
        # Real rows untouched, in place.
        self.assertEqual(padded.req_pool_indices[:3].tolist(), [5, 6, 7])
        self.assertEqual(padded.out_cache_loc[:3].tolist(), [100, 200, 300])
        self.assertEqual(padded.seq_lens[:3].tolist(), [10, 20, 30])
        # Pad row: req 0, length 1 (no history), K/V to the reserved sink slot.
        self.assertEqual(int(padded.req_pool_indices[3]), 0)
        self.assertEqual(int(padded.seq_lens[3]), 1)
        self.assertEqual(int(padded.seq_lens_cpu[3]), 1)
        self.assertEqual(int(padded.out_cache_loc[3]), _PAD_SINK_SLOT)
        self.assertEqual(padded.seq_lens_sum, 61)
        # Exact-size batches are passed through untouched.
        self.assertIs(runner._pad_decode_batch(batch, 3), batch)

    def test_execute_pads_the_call_and_drops_pad_logits(self):
        runner = _make_runner()
        seen = []

        def fake_execute(*args):
            seen.append(args)
            return torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7)

        runner._executors[("decode", 4)] = SimpleNamespace(execute=fake_execute)
        out = runner.execute(self._three_request_batch(runner))
        self.assertEqual(seen[0][ServingForwardArg.REQ_POOL_INDICES].shape, (4,))
        self.assertEqual(out.next_token_logits.shape, (3, 7))
        self.assertTrue(
            torch.equal(
                out.next_token_logits,
                torch.arange(21, dtype=torch.float32).reshape(3, 7),
            )
        )


class TestPrefillLogitsSelection(CustomTestCase):
    """Only sampling rows should reach the expensive vocabulary projection."""

    def _wrapper(self, batch):
        class Body(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([_radix_block()])
                self.embed_tokens = torch.nn.Embedding(16, 8)

            def forward(self, input_ids, positions, forward_batch):
                return self.embed_tokens(input_ids)

        model = _make_model()
        model.model = Body()
        model.logits_processor.vocab_size = 13
        runner = _make_runner().model_runner
        runner.attention_layers = []
        runner.attn_backend = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(1, 8)),
        )
        return ServingForwardExportWrapper(model, runner, batch).eval()

    def test_prefill_selects_each_requests_last_hidden_state_before_projection(self):
        runner = _make_runner()
        batch = dataclasses.replace(
            runner._synthetic_batch(("extend", 6)),
            batch_size=3,
            input_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
            req_pool_indices=torch.tensor([1, 2, 3]),
            seq_lens=torch.tensor([11, 3, 22]),
            extend_seq_lens=torch.tensor([1, 3, 2], dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([10, 0, 20], dtype=torch.int32),
        )
        wrapper = self._wrapper(batch)
        with torch.no_grad():
            full = wrapper.model.lm_head(
                wrapper.model.model.embed_tokens(batch.input_ids)
            )
            actual = wrapper(*serving_forward_args(batch))
        torch.testing.assert_close(actual, full[[0, 3, 5], :13])
        self.assertEqual(actual.shape, (3, 13))

    def test_export_reads_real_length_at_runtime_and_never_projects_pad_rows(self):
        runner = _make_runner()
        captured = runner._synthetic_batch(("extend", 8))
        wrapper = self._wrapper(captured)
        exported = torch.export.export(
            wrapper, serving_forward_args(captured), strict=False
        )
        # The same export serves shorter cold prompts and cached-prefix
        # continuations. Total sequence length must not select a packed row.
        for length, prefix in ((1, 0), (3, 0), (3, 20), (8, 0)):
            with self.subTest(length=length, prefix=prefix):
                real = dataclasses.replace(
                    runner._synthetic_batch(("extend", length)),
                    input_ids=torch.arange(1, length + 1),
                    seq_lens=torch.tensor([length + prefix]),
                    extend_prefix_lens=torch.tensor([prefix], dtype=torch.int32),
                )
                padded = runner._pad_extend_batch(real, 8)
                with torch.no_grad():
                    full = wrapper.model.lm_head(
                        wrapper.model.model.embed_tokens(padded.input_ids)
                    )
                    actual = exported.module()(*serving_forward_args(padded))
                torch.testing.assert_close(actual, full[length - 1 : length, :13])
        projections = [
            node
            for node in exported.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.matmul.default
        ]
        self.assertEqual(len(projections), 1)
        self.assertEqual(tuple(projections[0].meta["val"].shape), (1, 16))

    def test_decode_preserves_all_request_rows(self):
        batch = dataclasses.replace(
            _make_runner()._synthetic_batch(("decode", 3)),
            input_ids=torch.tensor([1, 2, 3]),
        )
        wrapper = self._wrapper(batch)
        with torch.no_grad():
            expected = wrapper.model.lm_head(
                wrapper.model.model.embed_tokens(batch.input_ids)
            )[:, :13]
            actual = wrapper(*serving_forward_args(batch))
        torch.testing.assert_close(actual, expected)

    def test_explicit_sampling_rows_survive_export_for_segment_padding(self):
        runner = _make_runner()
        packed = dataclasses.replace(
            runner._synthetic_batch(("extend", 12)),
            batch_size=3,
            input_ids=torch.arange(1, 13),
            req_pool_indices=torch.arange(3),
            extend_seq_lens=torch.full((3,), 4, dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        )
        wrapper = self._wrapper(packed)
        indices = torch.tensor([3, 7, 11])
        exported = torch.export.export(
            wrapper, serving_forward_args(packed, indices), strict=False
        )
        for live in (indices, torch.tensor([1, 5, 9])):
            with torch.no_grad():
                full = wrapper.model.lm_head(
                    wrapper.model.model.embed_tokens(packed.input_ids)
                )
                actual = exported.module()(*serving_forward_args(packed, live))
            torch.testing.assert_close(actual, full[live, :13])

    def test_execute_keeps_the_already_selected_logits_row_for_a_padded_prompt(self):
        runner = _make_runner()
        batch = dataclasses.replace(
            runner._synthetic_batch(("extend", 300)),
            out_cache_loc=torch.arange(1, 301),
        )
        logits = torch.randn(1, 13)
        seen = []

        def execute(*args):
            seen.append(args)
            return logits

        runner._executors[("extend", 384)] = SimpleNamespace(execute=execute)
        actual = runner.execute(batch)
        self.assertIs(actual.next_token_logits, logits)
        self.assertEqual(seen[0][ServingForwardArg.INPUT_IDS].shape, (384,))
        self.assertEqual(seen[0][ServingForwardArg.EXTEND_SEQ_LENS].tolist(), [300])
        self.assertTrue(
            bool(
                (seen[0][ServingForwardArg.OUT_CACHE_LOC][300:] == _PAD_SINK_SLOT).all()
            )
        )


class TestPackedPrefillBucketing(CustomTestCase):
    def _runner(self):
        runner = _make_runner()
        runner._prefill_batch_sizes = (1, 2, 4)
        runner._prefill_token_buckets = (7, 28)
        return runner

    def _batch(self, runner, lengths=(7, 7, 7), prefixes=None):
        lengths = list(lengths)
        prefixes = [0] * len(lengths) if prefixes is None else list(prefixes)
        size = sum(lengths)
        return dataclasses.replace(
            runner._synthetic_batch(("extend", size)),
            batch_size=len(lengths),
            input_ids=torch.arange(1, size + 1),
            positions=torch.cat([torch.arange(n) for n in lengths]),
            req_pool_indices=torch.arange(5, 5 + len(lengths)),
            seq_lens=torch.tensor([n + p for n, p in zip(lengths, prefixes)]),
            seq_lens_cpu=torch.tensor([n + p for n, p in zip(lengths, prefixes)]),
            out_cache_loc=torch.arange(101, 101 + size),
            extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens=torch.tensor(prefixes, dtype=torch.int32),
            extend_prefix_lens_cpu=prefixes,
            extend_start_loc=torch.tensor(
                [sum(lengths[:i]) for i in range(len(lengths))], dtype=torch.int32
            ),
        )

    def test_startup_shapes_are_bounded_by_existing_token_and_request_buckets(self):
        runner = self._runner()
        self.assertEqual(
            runner._packed_prefill_keys(),
            [("extend_batch", 2, 7), ("extend_batch", 4, 7)],
        )
        for key in runner._packed_prefill_keys():
            synthetic = runner._synthetic_batch(key)
            self.assertEqual(runner._executor_key(synthetic), key)
            self.assertEqual(synthetic.input_ids.numel(), key[1] * key[2])
            self.assertTrue(bool((synthetic.req_pool_indices == 0).all()))
            self.assertTrue(bool((synthetic.out_cache_loc == _PAD_SINK_SLOT).all()))

    def test_selects_the_smallest_segmented_shape_that_covers_the_batch(self):
        runner = self._runner()
        self.assertEqual(
            runner._executor_key(self._batch(runner)), ("extend_batch", 4, 7)
        )
        self.assertEqual(
            runner._executor_key(self._batch(runner, (5, 5))), ("extend_batch", 2, 7)
        )
        self.assertIsNone(runner._executor_key(self._batch(runner, (8, 8))))
        self.assertIsNone(runner._executor_key(self._batch(runner, (5,) * 5)))

    def test_unsupported_metadata_and_semantics_keep_the_fallback(self):
        runner = self._runner()
        cases = [
            self._batch(runner, (3, 5)),
            self._batch(runner, prefixes=(0, 1, 0)),
            dataclasses.replace(self._batch(runner), extend_seq_lens_cpu=None),
            dataclasses.replace(self._batch(runner), extend_prefix_lens_cpu=[0]),
            dataclasses.replace(
                self._batch(runner), input_ids=torch.zeros(14, dtype=torch.int64)
            ),
            dataclasses.replace(self._batch(runner), return_logprob=True),
            dataclasses.replace(self._batch(runner), forward_mode=ForwardMode.MIXED),
            self._batch(runner, (5, 5, 5)),  # Excessive shape padding stays eager.
        ]
        for batch in cases:
            with self.subTest(batch=batch):
                self.assertIsNone(runner._executor_key(batch))

    def test_padding_preserves_real_rows_and_sends_every_dummy_kv_to_reserved_slot(
        self,
    ):
        runner = self._runner()
        original = self._batch(runner, (5, 5, 5))
        padded, sampling_indices = runner._pad_packed_extend_batch(original, 4, 7)
        self.assertEqual(
            padded.input_ids.reshape(4, 7).tolist(),
            [
                [1, 2, 3, 4, 5, 0, 0],
                [6, 7, 8, 9, 10, 0, 0],
                [11, 12, 13, 14, 15, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        )
        slots = padded.out_cache_loc.reshape(4, 7)
        torch.testing.assert_close(slots[:3, :5].reshape(-1), original.out_cache_loc)
        self.assertTrue(bool((slots[:3, 5:] == _PAD_SINK_SLOT).all()))
        self.assertTrue(bool((slots[3] == _PAD_SINK_SLOT).all()))
        self.assertEqual(padded.req_pool_indices.tolist(), [5, 6, 7, 0])
        self.assertEqual(padded.extend_start_loc.tolist(), [0, 7, 14, 21])
        self.assertEqual(padded.extend_seq_lens.tolist(), [7] * 4)
        self.assertEqual(sampling_indices[:3].tolist(), [4, 11, 18])
        self.assertEqual(padded.extend_num_tokens, 28)
        self.assertEqual(
            padded.positions.reshape(4, 7)[:3, :5].tolist(), [list(range(5))] * 3
        )

    def test_execute_passes_explicit_sampling_rows_and_discards_dummy_requests(self):
        runner = self._runner()
        batch = self._batch(runner)
        logits = torch.arange(4 * 13).reshape(4, 13)
        seen = []

        def execute(*args):
            seen.append(args)
            return logits

        runner._executors[("extend_batch", 4, 7)] = SimpleNamespace(execute=execute)
        output = runner.execute(batch)
        torch.testing.assert_close(output.next_token_logits, logits[:3])
        self.assertEqual(
            seen[0][ServingForwardArg.SAMPLING_INDICES][:3].tolist(), [6, 13, 20]
        )
        self.assertEqual(seen[0][ServingForwardArg.INPUT_IDS].shape, (28,))


if __name__ == "__main__":
    unittest.main()
