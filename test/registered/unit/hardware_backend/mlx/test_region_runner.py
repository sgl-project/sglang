"""Gating and lifecycle contract of the decode MLX region runner.

``_forward_raw`` admits any ``is_cuda_graph()`` mode to the decode runner,
so every ``can_run_graph`` rejection below is the only barrier between an
unsupported serving shape and a region executor that would silently compute
the wrong thing (spec tokens, encoder batches, LoRA-adapted weights the
exported graph never saw). Each case guards one such silent-failure path.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import MlxRegionStartupExport, envs
from sglang.srt.hardware_backend.mlx.export_validation import (
    ServingForwardArg,
    resolve_body_call_kwargs,
    resolve_decoder_topology,
    serving_forward_args,
)
from sglang.srt.hardware_backend.mlx.region_runner import (
    _MAX_REGION_BATCH_SIZE,
    _PREFILL_TOKEN_BUCKETS,
    _STARTUP_DECODE_BATCH_SIZES,
    MlxRegionRunner,
    _kernel_contract_reject_reason,
    _nontrivial_logits_reason,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
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
        _decode_batch(batch_size=_MAX_REGION_BATCH_SIZE + 1)
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
    with mock.patch(
        "sglang.srt.hardware_backend.mlx.export_validation.build_serving_mlx_executor",
        side_effect=RuntimeError("stop before real export"),
    ), mock.patch(
        "sglang.srt.hardware_backend.mlx.export_validation.serving_export_context"
    ), mock.patch(
        "sglang.srt.compilation.torch_compile_decoration._to_torch"
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
    startup pass would then cost boot time and buy nothing. The level knob
    is the other failure mode: a ladder that silently drifts from the
    documented levels.
    """

    def _exported_keys(self, level) -> list:
        runner = _make_runner()
        seen = []

        def fake_ensure(self, batch, key):
            seen.append(key)
            return object()

        with (
            envs.SGLANG_MLX_REGION_STARTUP_EXPORT.override(int(level)),
            mock.patch.object(MlxRegionRunner, "_ensure_executor", fake_ensure),
            mock.patch.object(MlxRegionRunner, "execute", lambda self, b: None),
        ):
            runner._export_at_startup()
        return seen

    def test_levels_select_the_documented_ladders(self):
        self.assertEqual(self._exported_keys(MlxRegionStartupExport.OFF), [])
        self.assertEqual(
            self._exported_keys(MlxRegionStartupExport.DECODE),
            [("decode", bs) for bs in _STARTUP_DECODE_BATCH_SIZES],
        )
        self.assertEqual(
            self._exported_keys(MlxRegionStartupExport.ALL),
            [("decode", bs) for bs in _STARTUP_DECODE_BATCH_SIZES]
            + [("extend", b) for b in _PREFILL_TOKEN_BUCKETS],
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
            if batch.batch_size == 2:
                raise RuntimeError("metal dispatch failed")

        with (
            envs.SGLANG_MLX_REGION_STARTUP_EXPORT.override(
                int(MlxRegionStartupExport.DECODE)
            ),
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
        # Position arg.value must hold the ForwardBatch field named
        # arg.name.lower(); sentinel strings make any permutation visible.
        batch = SimpleNamespace(
            **{arg.name.lower(): arg.name.lower() for arg in ServingForwardArg}
        )
        args = serving_forward_args(batch)
        self.assertEqual(len(args), len(ServingForwardArg))
        for arg in ServingForwardArg:
            self.assertEqual(args[arg.value], arg.name.lower())


if __name__ == "__main__":
    unittest.main()
