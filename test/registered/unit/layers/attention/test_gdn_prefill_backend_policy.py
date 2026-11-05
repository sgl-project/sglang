import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
    _validate_gdn_linear_attn_backends,
    flashinfer_gdn_prefill_default,
    validate_gdn_mis_backend,
)
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    maybe_build_flashinfer_checkpoint_plan,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    resolve_linear_attn_backends,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _publish(testcase, **fields):
    """Install a published config for one case and restore on its cleanup --
    a failed case must not leave a partial publish for a later file in a
    monolithic local run."""
    from sglang.srt.runtime_context import get_context, get_server_args

    override = get_context().override_server_args(**fields)
    override.install()
    testcase.addCleanup(override.restore)
    return get_server_args()


def make_runner(
    testcase,
    *,
    state_dtype=torch.bfloat16,
    key_dim=128,
    value_dim=128,
    **arg_overrides,
):
    # The policy reads the published bags, so the fixture publishes the
    # configuration under test.
    fields = dict(
        linear_attn_backend="triton",
        linear_attn_prefill_backend=None,
        uses_mamba_radix_cache=False,
        enable_page_major_kv_layout=False,
        mamba_radix_cache_strategy="no_buffer",
        enable_dynamic_chunking=False,
        chunked_prefill_size=8192,
        enable_mis=False,
    )
    fields.update(arg_overrides)
    args = _publish(testcase, **fields)

    return SimpleNamespace(
        server_args=args,
        model_config=SimpleNamespace(),
        hybrid_gdn_config=SimpleNamespace(
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
        ),
        token_to_kv_pool=SimpleNamespace(
            layer_router=SimpleNamespace(
                part_view=lambda part: SimpleNamespace(
                    mamba_cache=SimpleNamespace(
                        temporal=SimpleNamespace(dtype=state_dtype)
                    )
                )
            )
        ),
    )


class TestGDNPrefillGraphExtend(CustomTestCase):
    @contextmanager
    def _setup(self, *, cuda=True, enable_mis=False, draft=False):
        """A target's GDN backend on a setup the graph-safe chain supports, unless
        cuda, enable_mis or draft say otherwise."""
        backend = GDNAttnBackend.__new__(GDNAttnBackend)
        backend.is_draft_worker = draft
        backend.enable_mis = enable_mis
        backend._mamba_chunk_size = 64
        backend.kernel_dispatcher = SimpleNamespace(
            extend_kernel=TritonGDNKernel.__new__(TritonGDNKernel)
        )
        cache = SimpleNamespace(conv=[torch.zeros(2, 3)], temporal=torch.zeros(2, 3))
        backend.state_pool = SimpleNamespace(mamba_cache=cache)
        graph = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=[512]))
        )
        with (
            patch.object(gdn_backend, "is_cuda", return_value=cuda),
            patch.object(gdn_backend, "FLA_CHUNK_SIZE", 64, create=True),
            patch.object(
                gdn_backend, "buckets_keep_chunk_o_block_size", return_value=True
            ),
            patch.object(
                gdn_backend, "get_exec", return_value=SimpleNamespace(graph=graph)
            ),
            patch(
                "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
                "get_prefill_num_tokens_to_capture",
                return_value=[512],
            ),
        ):
            yield backend

    def test_the_switch_captures_the_extend_only_where_the_setup_supports_it(self):
        """Off, the default, breakable prefill graphs keep breaking at every GDN
        layer; on, they capture the target's extend only where this setup supports
        the graph-safe chain, and a draft's prefill graphs keep their eager path."""
        for switch, setup, captured in (
            (False, {}, False),
            (True, {}, True),
            (True, dict(cuda=False), False),
            (True, dict(enable_mis=True), False),
            (True, dict(draft=True), False),
        ):
            with (
                self.subTest(switch=switch, **setup),
                envs.SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND.override(switch),
                self._setup(**setup) as backend,
            ):
                self.assertEqual(backend.supports_prefill_graph_extend(), captured)


class TestFlashInferGDNPrefillBackendPolicy(CustomTestCase):
    def test_prefill_conv_table_covers_only_nonempty_chunks(self):
        backend = object.__new__(GDNAttnBackend)
        backend.device = "cpu"
        backend.mis_metadata = None
        backend.forward_metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, 65, 66, 66, 67, 68, 69, 70, 71]),
            mamba_cache_indices=torch.arange(8),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            tbo_parent_token_range=None,
            extend_seq_lens_cpu=[65, 1, 0, 1, 1, 1, 1, 1],
        )
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch.object(gdn_backend, "is_pin_memory_available", return_value=False),
        ):
            backend._init_prefill_conv_block_table(batch)
        torch.testing.assert_close(
            backend.forward_metadata.conv_block_table,
            torch.tensor(
                [[0, i] for i in range(9)] + [[i, 0] for i in (1, 3, 4, 5, 6, 7)],
                dtype=torch.int32,
            ),
        )

    def test_prefill_conv_table_falls_back_and_clears_previous_batch(self):
        cases = (
            ("uniform", {"extend_seq_lens_cpu": [7, 8]}, {}),
            ("mildly_ragged", {}, {}),
            ("no_cpu_lengths", {"extend_seq_lens_cpu": None}, {}),
            ("different_batch", {"extend_seq_lens_cpu": [17, 1, 1]}, {}),
            ("tbo", {"tbo_parent_token_range": (0, 18)}, {}),
            ("decode", {"forward_mode": ForwardMode.DECODE}, {}),
            ("verify", {"forward_mode": ForwardMode.TARGET_VERIFY}, {}),
            ("multi_item_scoring", {}, {"mis_metadata": sentinel.metadata}),
            ("non_cuda", {}, {"cuda": False}),
        )
        for name, batch_fields, options in cases:
            with self.subTest(name=name):
                backend = object.__new__(GDNAttnBackend)
                backend.device = "cpu"
                backend.mis_metadata = options.get("mis_metadata")
                backend.forward_metadata = ForwardMetadata(
                    query_start_loc=torch.tensor([0, 17, 18]),
                    mamba_cache_indices=torch.arange(2),
                    conv_block_table=sentinel.previous_batch,
                )
                fields = dict(
                    forward_mode=ForwardMode.EXTEND,
                    tbo_parent_token_range=None,
                    extend_seq_lens_cpu=[17, 1],
                )
                fields.update(batch_fields)
                with patch.object(
                    gdn_backend, "is_cuda", return_value=options.get("cuda", True)
                ):
                    backend._init_prefill_conv_block_table(SimpleNamespace(**fields))
                self.assertIsNone(backend.forward_metadata.conv_block_table)

    @staticmethod
    def make_target_verify_routing_backend():
        backend = object.__new__(GDNAttnBackend)
        nominal_capability = MagicMock(return_value=False)
        backend.kernel_dispatcher = SimpleNamespace(
            verify_kernel_is_flashinfer=True,
            target_verify_supports_strided_qkv=nominal_capability,
        )
        return backend, nominal_capability

    def test_mis_requires_triton_prefill_backend(self):
        runner = make_runner(self, enable_mis=True)
        with self.assertRaisesRegex(ValueError, "Triton linear-attention prefill"):
            validate_gdn_mis_backend(LinearAttnKernelBackend.FLASHINFER)

    def test_mis_rejects_page_major_layout(self):
        make_runner(self, enable_mis=True, enable_page_major_kv_layout=True)
        with self.assertRaisesRegex(ValueError, "page-major"):
            validate_gdn_mis_backend(LinearAttnKernelBackend.TRITON)

    def test_non_gdn_linear_backend_rejects_mis(self):
        with self.assertRaisesRegex(ValueError, "does not support multi-item scoring"):
            MambaAttnBackendBase.validate_mis_support(SimpleNamespace(enable_mis=True))

        GDNAttnBackend.validate_mis_support(SimpleNamespace(enable_mis=True))

    def apply_policy(
        self,
        runner,
        *,
        cuda=True,
        capability=(10, 0),
        cuda_version="13.0",
        flashinfer_available=True,
    ):
        with (
            patch.object(
                gdn_backend,
                "hybrid_gdn_config",
                return_value=runner.hybrid_gdn_config,
            ),
            patch.object(gdn_backend, "is_cuda", return_value=cuda),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch.object(torch.version, "cuda", cuda_version),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "is_flashinfer_gdn_prefill_available",
                return_value=flashinfer_available,
            ),
        ):
            return flashinfer_gdn_prefill_default(runner)

    def test_selects_flashinfer_for_supported_sm100_gdn(self):
        self.assertEqual(self.apply_policy(make_runner(self)), "flashinfer")

    def test_selects_flashinfer_for_radix_cache_strategies(self):
        for strategy in ("no_buffer", "extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                runner = make_runner(
                    self,
                    uses_mamba_radix_cache=True,
                    mamba_radix_cache_strategy=strategy,
                )
                self.assertEqual(self.apply_policy(runner), "flashinfer")

    def test_declines_when_the_prefill_backend_is_explicit(self):
        for backend in ("triton", "flashinfer", "cutedsl"):
            with self.subTest(backend=backend):
                runner = make_runner(self, linear_attn_prefill_backend=backend)
                self.assertIsNone(self.apply_policy(runner))

    def test_declines_when_deterministic_inference_is_enabled(self):
        """Batch-sensitive GDN prefill must not bypass deterministic inference."""
        runner = make_runner(
            self,
            state_dtype=torch.float32,
            enable_deterministic_inference=True,
        )

        self.assertIsNone(
            self.apply_policy(
                runner,
                capability=(9, 0),
                cuda_version="12.9",
            )
        )

    def test_rejects_explicit_flashinfer_prefill_in_deterministic_mode(self):
        """Explicit backend precedence must not bypass deterministic GDN startup."""
        cases = (
            {"linear_attn_prefill_backend": "flashinfer"},
            {"linear_attn_backend": "flashinfer"},
        )
        for fields in cases:
            with self.subTest(fields=fields):
                make_runner(
                    self,
                    enable_deterministic_inference=True,
                    **fields,
                )
                backends = resolve_linear_attn_backends()

                with self.assertRaisesRegex(
                    ValueError,
                    "FlashInfer GDN prefill is not supported with "
                    "--enable-deterministic-inference",
                ):
                    _validate_gdn_linear_attn_backends(backends)

    def test_rejects_unsupported_capability(self):
        cases = (
            ("non_cuda", {}, {"cuda": False}),
            ("hopper", {}, {"capability": (9, 0)}),
            ("future_sm", {}, {"capability": (12, 0)}),
            ("cuda_12", {}, {"cuda_version": "12.9"}),
            ("fp32_state", {"state_dtype": torch.float32}, {}),
            ("key_dim", {"key_dim": 64}, {}),
            ("value_dim", {"value_dim": 64}, {}),
            ("missing_api", {}, {"flashinfer_available": False}),
        )
        for name, runner_args, hardware in cases:
            with self.subTest(name=name):
                self.assertIsNone(
                    self.apply_policy(make_runner(self, **runner_args), **hardware)
                )

    def test_selects_flashinfer_for_a_page_major_pool(self):
        """The unified pool runs page-major. Its strided state reaches every
        prefill kernel as contiguous per-sequence copies, so FlashInfer prefill
        qualifies there exactly as on the contiguous pool."""
        cases = (
            ("sm100", {}, {}),
            (
                "sm90",
                {"state_dtype": torch.float32},
                {"capability": (9, 0), "cuda_version": "12.9"},
            ),
        )
        for name, runner_args, hardware in cases:
            with self.subTest(name=name):
                runner = make_runner(
                    self, enable_page_major_kv_layout=True, **runner_args
                )
                self.assertEqual(self.apply_policy(runner, **hardware), "flashinfer")

    def test_rejects_gdn_config_without_qwen_head_dims(self):
        runner = make_runner(self)
        runner.hybrid_gdn_config = SimpleNamespace()
        self.assertIsNone(self.apply_policy(runner))

    def test_rejects_unvalidated_runtime_modes(self):
        cases = (
            ("non_triton_base", {"linear_attn_backend": "cutedsl"}),
            ("dynamic_chunk", {"enable_dynamic_chunking": True}),
            ("unchunked", {"chunked_prefill_size": -1}),
            ("unknown_chunk", {"chunked_prefill_size": None}),
            ("large_chunk", {"chunked_prefill_size": 8193}),
        )
        for name, runner_args in cases:
            with self.subTest(name=name):
                self.assertIsNone(self.apply_policy(make_runner(self, **runner_args)))

    def test_builds_compact_checkpoint_plan_for_packed_sequences(self):
        forward_batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([63, 64, 65, 127, 128, 129]),
            mamba_track_mask=torch.tensor([False, True, True, True, True, True]),
            # 65 on the 128-token sequence represents an interior S64
            # boundary encoded as S64 + 1 by the scheduler.
            mamba_track_seqlens=torch.tensor([63, 64, 65, 127, 65, 129]),
            extend_prefix_lens=torch.zeros(6, dtype=torch.int64),
        )
        metadata = SimpleNamespace(
            track_ssm_h_src=torch.empty(4),
            track_ssm_h_dst=torch.empty(4),
        )

        # The chunk size is a derived config member; seed its private cache on a
        # published config rather than patching an import binding.
        override = get_context().override_server_args(_mamba_cache_chunk_size=64)
        override.install()
        self.addCleanup(override.restore)
        maybe_build_flashinfer_checkpoint_plan(forward_batch, metadata, "cpu")

        torch.testing.assert_close(
            metadata.state_checkpoint_cu_starts,
            torch.tensor([0, 0, 1, 2, 3, 5, 7]),
        )
        torch.testing.assert_close(metadata.track_ssm_h_src, torch.tensor([1, 2, 3, 6]))
        self.assertEqual(metadata.num_state_checkpoints, 7)
        self.assertEqual(metadata.state_checkpoint_every_n_tokens, 64)

    def test_decode_tracking_without_h_source_skips_checkpoint_plan(self):
        backend = object.__new__(GDNAttnBackend)
        backend.device = "cpu"
        backend.kernel_dispatcher = SimpleNamespace(extend_uses_state_checkpoints=True)
        metadata = SimpleNamespace(has_mamba_track_mask=True, track_ssm_h_src=None)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            prefill_cp_metadata=None,
            multi_item_delimiter_indices=None,
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([7]),
        )

        def init_base(instance, _forward_batch):
            instance.forward_metadata = metadata

        with patch.object(MambaAttnBackendBase, "init_forward_metadata", init_base):
            backend.init_forward_metadata(forward_batch)

        torch.testing.assert_close(metadata.conv_states_mask_indices, torch.tensor([7]))

    def test_tree_verify_uses_triton_kernel(self):
        flashinfer_kernel = MagicMock(
            supports_target_verify=True,
            supports_strided_target_verify_qkv=False,
        )
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "FlashInferGDNKernel",
                return_value=flashinfer_kernel,
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.FLASHINFER,
            )

        self.assertIsInstance(dispatcher.tree_verify_kernel, TritonGDNKernel)
        self.assertFalse(dispatcher.target_verify_supports_strided_qkv(None))
        self.assertTrue(
            dispatcher.target_verify_supports_strided_qkv(sentinel.parent_token)
        )

        tensor = sentinel.tensor
        with patch.object(
            dispatcher.tree_verify_kernel, "target_verify"
        ) as tree_verify:
            dispatcher.target_verify(
                *([tensor] * 7),
                ssm_states=tensor,
                cache_indices=tensor,
                query_start_loc=tensor,
                retrieve_parent_token=sentinel.parent_token,
            )

        tree_verify.assert_called_once()
        flashinfer_kernel.target_verify.assert_not_called()

    def test_target_verify_strided_input_capability_is_opt_in(self):
        dispatcher = GDNKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
        )

        self.assertTrue(dispatcher.target_verify_supports_strided_qkv(None))
        dispatcher.verify_kernel = SimpleNamespace()
        self.assertFalse(dispatcher.target_verify_supports_strided_qkv(None))

    def test_target_verify_strided_qkv_routing(self):
        backend, nominal_capability = self.make_target_verify_routing_backend()
        cases = (
            ("fp32_fold", True, False, torch.float32, 4, True, False),
            ("short_bf16_fold", True, False, torch.bfloat16, 2, True, False),
            ("cutedsl_fold", True, False, torch.bfloat16, 4, False, False),
            ("circular", False, True, torch.bfloat16, 4, True, False),
            ("nominal", False, False, torch.bfloat16, 4, False, True),
        )
        for name, fold, circular, dtype, draft_tokens, expected, delegates in cases:
            with self.subTest(name=name):
                nominal_capability.reset_mock()
                actual = backend._target_verify_supports_strided_qkv(
                    retrieve_parent_token=None,
                    use_replayssm_fold=fold,
                    use_replayssm_spec=circular,
                    ssm_dtype=dtype,
                    draft_token_num=draft_tokens,
                )
                self.assertEqual(actual, expected)
                if delegates:
                    nominal_capability.assert_called_once_with(None)
                else:
                    nominal_capability.assert_not_called()

    def test_helion_backend_reports_kda_only(self):
        cases = (
            (LinearAttnKernelBackend.HELION, LinearAttnKernelBackend.TRITON),
            (LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.HELION),
        )
        for decode_backend, prefill_backend in cases:
            with self.subTest(
                decode_backend=decode_backend,
                prefill_backend=prefill_backend,
            ):
                with self.assertRaisesRegex(ValueError, "supports KDA only"):
                    GDNKernelDispatcher(decode_backend, prefill_backend)


class TestFlashQLAGDNPrefillPolicy(CustomTestCase):
    def test_flashqla_uses_the_cp_group_without_borrowing_tp(self):
        from sglang.srt.runtime_context import get_parallel

        for world in (1, 2, 4):
            with self.subTest(cp_size=world):
                cp_group = SimpleNamespace(world_size=world, device_group=object())
                with (
                    get_context().override_server_args(
                        tp_size=8,
                        attn_cp_size=world,
                        linear_attn_prefill_backend="flashqla",
                        disaggregation_mode="prefill",
                    ),
                    get_parallel().override(attn_cp_group=cp_group),
                    patch(
                        "sglang.srt.layers.attention.linear.kernels.gdn_flashqla."
                        "FlashQLAGDNKernel",
                        side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
                    ),
                ):
                    dispatcher = GDNKernelDispatcher(
                        LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.FLASHQLA
                    )
                    self.assertEqual(dispatcher.use_context_parallel, world > 1)
                    self.assertIs(
                        dispatcher.extend_kernel.cp_group,
                        cp_group.device_group if world > 1 else None,
                    )
                    self.assertIsInstance(dispatcher.decode_kernel, TritonGDNKernel)
                    self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)

    def test_gdn_plan_matches_the_selected_attention_shards(self):
        """A zigzag attention shard must not be routed as round-robin GDN input."""
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
        from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
        from sglang.srt.layers.cp.utils import (
            cp_gather_full_sequence_states,
            cp_shard_position_ids,
        )
        from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.runtime_context import get_parallel

        for strategy, strategy_type in (
            ("interleave", InterleaveCPStrategy),
            ("zigzag", ZigzagCPStrategy),
        ):
            for world in (2, 4):
                for lengths in ((512,), (513, 259), (8, 3)):
                    for rank in range(world):
                        with self.subTest(
                            strategy=strategy, world=world, lengths=lengths, rank=rank
                        ):
                            prefixes = tuple(5 + 16 * i for i in range(len(lengths)))
                            group = SimpleNamespace(
                                world_size=world,
                                rank_in_group=rank,
                                device_group=object(),
                            )
                            backend = object.__new__(HybridLinearAttnBackend)
                            backend.full_attn_backend = SimpleNamespace(
                                cp_exchange=None
                            )
                            backend.linear_attn_backend = SimpleNamespace(
                                device="cpu", _cp_residual_exchange=None
                            )
                            backend._cp_metadata_plan = None
                            batch = SimpleNamespace(
                                forward_mode=ForwardMode.EXTEND,
                                prefill_cp_metadata=None,
                                extend_lens=lengths,
                                prefix_lens=prefixes,
                            )
                            with (
                                get_context().override_server_args(
                                    device="cpu",
                                    tp_size=world,
                                    attn_cp_size=world,
                                    enable_prefill_cp=True,
                                    cp_strategy=strategy,
                                    linear_attn_prefill_backend="flashqla",
                                    disaggregation_mode="prefill",
                                    enable_cp_tp_group_sharing=True,
                                ),
                                get_parallel().override(
                                    attn_cp_group=group, attn_cp_rank=rank
                                ),
                                patch(
                                    "torch.distributed.get_world_size",
                                    return_value=world,
                                ),
                                patch("torch.distributed.get_rank", return_value=rank),
                            ):
                                backend.prepare_cp_metadata(batch)
                                attention = strategy_type(cp_size=world)
                                batch.attn_cp_metadata = attention.build_metadata(
                                    sum(lengths),
                                    [p + n for p, n in zip(prefixes, lengths)],
                                    list(lengths),
                                )
                                pad_logical_token_to_physical(batch.attn_cp_metadata)
                                local = attention.shard_hidden_states(
                                    torch.arange(sum(lengths)), batch
                                )
                                indices = batch.prefill_cp_metadata.attention[rank]
                                torch.testing.assert_close(
                                    local[: len(indices)], torch.tensor(indices)
                                )
                                # The common plan must not activate whole-layer
                                # residual movement before the model splits rows.
                                self.assertIsNone(
                                    ForwardBatch.layer_cp_metadata.fget(batch)
                                )
                                positions = torch.arange(sum(lengths)).repeat(3, 1)
                                torch.testing.assert_close(
                                    cp_shard_position_ids(positions, batch),
                                    attention.shard_position_ids(positions, batch),
                                )
                                frame = batch.prefill_cp_metadata
                                frame.current = frame.layout_names["attention"]
                                self.assertIs(
                                    ForwardBatch.layer_cp_metadata.fget(batch), frame
                                )
                                plan = frame.token_collectives()
                                frame.original_non_padded = torch.tensor(sum(lengths))

                                def gather_rows(output, source, *, group):
                                    torch.testing.assert_close(
                                        source[: len(indices)], local[: len(indices)]
                                    )
                                    for peer, rows in enumerate(frame.attention):
                                        start = peer * plan.padded_rows
                                        output[start : start + len(rows)] = (
                                            torch.tensor(rows)
                                        )

                                with patch(
                                    "torch.distributed.all_gather_into_tensor",
                                    gather_rows,
                                ):
                                    restored = cp_gather_full_sequence_states(
                                        local, batch
                                    )
                                torch.testing.assert_close(
                                    restored, torch.arange(sum(lengths))
                                )
                                self.assertIs(
                                    batch.num_token_non_padded,
                                    frame.original_non_padded,
                                )

    def test_shared_batch_plan_binds_both_backends_without_late_planning(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
        from sglang.srt.layers.cp.utils import ShardedContextParallelMetadata
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.runtime_context import get_parallel

        table = torch.arange(1024, dtype=torch.int32).reshape(2, 512) * 2
        pool = SimpleNamespace(req_to_token=table)
        group = SimpleNamespace(world_size=2, rank_in_group=0, device_group=object())
        for cp_mode in ("round-robin-split", "in-seq-split"):
            with (
                self.subTest(cp_mode=cp_mode),
                get_context().override_server_args(
                    device="cpu",
                    tp_size=2,
                    attn_cp_size=2,
                    linear_attn_prefill_backend="flashqla",
                    disaggregation_mode="prefill",
                ),
                get_parallel().override(attn_cp_group=group, attn_cp_rank=0),
                patch("torch.distributed.get_world_size", return_value=2),
                patch("torch.distributed.get_rank", return_value=0),
            ):
                full = SimpleNamespace(
                    cp_exchange=SimpleNamespace(capacity=512),
                    prefill_cp_mode=cp_mode,
                    cp_packed_kv=False,
                    kv_cache_dtype=torch.bfloat16,
                    residual_exchange=None,
                    forward_metadata=None,
                )
                linear = object.__new__(GDNAttnBackend)
                linear.device = "cpu"
                linear._cp_residual_exchange = None
                linear.kernel_dispatcher = SimpleNamespace(use_context_parallel=True)
                linear.req_to_token_pool = SimpleNamespace(
                    mamba_pool=SimpleNamespace(synchronize=lambda: None),
                    get_mamba_indices=lambda indices: indices + 5,
                )
                hybrid = object.__new__(HybridLinearAttnBackend)
                hybrid.full_attn_backend = full
                hybrid.linear_attn_backend = linear
                hybrid.req_to_token_pool = pool
                hybrid._cp_metadata_plan = None
                runner = SimpleNamespace(server_args=None, attn_backend=hybrid)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    prefill_cp_metadata=None,
                    extend_lens=[128, 64],
                    prefix_lens=[7, 3],
                    reqs=[
                        SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i))
                        for i in (1, 0)
                    ],
                    out_cache_loc=torch.arange(192) * 2,
                    num_token_non_padded=torch.tensor(192, dtype=torch.int32),
                    batch_size=2,
                    req_pool_indices=torch.tensor([1, 0]),
                    mamba_track_checkpoints=((), ()),
                )
                ModelRunner.prepare_cp_metadata(runner, batch)
                frame = batch.prefill_cp_metadata
                self.assertIsNone(full.forward_metadata)
                self.assertIs(ForwardBatch.layer_cp_metadata.fget(batch), frame)
                torch.testing.assert_close(
                    frame.kv_read_slots,
                    torch.cat((torch.arange(0, 135, 2) + 512, torch.arange(0, 67, 2))),
                )
                torch.testing.assert_close(
                    frame.kv_write_slots,
                    torch.tensor(frame.layout.new_kv_indices[0]),
                )
                with (
                    patch.object(
                        ShardedContextParallelMetadata,
                        "plan_prefill",
                        side_effect=AssertionError("late plan"),
                    ),
                    patch("torch.cuda.current_stream", return_value=None),
                ):
                    ModelRunner.prepare_cp_metadata(runner, batch)
                    AttentionBackend._init_cp_forward_metadata(full, batch)
                    linear.init_forward_metadata(batch)
                self.assertIs(full.forward_metadata, frame)
                self.assertIs(linear.cp_metadata, frame)
                torch.testing.assert_close(linear.cp_state_slots, torch.tensor([6, 5]))
                # A later chunk owns new mutable state while reusing only maps.
                frame.current = frame.layout_names["contiguous"]
                batch.prefill_cp_metadata = None
                batch.prefix_lens = [135, 67]
                ModelRunner.prepare_cp_metadata(runner, batch)
                next_frame = batch.prefill_cp_metadata
                self.assertIsNot(next_frame, frame)
                self.assertIsNone(next_frame.current)
                self.assertEqual(next_frame.prefixes, (135, 67))
                self.assertIs(full.forward_metadata, frame)
                self.assertIs(linear.cp_metadata, frame)
                self.assertIs(next_frame.query_indices, frame.query_indices)

    def test_cp_preparation_skips_decode_verify_and_idle(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        with get_context().override_server_args(
            attn_cp_size=2,
            linear_attn_prefill_backend="flashqla",
            disaggregation_mode="prefill",
        ):
            for mode in (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.IDLE,
            ):
                with self.subTest(mode=mode):
                    batch = SimpleNamespace(forward_mode=mode, prefill_cp_metadata=None)
                    ModelRunner.prepare_cp_metadata(
                        SimpleNamespace(server_args=None), batch
                    )
                    self.assertIsNone(batch.prefill_cp_metadata)


if __name__ == "__main__":
    unittest.main()
