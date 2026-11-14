import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.mamba import FalconMambaConfig, MambaConfig
from sglang.srt.configs.mamba2 import Mamba2Config
from sglang.srt.layers.attention.hybrid_linear_attn_backend import Mamba2AttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    Mamba2Metadata,
    Mamba2MISMetadata,
    build_mamba2_mis_metadata,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMamba2MISMetadata(CustomTestCase):
    def test_single_request_segments_delimiters_and_chunk_alignment(self):
        mis_metadata = _build_metadata_for_single_req()
        self.assertIsInstance(mis_metadata, Mamba2MISMetadata)
        self.assertEqual(mis_metadata.chunk_size, 4)

        # Delimiters at 3, 10, 16 must be excluded from both query and item tokens.
        self.assertEqual(mis_metadata.reset_cache_indices.tolist(), [7])
        self.assertEqual(mis_metadata.query_token_indices.tolist(), [0, 1, 2])
        self.assertEqual(mis_metadata.query_cu_seqlens.tolist(), [0, 3])
        self.assertEqual(mis_metadata.query_seq_lens_cpu, [3])
        self.assertEqual(mis_metadata.query_seq_idx.reshape(-1).tolist(), [0, 0, 0])
        self.assertEqual(mis_metadata.query_cache_indices.tolist(), [7])
        self.assertEqual(mis_metadata.query_cache_indices_i64.tolist(), [7])
        self.assertEqual(mis_metadata.query_has_initial_states.tolist(), [False])

        expected_item_tokens = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
        self.assertEqual(mis_metadata.item_token_indices.tolist(), expected_item_tokens)
        self.assertEqual(mis_metadata.item_cu_seqlens.tolist(), [0, 6, 11])
        self.assertEqual(mis_metadata.item_seq_lens_cpu, [6, 5])
        self.assertEqual(
            mis_metadata.item_seq_idx.reshape(-1).tolist(),
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )
        self.assertEqual(mis_metadata.item_conv_indices.tolist(), [0, 1])
        self.assertEqual(mis_metadata.item_ssm_indices_i64.tolist(), [7, 7])
        self.assertEqual(mis_metadata.item_has_initial_states.tolist(), [True, True])

        # With chunk_size=4, Item 0 covers [0, 6) -> chunks (0, 0) and (1, 0),
        # and Item 1 covers [6, 11) -> chunks (1, 2) and (2, 0).
        self.assertEqual(mis_metadata.item_chunk_indices.tolist(), [0, 1, 1, 2])
        self.assertEqual(mis_metadata.item_chunk_offsets.tolist(), [0, 0, 2, 0])

        for delimiter_idx in (3, 10, 16):
            self.assertNotIn(delimiter_idx, mis_metadata.query_token_indices.tolist())
            self.assertNotIn(delimiter_idx, mis_metadata.item_token_indices.tolist())

    def test_mixed_batch_empty_query_and_dp_padding(self):
        # Req 0: seq_len=8, delimiters=[0, 2, 7] -> empty query, item 0 [1], item 1 [3..6]
        # Req 1: seq_len=6, delimiters=[2, 5] -> query [8, 9], item 2 [11, 12]
        # num_input_tokens=20 leaves trailing DP padding [14..19] that must be ignored.
        mis_metadata = build_mamba2_mis_metadata(
            is_prefill_only=True,
            extend_prefix_lens_cpu=[0, 0],
            extend_seq_lens_cpu=[8, 6],
            multi_item_delimiter_indices=[
                torch.tensor([0, 2, 7], dtype=torch.int64),
                torch.tensor([2, 5], dtype=torch.int64),
            ],
            num_input_tokens=20,
            cache_indices=torch.tensor([4, 9], dtype=torch.int32),
            chunk_size=4,
            device=torch.device("cpu"),
        )

        # Every request slot is cleared up front before segment execution.
        self.assertEqual(mis_metadata.reset_cache_indices.tolist(), [4, 9])

        self.assertEqual(mis_metadata.query_token_indices.tolist(), [8, 9])
        self.assertEqual(mis_metadata.query_cu_seqlens.tolist(), [0, 2])
        self.assertEqual(mis_metadata.query_seq_lens_cpu, [2])
        self.assertEqual(mis_metadata.query_cache_indices.tolist(), [9])
        self.assertEqual(mis_metadata.query_has_initial_states.tolist(), [False])

        self.assertEqual(
            mis_metadata.item_token_indices.tolist(), [1, 3, 4, 5, 6, 11, 12]
        )
        self.assertEqual(mis_metadata.item_cu_seqlens.tolist(), [0, 1, 5, 7])
        self.assertEqual(mis_metadata.item_seq_lens_cpu, [1, 4, 2])
        self.assertEqual(mis_metadata.item_conv_indices.tolist(), [0, 1, 2])
        self.assertEqual(mis_metadata.item_ssm_indices_i64.tolist(), [4, 4, 9])
        self.assertEqual(
            mis_metadata.item_has_initial_states.tolist(), [True, True, True]
        )
        self.assertEqual(mis_metadata.item_chunk_indices.tolist(), [0, 0, 1, 1])
        self.assertEqual(mis_metadata.item_chunk_offsets.tolist(), [0, 1, 0, 1])

        all_used = set(mis_metadata.query_token_indices.tolist()) | set(
            mis_metadata.item_token_indices.tolist()
        )
        for padded_pos in range(14, 20):
            self.assertNotIn(padded_pos, all_used)

    def test_empty_query_single_token_item_batch(self):
        mis_metadata = build_mamba2_mis_metadata(
            is_prefill_only=True,
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[3],
            multi_item_delimiter_indices=[torch.tensor([0, 2], dtype=torch.int64)],
            num_input_tokens=3,
            cache_indices=torch.tensor([3], dtype=torch.int32),
            chunk_size=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(mis_metadata.reset_cache_indices.tolist(), [3])
        self.assertEqual(mis_metadata.query_token_indices.numel(), 0)
        self.assertEqual(mis_metadata.query_cu_seqlens.tolist(), [0])
        self.assertEqual(mis_metadata.item_token_indices.tolist(), [1])
        self.assertEqual(mis_metadata.item_cu_seqlens.tolist(), [0, 1])
        self.assertEqual(mis_metadata.item_seq_lens_cpu, [1])
        self.assertEqual(mis_metadata.item_conv_indices.tolist(), [0])
        self.assertEqual(mis_metadata.item_ssm_indices_i64.tolist(), [3])
        self.assertEqual(mis_metadata.item_has_initial_states.tolist(), [True])
        self.assertEqual(mis_metadata.item_chunk_indices.tolist(), [0])
        self.assertEqual(mis_metadata.item_chunk_offsets.tolist(), [0])

    def test_cache_indices_dtypes_and_caller_mutation_isolation(self):
        for cache_dtype in (torch.int32, torch.int64):
            with self.subTest(cache_dtype=cache_dtype):
                cache_indices = torch.tensor([11], dtype=cache_dtype)
                delims = [torch.tensor([2, 5], dtype=torch.int64)]
                mis_metadata = build_mamba2_mis_metadata(
                    is_prefill_only=True,
                    extend_prefix_lens_cpu=[0],
                    extend_seq_lens_cpu=[6],
                    multi_item_delimiter_indices=delims,
                    num_input_tokens=6,
                    cache_indices=cache_indices,
                    chunk_size=4,
                    device=torch.device("cpu"),
                )
                self.assertEqual(mis_metadata.query_cache_indices.dtype, torch.int32)
                self.assertEqual(mis_metadata.item_conv_indices.dtype, torch.int32)
                self.assertEqual(
                    mis_metadata.query_cache_indices_i64.dtype, torch.int64
                )
                self.assertEqual(mis_metadata.item_ssm_indices_i64.dtype, torch.int64)
                self.assertEqual(mis_metadata.item_conv_indices.tolist(), [0])

                # Caller mutations after metadata build must not alias stored metadata tensors.
                cache_indices.fill_(999)
                delims[0].fill_(0)
                self.assertEqual(mis_metadata.reset_cache_indices.tolist(), [11])
                self.assertEqual(mis_metadata.query_cache_indices.tolist(), [11])
                self.assertEqual(mis_metadata.query_cache_indices_i64.tolist(), [11])
                self.assertEqual(mis_metadata.item_ssm_indices_i64.tolist(), [11])
                self.assertEqual(mis_metadata.item_token_indices.tolist(), [3, 4])

    def test_batch_validation_errors(self):
        invalid_cases = [
            ("non_prefill_only", {"is_prefill_only": False}, "prefill-only requests"),
            ("non_zero_prefix", {"extend_prefix_lens_cpu": [1]}, "cached prefixes"),
            (
                "missing_delims",
                {"multi_item_delimiter_indices": None},
                "delimiter indices for every request",
            ),
            (
                "wrong_delims_len",
                {"multi_item_delimiter_indices": []},
                "delimiter indices for every request",
            ),
            (
                "tokens_exceed_input",
                {"num_input_tokens": 5},
                "exceed the input tokens",
            ),
            ("non_positive_chunk_size", {"chunk_size": 0}, "positive chunk_size"),
            ("negative_chunk_size", {"chunk_size": -4}, "positive chunk_size"),
            (
                "2d_cache_indices",
                {"cache_indices": torch.tensor([[1]], dtype=torch.int32)},
                "1D integer cache_indices",
            ),
            (
                "wrong_len_cache_indices",
                {"cache_indices": torch.tensor([1, 2], dtype=torch.int32)},
                "1D integer cache_indices",
            ),
            (
                "float_cache_indices",
                {"cache_indices": torch.tensor([1.0], dtype=torch.float32)},
                "1D integer cache_indices",
            ),
            (
                "meta_cache_indices",
                {"cache_indices": torch.empty(1, dtype=torch.int32, device="meta")},
                "1D integer cache_indices",
            ),
        ]
        _assert_invalid_mis_cases(testcase=self, invalid_cases=invalid_cases)

    def test_delimiter_validation_errors(self):
        invalid_cases = [
            (
                "too_few_delims",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([5], dtype=torch.int64)
                    ]
                },
                "at least two delimiters",
            ),
            (
                "duplicate_delims",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 2, 5], dtype=torch.int64)
                    ]
                },
                "strictly increasing",
            ),
            (
                "decreasing_delims",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([3, 2, 5], dtype=torch.int64)
                    ]
                },
                "strictly increasing",
            ),
            (
                "negative_delim",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([-1, 5], dtype=torch.int64)
                    ]
                },
                "outside the request",
            ),
            (
                "overflow_delim",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 6], dtype=torch.int64)
                    ]
                },
                "outside the request",
            ),
            (
                "final_delim_not_last",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 4], dtype=torch.int64)
                    ]
                },
                "final delimiter must be the last request token",
            ),
            (
                "empty_single_item",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 3, 5], dtype=torch.int64)
                    ]
                },
                "does not support empty items",
            ),
            (
                "trailing_empty_item",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 4, 5], dtype=torch.int64)
                    ]
                },
                "does not support empty items",
            ),
            (
                "empty_item_after_empty_query",
                {
                    "extend_seq_lens_cpu": [2],
                    "num_input_tokens": 2,
                    "multi_item_delimiter_indices": [
                        torch.tensor([0, 1], dtype=torch.int64)
                    ],
                },
                "does not support empty items",
            ),
        ]
        _assert_invalid_mis_cases(testcase=self, invalid_cases=invalid_cases)

    def test_backend_supports_mis_and_mamba1_guards(self):
        self.assertIs(Mamba2AttnBackend.supports_mis, True)

        with get_context().override_server_args(
            model_path="dummy", device="cpu", enable_mis=False
        ):
            runner_no_mis = _ForbiddenModelConfigRunner(
                server_args=get_server_args(),
                use_forbidden_model_config=True,
            )
            backend = Mamba2AttnBackend(runner_no_mis)
            self.assertFalse(backend.enable_mis)

        with get_context().override_server_args(
            model_path="dummy", device="cpu", enable_mis=True
        ):
            runner_mamba2 = _ForbiddenModelConfigRunner(
                server_args=get_server_args(),
                use_forbidden_model_config=False,
                hf_config=Mamba2Config(),
            )
            backend_mamba2 = Mamba2AttnBackend(runner_mamba2)
            self.assertTrue(backend_mamba2.enable_mis)

            for mamba1_cfg in (MambaConfig(), FalconMambaConfig()):
                with self.subTest(config_cls=type(mamba1_cfg).__name__):
                    runner_mamba1 = _ForbiddenModelConfigRunner(
                        server_args=get_server_args(),
                        use_forbidden_model_config=False,
                        hf_config=mamba1_cfg,
                    )
                    with self.assertRaisesRegex(ValueError, "Mamba-1"):
                        Mamba2AttnBackend(runner_mamba1)

    def test_backend_init_forward_metadata_mis_fast_path(self):
        delimiters = [torch.tensor([2, 5], dtype=torch.int64)]
        prepared = Mamba2Metadata(
            query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([7], dtype=torch.int32),
            num_prefills=1,
            num_prefill_tokens=6,
            num_decodes=0,
        )

        for enable_mis in (False, True):
            with self.subTest(case="non_mis_batch", enable_mis=enable_mis):
                backend = _make_stub_backend(enable_mis=enable_mis)
                forward_batch = _make_forward_batch(delimiters=None)
                with (
                    patch.object(
                        backend, "_forward_metadata", return_value=SimpleNamespace()
                    ) as mock_fwd,
                    patch.object(
                        Mamba2Metadata, "prepare_mixed", return_value=prepared
                    ) as mock_mixed,
                ):
                    backend.init_forward_metadata(forward_batch)
                mock_fwd.assert_called_once()
                mock_mixed.assert_called_once()
                self.assertIsNone(backend.forward_metadata.mis_metadata)

        with self.subTest(case="delimiters_without_enable_mis"):
            backend = _make_stub_backend(enable_mis=False)
            forward_batch = _make_forward_batch(delimiters=delimiters)
            with (
                patch.object(backend, "_forward_metadata") as mock_fwd,
                patch.object(Mamba2Metadata, "prepare_mixed") as mock_mixed,
                self.assertRaisesRegex(ValueError, "--enable-mis"),
            ):
                backend.init_forward_metadata(forward_batch)
            mock_fwd.assert_not_called()
            mock_mixed.assert_not_called()

        with self.subTest(case="mis_fast_path_skips_prepare_mixed"):
            backend = _make_stub_backend(enable_mis=True, ring_ids=[11])
            forward_batch = _make_forward_batch(delimiters=delimiters)
            with (
                patch.object(backend, "_forward_metadata") as mock_fwd,
                patch.object(Mamba2Metadata, "prepare_mixed") as mock_mixed,
            ):
                backend.init_forward_metadata(forward_batch)
            mock_fwd.assert_not_called()
            mock_mixed.assert_not_called()
            md = backend.forward_metadata
            self.assertIsNone(md.mixed_metadata)
            self.assertIsInstance(md.mis_metadata, Mamba2MISMetadata)
            self.assertEqual(md.mis_metadata.chunk_size, 4)
            self.assertEqual(md.mis_metadata.query_token_indices.tolist(), [0, 1])
            self.assertEqual(md.mis_metadata.item_token_indices.tolist(), [3, 4])
            self.assertEqual(md.mamba_cache_indices.tolist(), [7])
            self.assertEqual(md.replayssm_ring_indices.tolist(), [11])
            self.assertEqual(md.num_prefills, 1)
            self.assertEqual(md.num_prefill_tokens, 6)
            self.assertEqual(md.num_decodes, 0)

    def test_backend_mis_rejects_radix_tracking_and_non_prefill(self):
        delimiters = [torch.tensor([2, 5], dtype=torch.int64)]
        cases = [
            (
                "cpu_track_mask",
                {"mamba_prefill_track_mask_cpu": [True]},
                "disable-radix-cache",
            ),
            (
                "gpu_track_mask_without_cpu",
                {
                    "mamba_prefill_track_mask_cpu": None,
                    "mamba_track_mask": torch.ones(1, dtype=torch.bool),
                },
                "disable-radix-cache",
            ),
            (
                "track_indices_present",
                {"mamba_track_indices": torch.tensor([1], dtype=torch.int32)},
                "disable-radix-cache",
            ),
            ("non_prefill_only", {"is_prefill_only": False}, "prefill-only"),
            (
                "target_verify",
                {"forward_mode": ForwardMode.TARGET_VERIFY},
                "prefill-only",
            ),
            ("decode_mode", {"forward_mode": ForwardMode.DECODE}, "prefill-only"),
            (
                "missing_extend_seq_lens",
                {"extend_seq_lens_cpu": None},
                "extend_seq_lens_cpu",
            ),
            (
                "missing_extend_prefix_lens",
                {"extend_prefix_lens_cpu": None},
                "extend_prefix_lens_cpu",
            ),
        ]
        for name, overrides, expected_msg in cases:
            with self.subTest(name=name):
                backend = _make_stub_backend(enable_mis=True)
                forward_batch = _make_forward_batch(delimiters=delimiters, **overrides)
                with self.assertRaisesRegex(ValueError, expected_msg):
                    backend.init_forward_metadata(forward_batch)

    def test_mamba1_mixer_asserts_no_mis_metadata(self):
        import sglang.srt.layers.attention.mamba.mamba as mamba2_mod

        missing = {
            name: None
            for name in (
                "causal_conv1d_fn",
                "causal_conv1d_fn_triton",
                "causal_conv1d_update",
                "causal_conv1d_update_triton",
            )
            if name not in vars(mamba2_mod)
        }
        mis_metadata = _build_metadata_for_single_req()
        metadata = SimpleNamespace(
            is_target_verify=False,
            has_mamba_track_mask=False,
            mis_metadata=mis_metadata,
        )
        patch_ctx = (
            patch.multiple(mamba2_mod, create=True, **missing)
            if missing
            else nullcontext()
        )
        with patch_ctx:
            from sglang.srt.layers.attention.mamba.mamba1 import MambaMixer1

            mixer = MambaMixer1.__new__(MambaMixer1)
            with self.assertRaisesRegex(AssertionError, "Mamba-1"):
                mixer.forward(
                    hidden_states=torch.zeros(1, 4),
                    output=torch.zeros(1, 4),
                    layer_cache=SimpleNamespace(),
                    metadata=metadata,
                )


def _assert_invalid_mis_cases(*, testcase: CustomTestCase, invalid_cases: list):
    base_kwargs = {
        "is_prefill_only": True,
        "extend_prefix_lens_cpu": [0],
        "extend_seq_lens_cpu": [6],
        "multi_item_delimiter_indices": [torch.tensor([2, 5], dtype=torch.int64)],
        "num_input_tokens": 6,
        "cache_indices": torch.tensor([1], dtype=torch.int32),
        "chunk_size": 4,
        "device": torch.device("cpu"),
    }
    for name, overrides, expected_msg in invalid_cases:
        with testcase.subTest(name=name):
            kwargs = dict(base_kwargs, **overrides)
            with testcase.assertRaisesRegex(ValueError, expected_msg):
                build_mamba2_mis_metadata(**kwargs)


def _make_stub_backend(
    *, enable_mis: bool, ring_ids: list[int] | None = None
) -> Mamba2AttnBackend:
    backend = object.__new__(Mamba2AttnBackend)
    backend.enable_mis = enable_mis
    backend._mamba_chunk_size = 4
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda req_pool_indices: torch.tensor([7], dtype=torch.int32)
    )
    backend._state_ids = SimpleNamespace(ids=lambda idx: idx)
    backend._replayssm_request_ring = lambda: ring_ids is not None
    backend._ring_ids = SimpleNamespace(
        ids=lambda idx: torch.tensor(ring_ids or [], dtype=torch.int32)
    )
    return backend


def _make_forward_batch(*, delimiters, **overrides) -> SimpleNamespace:
    defaults = {
        "batch_size": 1,
        "forward_mode": ForwardMode.EXTEND,
        "is_prefill_only": True,
        "extend_prefix_lens_cpu": [0],
        "extend_seq_lens_cpu": [6],
        "multi_item_delimiter_indices": delimiters,
        "input_ids": torch.zeros(6, dtype=torch.int64),
        "req_pool_indices": torch.tensor([0], dtype=torch.int32),
        "mamba_prefill_track_mask_cpu": [False],
        "mamba_track_mask": None,
        "mamba_track_indices": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build_metadata_for_single_req():
    return build_mamba2_mis_metadata(
        is_prefill_only=True,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[17],
        multi_item_delimiter_indices=[torch.tensor([3, 10, 16], dtype=torch.int64)],
        num_input_tokens=17,
        cache_indices=torch.tensor([7], dtype=torch.int32),
        chunk_size=4,
        device=torch.device("cpu"),
    )


class _ForbiddenModelConfigRunner:
    def __init__(
        self, *, server_args, use_forbidden_model_config: bool, hf_config=None
    ):
        self.server_args = server_args
        self.device = "cpu"
        self.is_draft_worker = False
        self.req_to_token_pool = SimpleNamespace(
            state_sub_pool=None,
            holds_state_slots=False,
        )
        self.token_to_kv_pool = SimpleNamespace(
            layer_router=SimpleNamespace(
                part_view=lambda _: SimpleNamespace(
                    mamba_cache=SimpleNamespace(conv=[torch.zeros(2, 8, 3)])
                ),
                part_indices=lambda _: [0],
            )
        )
        self.runner_ids = SimpleNamespace(
            state_ids=lambda _: None,
            request_ids=lambda: None,
        )
        self._use_forbidden_model_config = use_forbidden_model_config
        self._hf_config = hf_config

    @property
    def model_config(self):
        if self._use_forbidden_model_config:
            raise AssertionError(
                "model_config must not be accessed when enable_mis is False"
            )
        return SimpleNamespace(hf_config=self._hf_config)


if __name__ == "__main__":
    unittest.main()
