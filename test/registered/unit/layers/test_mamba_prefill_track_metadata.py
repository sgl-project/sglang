import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    Mamba2AttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative import spec_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class NoHostRead(torch.Tensor):
    def cpu(self, *args, **kwargs):
        raise AssertionError("CPU tracking must not copy device metadata to the host")


def make_batch(lengths, prefix, track_lens, mask, mirrored):
    def tensor(values):
        return torch.tensor(values).as_subclass(
            NoHostRead if mirrored else torch.Tensor
        )

    return SimpleNamespace(
        batch_size=len(lengths),
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=tensor(lengths),
        extend_prefix_lens=tensor(prefix),
        mamba_track_seqlens=tensor(track_lens),
        mamba_track_mask=tensor(mask),
        mamba_track_indices=tensor([37 + i * 17 for i in range(len(lengths))]),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens_cpu=prefix,
        mamba_track_seqlens_cpu=track_lens if mirrored else None,
        mamba_prefill_track_mask_cpu=mask if mirrored else None,
    )


def make_forward_batch(lengths, starts, cpu_lengths, mode=ForwardMode.EXTEND):
    return ForwardBatch(
        forward_mode=mode,
        batch_size=len(lengths),
        input_ids=torch.zeros(sum(lengths), dtype=torch.int64),
        req_pool_indices=torch.arange(len(lengths)),
        seq_lens=torch.tensor(lengths),
        seq_lens_sum=sum(lengths),
        out_cache_loc=torch.zeros(sum(lengths), dtype=torch.int64),
        extend_start_loc=torch.tensor(starts, dtype=torch.int32),
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
        extend_seq_lens_cpu=cpu_lengths,
    )


def make_metadata_backend():
    backend = object.__new__(MambaAttnBackendBase)
    backend.device = "cpu"
    backend.topk = 1
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda rows: rows,
        translate_mamba_indices=lambda slots: slots,
    )
    return backend


def make_relabeled_decode_batch(seq_lens, track_mask=None):
    """A decoding DP rank after MAX_LEN padding: EXTEND, one token per decode row."""
    bs = len(seq_lens)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=torch.zeros(bs, dtype=torch.int64),
        req_pool_indices=torch.arange(bs),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_sum=sum(seq_lens),
        out_cache_loc=torch.zeros(bs, dtype=torch.int64),
        extend_num_tokens=bs,
        extend_start_loc=torch.arange(bs, dtype=torch.int32),
        extend_seq_lens=torch.ones(bs, dtype=torch.int32),
        extend_prefix_lens=torch.tensor(seq_lens) - 1,
        extend_seq_lens_cpu=[1] * bs,
        extend_prefix_lens_cpu=[n - 1 for n in seq_lens],
        # Decode sets the mask on track boundaries but never mamba_track_seqlens.
        mamba_track_mask=None if track_mask is None else torch.tensor(track_mask),
        mamba_track_indices=None if track_mask is None else torch.arange(bs) + 10,
    )
    batch._original_forward_mode = ForwardMode.DECODE
    return batch


class TestMambaPrefillTrackMetadata(unittest.TestCase):
    def test_cpu_plan_matches_existing_tensor_planner(self):
        rng = random.Random(2026)
        for backend_type in (MambaAttnBackendBase, Mamba2AttnBackend):
            for chunk in (16, 64, 128):
                backend = object.__new__(backend_type)
                backend.device = "cpu"
                backend._mamba_chunk_size = chunk
                cases = [
                    (
                        [chunk + 6, 2 * chunk + 1, chunk],
                        [0, 2 * chunk, 0],
                        [chunk + 1, 3 * chunk + 1, chunk],
                        [True, True, True],
                    ),
                    (
                        [2 * chunk + 1, 1, 1],
                        [chunk, 0, 0],
                        [2 * chunk + 1, 0, 0],
                        [True, False, False],
                    ),
                    ([1, chunk], [0, 0], [0, 0], [False, False]),
                ]
                for _ in range(10):
                    lengths = [rng.randrange(1, chunk * 6) for _ in range(5)]
                    prefix = [rng.randrange(4) * chunk for _ in lengths]
                    cases.append(
                        (
                            lengths,
                            prefix,
                            [
                                p + rng.randrange(1, n + 1)
                                for p, n in zip(prefix, lengths)
                            ],
                            [bool(rng.randrange(2)) for _ in lengths],
                        )
                    )
                for lengths, prefix, track, mask in cases:
                    slots = torch.tensor([111 - i * 5 for i in range(len(lengths))])
                    with self.subTest(
                        backend=backend_type.__name__, chunk=chunk, mask=mask
                    ):
                        expected = backend._init_track_ssm_indices(
                            slots, make_batch(lengths, prefix, track, mask, False)
                        )
                        actual = backend._init_track_ssm_indices(
                            slots.as_subclass(NoHostRead),
                            make_batch(lengths, prefix, track, mask, True),
                        )
                        for result, reference in zip(actual, expected):
                            if reference is None:
                                self.assertIsNone(result)
                            else:
                                torch.testing.assert_close(result, reference)

    def test_verify_and_incomplete_mirrors_use_existing_planner(self):
        batch = make_batch([64], [0], [64], [True], True)
        eligible = MambaAttnBackendBase._has_cpu_prefill_track_metadata
        self.assertTrue(eligible(batch))
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        self.assertFalse(eligible(batch))
        batch.forward_mode = ForwardMode.EXTEND
        batch.mamba_track_seqlens_cpu = None
        self.assertFalse(eligible(batch))
        batch.mamba_track_seqlens_cpu = [64, 0]
        self.assertFalse(eligible(batch))

    def test_logical_token_extent_avoids_scalar_reads_with_valid_cpu_lengths(self):
        backend = make_metadata_backend()
        original_int = torch.Tensor.__int__
        for lengths, starts, cpu_lengths, tbo_range, expected, scalar_reads in (
            ([3, 5], [0, 3], [3, 5], None, 8, 0),
            ([3, 5, 0], [0, 3, 8], [3, 5, 0], None, 8, 0),
            ([3, 5], [4, 7], None, None, 12, 1),
            ([3, 5], [4, 7], [8], None, 12, 1),
            ([3, 5], [4, 7], [3, 5], (4, 12), 12, 1),
        ):
            with self.subTest(cpu_lengths=cpu_lengths, tbo_range=tbo_range):
                batch = make_forward_batch(lengths, starts, cpu_lengths)
                batch.tbo_parent_token_range = tbo_range
                reads = []

                def read_scalar(tensor):
                    if scalar_reads == 0:
                        raise AssertionError(
                            "Valid CPU lengths must avoid scalar reads"
                        )
                    reads.append(tensor.clone())
                    return original_int(tensor)

                with patch.object(torch.Tensor, "__int__", read_scalar):
                    metadata = backend._forward_metadata(batch)
                self.assertEqual(metadata.logical_num_tokens, expected)
                self.assertEqual(len(reads), scalar_reads)
                self.assertEqual(metadata.query_start_loc[-1].item(), expected)

    def test_verify_decode_and_idle_ignore_stale_cpu_token_lengths(self):
        backend = make_metadata_backend()
        for mode, lengths, expected_starts in (
            (ForwardMode.TARGET_VERIFY, [3, 3], [0, 3, 6]),
            (ForwardMode.DECODE, [1, 1], [0, 1, 2]),
            (ForwardMode.IDLE, [], [0]),
        ):
            with self.subTest(mode=mode):
                batch = make_forward_batch(
                    lengths, [0, 3][: len(lengths)], [100, 200], mode
                )
                if mode == ForwardMode.TARGET_VERIFY:
                    batch.spec_info = SimpleNamespace(
                        ragged_verify_layout=None, draft_token_num=3
                    )
                with patch.object(
                    torch.Tensor,
                    "__int__",
                    side_effect=AssertionError("This mode must not read token scalars"),
                ):
                    metadata = backend._forward_metadata(batch)
                self.assertIsNone(metadata.logical_num_tokens)
                torch.testing.assert_close(
                    metadata.query_start_loc,
                    torch.tensor(expected_starts, dtype=torch.int32),
                )

    def test_forward_snapshot_and_padding_do_not_mutate_scheduler_lists(self):
        override = get_context().override_server_args(device="cpu")
        override.install()
        self.addCleanup(override.restore)
        batch = ScheduleBatch(
            reqs=[SimpleNamespace(rid="one", lora_id=None, token_type_ids=None)],
            device="cpu",
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.tensor([3]),
            req_pool_indices=torch.tensor([2]),
            seq_lens=torch.tensor([65]),
            seq_lens_cpu=torch.tensor([65]),
            seq_lens_sum=65,
            out_cache_loc=torch.tensor([1]),
            extend_lens=[1],
            prefix_lens=[64],
            extend_num_tokens=1,
            mamba_track_mask=torch.tensor([True]),
            mamba_track_seqlens=torch.tensor([65]),
            mamba_prefill_track_mask_cpu=[True],
            mamba_track_seqlens_cpu=[65],
        )
        runner = SimpleNamespace(
            device="cpu",
            is_draft_worker=False,
            model_config=SimpleNamespace(
                requires_mm_token_modalities=False, model_is_mrope=False
            ),
            kv_index_translator=SimpleNamespace(rebind_write_loc=lambda forward: None),
            prefill_attention_backend_str="torch_native",
            ngram_embedding_manager=SimpleNamespace(enabled=False),
            lora_manager=None,
            attn_backend=SimpleNamespace(
                get_cpu_graph_seq_len_fill_value=lambda: 1,
                get_cuda_graph_seq_len_fill_value=lambda: 1,
            ),
        )
        forward = ForwardBatch.init_new(
            batch,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        for target, source in (
            ("mamba_prefill_track_mask_cpu", "mamba_prefill_track_mask_cpu"),
            ("mamba_track_seqlens_cpu", "mamba_track_seqlens_cpu"),
            ("extend_seq_lens_cpu", "extend_lens"),
            ("extend_prefix_lens_cpu", "prefix_lens"),
        ):
            self.assertEqual(getattr(forward, target), getattr(batch, source))
            self.assertIsNot(getattr(forward, target), getattr(batch, source))
        forward._pad_inputs_to_size(runner, num_tokens=3, bs=3)
        self.assertEqual(batch.mamba_prefill_track_mask_cpu, [True])
        self.assertEqual(batch.mamba_track_seqlens_cpu, [65])
        self.assertEqual(batch.extend_lens, [1])
        self.assertEqual(batch.prefix_lens, [64])
        for host, device, expected in (
            ("mamba_prefill_track_mask_cpu", "mamba_track_mask", [True, False, False]),
            ("mamba_track_seqlens_cpu", "mamba_track_seqlens", [65, 0, 0]),
            ("extend_seq_lens_cpu", "extend_seq_lens", [1, 0, 0]),
            ("extend_prefix_lens_cpu", "extend_prefix_lens", [64, 0, 0]),
        ):
            self.assertEqual(getattr(forward, host), expected)
            self.assertEqual(getattr(forward, device).tolist(), expected)

    def test_decode_and_verify_clear_prefill_lists_without_losing_snapshot(self):
        for verify in (False, True):
            with self.subTest(verify=verify):
                batch = ScheduleBatch(
                    reqs=[],
                    spec_algorithm=SimpleNamespace(is_none=lambda: False),
                    mamba_track_mask=torch.tensor([True]),
                    mamba_track_seqlens=torch.tensor([65]),
                    mamba_prefill_track_mask_cpu=[True],
                    mamba_track_seqlens_cpu=[65],
                )
                snapshot = batch.copy()
                if verify:
                    settings = SimpleNamespace(
                        mamba=SimpleNamespace(
                            enable_mamba_extra_buffer=True,
                            enable_mamba_extra_buffer_lazy=False,
                        )
                    )
                    with (
                        patch.object(spec_utils, "get_exec", return_value=settings),
                        patch.object(spec_utils, "set_mamba_track_indices_from_reqs"),
                    ):
                        spec_utils.prepare_mamba_track_for_verify(batch)
                    self.assertIsNone(batch.mamba_track_mask)
                    self.assertIsNone(batch.mamba_track_seqlens)
                else:
                    with patch.object(spec_utils, "spec_prepare_for_decode"):
                        batch.prepare_for_decode()
                self.assertIsNone(batch.mamba_prefill_track_mask_cpu)
                self.assertIsNone(batch.mamba_track_seqlens_cpu)
                self.assertEqual(snapshot.mamba_prefill_track_mask_cpu, [True])
                self.assertEqual(snapshot.mamba_track_seqlens_cpu, [65])
                self.assertIsNone(snapshot.mamba_track_mask_cpu)


class TestDpDecodeRelabel(unittest.TestCase):
    """Decode rows relabeled as 1-token extends for DP MAX_LEN padding must keep
    decode semantics on the mamba side."""

    def test_forward_metadata_on_a_track_boundary_stays_decode(self):
        backend = make_metadata_backend()
        backend.conv_states_shape = (1, 8, 4)
        batch = make_relabeled_decode_batch([256, 9], track_mask=[True, False])
        metadata = backend._forward_metadata(batch)
        torch.testing.assert_close(
            metadata.query_start_loc, torch.arange(3, dtype=torch.int32)
        )
        self.assertIsNone(metadata.logical_num_tokens)
        self.assertIsNone(metadata.track_conv_indices)

    def test_prepare_mixed_builds_decode_metadata(self):
        metadata = Mamba2Metadata.prepare_mixed(
            ForwardMetadata(
                query_start_loc=torch.arange(4, dtype=torch.int32),
                mamba_cache_indices=torch.arange(3, dtype=torch.int32),
            ),
            64,
            make_relabeled_decode_batch([8, 8, 8]),
        )
        self.assertEqual(metadata.num_decodes, 3)
        self.assertEqual(metadata.num_prefills, 0)
        self.assertIsNone(metadata.mixed_metadata)

    def test_prepare_mixed_keeps_real_extends_as_prefills(self):
        batch = make_forward_batch([4, 3], [0, 4], [4, 3])
        batch.extend_num_tokens = 7
        batch.extend_prefix_lens = torch.zeros(2, dtype=torch.int64)
        metadata = Mamba2Metadata.prepare_mixed(
            ForwardMetadata(
                query_start_loc=torch.tensor([0, 4, 7], dtype=torch.int32),
                mamba_cache_indices=torch.arange(2, dtype=torch.int32),
            ),
            64,
            batch,
        )
        self.assertEqual(metadata.num_prefills, 2)
        self.assertEqual(metadata.num_decodes, 0)
        self.assertIsNotNone(metadata.mixed_metadata)

    def test_logical_mode_ignores_idle_relabels(self):
        # Idle ranks also set _original_forward_mode; their fabricated rows stay
        # in the relabeled mode.
        for mode in (ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                batch = make_forward_batch([1], [0], [1], mode)
                batch._original_forward_mode = ForwardMode.IDLE
                self.assertEqual(batch.logical_forward_mode, mode)


if __name__ == "__main__":
    unittest.main()
