import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    Mamba2AttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative import spec_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
    backend.req_to_token_pool = SimpleNamespace(get_mamba_indices=lambda rows: rows)
    backend.state_pool = None
    backend._state_ids = IdSpace(key=("mamba", None))
    return backend


def make_relabeled_decode_batch(seq_lens, num_tokens, track_mask=None):
    """A decoding DP rank after MAX_LEN padding: EXTEND, one token per decode row.

    ``num_tokens`` pads the token-indexed inputs to the cross-rank max, as
    ``_pad_inputs_to_size`` does; the per-row extend lists stay at one token.
    """
    bs = len(seq_lens)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        req_pool_indices=torch.arange(bs),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_sum=sum(seq_lens),
        out_cache_loc=torch.zeros(num_tokens, dtype=torch.int64),
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
    def test_short_conv_cpu_capture_without_cuda_tracking_buffer(self):
        """CPU capture must initialize conv metadata without CUDA-only tracking."""
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend.device = "cpu"
        backend.pad_slot_id = -1
        backend.topk = 1
        backend.state_indices_list = []
        backend.query_start_loc_list = []
        backend.mamba_track_indices_buf = None
        backend.replayssm_write_pos_list = None
        backend.replayssm_force_flush_list = None
        backend._cache_indices_buf = None
        backend._state_ids = SimpleNamespace(ids=lambda ids: ids)
        mapping = torch.tensor([0, 7, 3], dtype=torch.int32)
        backend.req_to_token_pool = SimpleNamespace(
            get_mamba_indices=lambda rows: mapping[rows]
        )
        backend.init_cpu_graph_state(2, 2)
        backend.init_forward_metadata_capture_cpu_graph(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([8, 9]),
            out_cache_loc=torch.tensor([4, 5]),
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
        torch.testing.assert_close(
            backend._cache_indices, torch.tensor([7, 3], dtype=torch.int64)
        )
        torch.testing.assert_close(
            backend.forward_metadata.query_start_loc,
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        self.assertIsNone(backend.state_tracking_indices)

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
            reqs=[
                SimpleNamespace(
                    rid="one",
                    lora_id=None,
                    token_type_ids=None,
                    kv=SimpleNamespace(kv_committed_len=64),
                    extend_end=65,
                    full_untruncated_fill_ids=[3] * 65,
                )
            ],
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
            # A pass-through runner: its plan hands the batch its ids untouched.
            runner_ids=SimpleNamespace(
                own_plan=lambda batch: SimpleNamespace(
                    bind=lambda batch, reader, cols=None: None
                )
            ),
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
        # A peer's prefill pads the 2 decode rows to 697 tokens; the metadata
        # must still cover only the real rows.
        batch = make_relabeled_decode_batch(
            [256, 9], num_tokens=697, track_mask=[True, False]
        )
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
            make_relabeled_decode_batch([8, 8, 8], num_tokens=64),
        )
        self.assertEqual(metadata.num_decodes, 3)
        self.assertEqual(metadata.num_prefills, 0)
        self.assertIsNone(metadata.mixed_metadata)

    def test_two_batch_overlap_rejects_relabeled_batches(self):
        with self.assertRaisesRegex(AssertionError, "Two-batch overlap"):
            TboForwardBatchPreparer.filter_batch(
                make_relabeled_decode_batch([8, 8], num_tokens=4),
                start_token_index=0,
                end_token_index=1,
                start_seq_index=0,
                end_seq_index=1,
                out_num_token_non_padded=torch.zeros(1, dtype=torch.int32),
            )

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


def track_mapping(batch):
    return batch.req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping


def make_verify_batch():
    mapping = torch.tensor([[10, 11], [20, 21], [30, 31]])
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=p))
            for p in (0, None, 1)
        ],
        req_pool_indices=torch.tensor([2, 0, 1]),
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
        ),
        mamba_lazy_spec_track_positions_cpu=[1, 0, 0],
    )


class TestVerifyTrackIndexReuse(unittest.TestCase):
    """Verify track indices reused from the last round must equal freshly built ones
    after any change to what they are built from."""

    def setUp(self):
        self.builds = 0
        build, tensor = spec_utils.set_mamba_track_indices_from_reqs, torch.tensor

        def build_unpinned(batch, track_positions=None):
            self.builds += 1
            # no pinned host memory on CPU-only machines
            with patch.object(
                torch, "tensor", lambda *a, pin_memory=False, **k: tensor(*a, **k)
            ):
                build(batch, track_positions)

        patcher = patch.object(
            spec_utils, "set_mamba_track_indices_from_reqs", build_unpinned
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _prepare(self, batch, reuse, lazy=False):
        settings = SimpleNamespace(
            mamba=SimpleNamespace(
                enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=lazy
            )
        )
        # reuse: the process's indices under SGLANG_Q35MF_MKH_N3B, None without it
        with (
            patch.object(spec_utils, "get_exec", return_value=settings),
            patch.object(spec_utils, "_VERIFY_TRACK_INDICES", reuse),
        ):
            spec_utils.prepare_mamba_track_for_verify(batch)
        return batch.mamba_track_indices.tolist(), batch.mamba_track_buffer_indices

    def test_reuse_matches_a_fresh_build(self):
        changes = {
            "nothing": lambda b: None,
            "slots replaced": lambda b: setattr(
                b, "req_pool_indices", torch.tensor([0, 2, 1])
            ),
            "slots written": lambda b: b.req_pool_indices.copy_(
                torch.tensor([1, 2, 0])
            ),
            "mapping replaced": lambda b: setattr(
                b.req_to_token_pool,
                "req_index_to_mamba_ping_pong_track_buffer_mapping",
                torch.tensor([[40, 41], [50, 51], [60, 61]]),
            ),
            "mapping written": lambda b: track_mapping(b)[0, 1].fill_(19),
            "position moved": lambda b: setattr(
                b.reqs[2].kv, "mamba_next_track_idx", 0
            ),
            "request freed": lambda b: setattr(
                b.reqs[0].kv, "mamba_next_track_idx", None
            ),
            "indices written": lambda b: b.mamba_track_indices.fill_(0),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                batch, reuse = make_verify_batch(), spec_utils.VerifyTrackIndices()
                self._prepare(batch, reuse)
                last_indices, builds = batch.mamba_track_indices, self.builds
                change(batch)
                got = self._prepare(batch, reuse)
                # reused, the very same tensor, exactly when nothing changed
                self.assertEqual(self.builds == builds, name == "nothing")
                self.assertEqual(
                    batch.mamba_track_indices is last_indices, name == "nothing"
                )
                self.assertEqual(got, self._prepare(batch, None))

    def test_inference_tensors_are_rebuilt_every_round(self):
        # SGLANG_ENABLE_TORCH_INFERENCE_MODE: inference tensors keep no version
        with torch.inference_mode():
            batch, reuse = make_verify_batch(), spec_utils.VerifyTrackIndices()
            want = self._prepare(batch, None)
            for _ in range(2):
                builds = self.builds
                self.assertEqual(self._prepare(batch, reuse), want)
                self.assertEqual(self.builds, builds + 1)

    def test_lazy_plan_is_never_reused_over(self):
        batch, reuse = make_verify_batch(), spec_utils.VerifyTrackIndices()
        self._prepare(batch, reuse)
        self.assertEqual(
            self._prepare(batch, reuse, lazy=True), ([31, -1, 20], [1, 0, 0])
        )


if __name__ == "__main__":
    unittest.main()
