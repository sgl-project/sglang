import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
    DeepseekV4MultiStepBackend,
    DSV4AttnMetadata,
    DSV4Metadata,
    UnifiedKvMetadata,
    _match_num_queries,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

SWA_WINDOW = 128
PAGE_SIZE = 256
NUM_REQ_SLOTS = 6
MAX_CONTEXT = 512
# Block slots (out_cache_loc) live past every req_to_token slot.
OUT_LOC_BASE = NUM_REQ_SLOTS * MAX_CONTEXT + 1
NUM_FULL_SLOTS = OUT_LOC_BASE + 4096


def _make_backend(*, block_size, device, is_dspark_draft=True, low_ratios=()):
    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.device = device
    backend.cuda_int32_kwargs = {"device": device, "dtype": torch.int32}
    backend.swa_page_size = SWA_WINDOW
    backend.page_size = PAGE_SIZE
    g = torch.Generator().manual_seed(0)
    # Distinct full slots per (request slot, position); slot 0 stays the dummy.
    req_to_token = (
        torch.randperm(NUM_REQ_SLOTS * MAX_CONTEXT, generator=g).view(
            NUM_REQ_SLOTS, MAX_CONTEXT
        )
        + 1
    )
    backend.req_to_token = req_to_token.to(device=device, dtype=torch.int32)
    backend.req_to_token_pool = SimpleNamespace(req_to_token=backend.req_to_token)
    backend.MAX_SEQ_LEN_FOR_CAPTURE = MAX_CONTEXT
    # full -> swa: injective and not the identity, so a wrong source shows.
    full_to_swa = (torch.arange(NUM_FULL_SLOTS) * 3 + 7).to(
        device=device, dtype=torch.int64
    )
    backend.token_to_kv_pool = SimpleNamespace(
        full_to_swa_index_mapping=full_to_swa,
        translate_loc_from_full_to_swa=lambda idx: full_to_swa[idx.to(torch.int64)],
        unified_swa_pages=0,
        request_window=None,
    )
    backend.index_topk = 512
    backend.present_ratios = tuple(sorted(low_ratios))
    backend.low_ratios = tuple(low_ratios)
    backend.has_c4 = False
    backend.has_c128 = False
    backend.candidate_masks = None
    backend.low_ratio_identity_skip = False
    backend.low_ratio_candidate_span = None
    backend.enable_deepseek_v4_fp4_indexer = False
    backend.topk = 1
    backend.mtp_enabled = True
    backend.speculative_num_steps = 0
    backend.speculative_step_id = 0
    backend.speculative_num_draft_tokens = block_size + 1
    backend.is_dspark = True
    backend.needs_cpu_seq_lens = False
    backend._fp4_graph_row_limit = None
    backend.is_draft_worker = is_dspark_draft
    # compressed-KV metadata is built only on the target worker
    backend.need_compress = not is_dspark_draft
    backend.is_dspark_draft = is_dspark_draft
    # The draft verifies gamma rows, the target gamma + 1 (bonus included).
    backend.target_verify_num_draft_tokens = (
        block_size if is_dspark_draft else block_size + 1
    )
    backend.forward_metadata = None
    return backend


def _expected_block_row(backend, *, req_slot, prefix, out_loc_block, width):
    """Reference window row: the last min(prefix, SWA_WINDOW) committed tokens,
    then the block's slots, -1 padded; all through the full -> swa map."""
    full_to_swa = backend.token_to_kv_pool.full_to_swa_index_mapping
    ctx = min(prefix, SWA_WINDOW)
    row = torch.full((width,), -1, dtype=torch.int32, device=full_to_swa.device)
    if ctx:
        ctx_full = backend.req_to_token[req_slot, prefix - ctx : prefix].to(torch.int64)
        row[:ctx] = full_to_swa[ctx_full].to(torch.int32)
    row[ctx : ctx + out_loc_block.numel()] = full_to_swa[out_loc_block].to(torch.int32)
    return row, ctx


@unittest.skipUnless(is_hip(), "DeepSeek V4 HIP radix backend requires ROCm")
class TestDSV4HipBreakableCudaGraphMetadata(unittest.TestCase):
    @staticmethod
    def _make_core_metadata(base: int) -> DSV4AttnMetadata:
        def tensor(offset: int) -> torch.Tensor:
            return torch.tensor([base + offset], dtype=torch.int32)

        def fill_optional_tensors(metadata, start: int) -> int:
            for metadata_field in dataclasses.fields(metadata):
                if "Tensor" not in str(metadata_field.type):
                    continue
                if getattr(metadata, metadata_field.name, None) is None:
                    setattr(metadata, metadata_field.name, tensor(start))
                    start += 1
            return start

        metadata = DSV4AttnMetadata(
            page_size=256,
            page_table=torch.tensor([[base + 1, base + 2]], dtype=torch.int32),
            raw_out_loc=torch.tensor([base + 3], dtype=torch.int32),
            cuda_int32_kwargs={"dtype": torch.int32},
            seq_lens_casual=torch.tensor([base + 4], dtype=torch.int32),
            positions_casual=torch.tensor([base + 5], dtype=torch.int32),
            swa_page_indices=torch.tensor([[base + 6, base + 7]], dtype=torch.int32),
            swa_topk_lengths=torch.tensor([base + 8], dtype=torch.int32),
            index_topk=512,
            swa_out_cache_loc=torch.tensor([base + 9], dtype=torch.int32),
            unified=UnifiedKvMetadata(),
        )
        next_offset = fill_optional_tensors(metadata, 10)
        fill_optional_tensors(metadata.unified, next_offset)
        metadata.c0_flashmla_metadata = None
        metadata.c4_flashmla_metadata = None
        metadata.c128_flashmla_metadata = None
        return metadata

    def test_backend_opts_into_captured_bcg_metadata(self):
        self.assertTrue(
            DeepseekV4HipRadixBackend.use_captured_forward_metadata_for_breakable_cuda_graph
        )
        self.assertTrue(
            DeepseekV4HipRadixBackend.prefer_eager_mixed_prefill_under_dp_attention
        )

    def test_non_unified_metadata_matches_underfilled_bucket(self):
        captured = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        replay = _match_num_queries(captured, 3, value=-1)
        self.assertEqual(replay.tolist(), [[1, 2], [3, 4], [5, 6]])

        short = torch.tensor([9, 10])
        replay = _match_num_queries(short, 3, value=1)
        self.assertEqual(replay.tolist(), [9, 10, 1])
        self.assertIsNone(_match_num_queries(None, 3, value=0))

    def test_unified_prefill_metadata_pads_to_capture_bucket(self):
        backend = object.__new__(DeepseekV4HipRadixBackend)
        backend.low_ratios = ()
        backend.has_c4 = True
        backend.has_c128 = True
        backend.token_to_kv_pool = SimpleNamespace(unified_swa_window=128)
        core = self._make_core_metadata(0)
        core.positions_casual = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
        core.unified = None

        with (
            mock.patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate."
                "is_unified_kv_triton",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix."
                "torch.repeat_interleave",
                wraps=torch.repeat_interleave,
            ) as repeat_interleave,
        ):
            backend._attach_unified_kv_prefill_meta(
                core,
                req_pool_indices=torch.tensor([7, 9], dtype=torch.int32),
                req_pool_indices_repeated=torch.tensor([7, 9, 9, 9], dtype=torch.int32),
                seq_lens=torch.tensor([1, 3], dtype=torch.int32),
                extend_seq_lens=torch.tensor([1, 2], dtype=torch.int32),
                num_tokens=3,
                exact_num_tokens=True,
            )

        repeat_interleave.assert_called_once()
        self.assertEqual(repeat_interleave.call_args.kwargs["output_size"], 3)
        self.assertEqual(core.unified.pf_state_slot.tolist(), [7, 9, 9, 9])
        self.assertEqual(core.unified.pf_chunk_start.tolist(), [0, 1, 1, 0])
        self.assertEqual(core.unified.pf_cu_q.tolist(), [0, 1, 1, 0])
        self.assertEqual(core.unified.pf_final_pos.tolist(), [0, 2, 2, 128])

    def test_eager_prefill_marks_host_proven_token_count_exact(self):
        backend = object.__new__(DeepseekV4HipRadixBackend)
        backend.low_ratios = ()
        backend.has_c4 = True
        backend.has_c128 = True
        backend.req_to_token = torch.zeros((2, 8), dtype=torch.int32)
        backend.token_to_kv_pool = object()
        core = self._make_core_metadata(0)
        extend_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        backend.make_core_attn_metadata = mock.Mock(return_value=core)
        backend._attach_unified_kv_prefill_meta = mock.Mock()
        backend.init_forward_metadata_indexer = mock.Mock(return_value=None)

        with (
            mock.patch(
                "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels."
                "ExpandPrefillCausally.execute",
                return_value=SimpleNamespace(
                    seq_lens_casual=core.seq_lens_casual,
                    req_pool_indices_repeated=torch.tensor(
                        [7, 9, 9], dtype=torch.int32
                    ),
                ),
            ) as expand_prefill,
            mock.patch(
                "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix."
                "create_paged_compressor_data",
                return_value=None,
            ),
        ):
            backend.init_forward_metadata_prefill(
                max_seq_len=4096,
                req_pool_indices=torch.tensor([7, 9], dtype=torch.int32),
                seq_lens=torch.tensor([1, 3], dtype=torch.int32),
                seq_lens_cpu=[1, 3],
                out_cache_loc=torch.zeros(3, dtype=torch.int64),
                num_tokens=3,
                extend_seq_lens=torch.tensor([1, 2], dtype=torch.int32),
                extend_seq_lens_cpu=[1, 2],
                extend_start_loc=extend_start_loc,
                use_prefill_cuda_graph=False,
                exact_num_tokens=False,
            )

        self.assertIs(
            expand_prefill.call_args.kwargs["extend_start_loc"], extend_start_loc
        )
        self.assertTrue(
            backend._attach_unified_kv_prefill_meta.call_args.kwargs["exact_num_tokens"]
        )

    def test_prefill_bcg_uses_bucket_sized_gpu_compressor_plans(self):
        backend = object.__new__(DeepseekV4HipRadixBackend)
        backend.low_ratios = ()
        backend.has_c4 = True
        backend.has_c128 = True
        backend.req_to_token = torch.zeros((2, 8), dtype=torch.int32)
        backend.token_to_kv_pool = object()
        core = self._make_core_metadata(0)
        core.positions_casual = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
        backend.make_core_attn_metadata = mock.Mock(return_value=core)
        backend._attach_unified_kv_prefill_meta = mock.Mock()
        backend.init_forward_metadata_indexer = mock.Mock(return_value=None)

        with mock.patch(
            "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix."
            "create_paged_compressor_data",
            side_effect=lambda compress_ratio, **kwargs: (compress_ratio, kwargs),
        ) as create_plan:
            backend.init_forward_metadata_prefill(
                max_seq_len=4096,
                req_pool_indices=torch.tensor([7, 9], dtype=torch.int32),
                seq_lens=torch.tensor([1, 3], dtype=torch.int32),
                seq_lens_cpu=[1, 3],
                out_cache_loc=torch.zeros(4, dtype=torch.int64),
                num_tokens=3,
                extend_seq_lens=torch.tensor([1, 2], dtype=torch.int32),
                extend_seq_lens_cpu=[1, 2],
                use_prefill_cuda_graph=True,
            )

        self.assertEqual(create_plan.call_count, 2)
        for call in create_plan.call_args_list:
            self.assertIsNone(call.kwargs["seq_lens_cpu"])
            self.assertIsNone(call.kwargs["extend_lens_cpu"])
            self.assertEqual(call.kwargs["num_q_tokens"], 4)
            self.assertTrue(call.kwargs["use_prefill_cuda_graph"])

    def test_gpu_compressor_plan_invalidates_bucket_tail(self):
        from sglang.kernels.ops.attention.dsv4 import CompressorPrefillPlan
        from sglang.test.kernels.deepseek_v4.common import make_paged_context

        seq_lens = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        extend_lens = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        for compress_ratio in (4, 128):
            with self.subTest(compress_ratio=compress_ratio):
                context = make_paged_context(bs=2, compress_ratio=compress_ratio)
                plan = CompressorPrefillPlan.generate(
                    compress_ratio=compress_ratio,
                    req_pool_indices=context.req_pool_indices,
                    seq_lens=seq_lens,
                    extend_lens=extend_lens,
                    req_to_token=context.req_to_token,
                    full_to_state=context.full_to_swa,
                    swa_page_size=context.swa_page_size,
                    ring_size=context.ring_size,
                    num_q_tokens=4,
                    use_cuda_graph=True,
                )

                self.assertEqual(plan.plan_c.shape, (4, 16))
                self.assertEqual(plan.plan_w.shape, (4, 8))
                ragged_ids = plan.plan_w.view(torch.uint32).view(-1, 2)[:, 0]
                self.assertEqual(ragged_ids[:3].cpu().tolist(), [0, 1, 2])
                self.assertEqual(int(ragged_ids[3].item()), 0xFFFFFFFF)

    def test_capture_builds_graph_compatible_metadata_and_workspace(self):
        capture_metadata = DSV4Metadata(object(), indexer_metadata=None)
        backend = object.__new__(DeepseekV4HipRadixBackend)
        backend.low_ratios = ()
        backend.has_c4 = True
        backend.has_c128 = True
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 4096
        backend._build_forward_metadata = mock.Mock(return_value=capture_metadata)
        backend.init_forward_metadata_in_graph = mock.Mock()
        backend._refresh_fp4_prefill_workspace = mock.Mock()
        forward_batch = SimpleNamespace(name="capture")

        result = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
            forward_batch
        )

        backend._build_forward_metadata.assert_called_once_with(
            forward_batch,
            max_seq_len_override=backend.MAX_SEQ_LEN_FOR_CAPTURE,
            use_prefill_cuda_graph=True,
        )
        backend.init_forward_metadata_in_graph.assert_called_once_with(forward_batch)
        backend._refresh_fp4_prefill_workspace.assert_called_once_with(forward_batch)
        self.assertIs(result, capture_metadata)
        self.assertIs(backend.forward_metadata, capture_metadata)

    def test_refresh_preserves_captured_hip_tensor_storage(self):
        capture_workspace = object()
        capture_metadata = DSV4Metadata(
            self._make_core_metadata(0),
            indexer_metadata=None,
            fp4_prefill_workspace=capture_workspace,
            fp4_k_write_metadata=(
                torch.tensor([14], dtype=torch.int64),
                torch.tensor([15], dtype=torch.int64),
            ),
            fp4_q_positions=torch.tensor([16], dtype=torch.int64),
        )
        replay_metadata = DSV4Metadata(
            self._make_core_metadata(100),
            indexer_metadata=None,
            fp4_k_write_metadata=(
                torch.tensor([114], dtype=torch.int64),
                torch.tensor([115], dtype=torch.int64),
            ),
            fp4_q_positions=torch.tensor([116], dtype=torch.int64),
        )
        capture_core = capture_metadata.core_attn_metadata
        replay_core = replay_metadata.core_attn_metadata
        captured_core_tensors = {
            field.name: getattr(capture_core, field.name)
            for field in dataclasses.fields(capture_core)
            if torch.is_tensor(getattr(capture_core, field.name))
        }
        captured_unified_tensors = {
            field.name: getattr(capture_core.unified, field.name)
            for field in dataclasses.fields(capture_core.unified)
            if torch.is_tensor(getattr(capture_core.unified, field.name))
        }
        captured_fp4_tensors = {
            "fp4_k_positions": capture_metadata.fp4_k_write_metadata[0],
            "fp4_k_slots": capture_metadata.fp4_k_write_metadata[1],
            "fp4_q_positions": capture_metadata.fp4_q_positions,
        }
        expected_core_tensors = {
            name: getattr(replay_core, name).clone() for name in captured_core_tensors
        }
        expected_unified_tensors = {
            name: getattr(replay_core.unified, name).clone()
            for name in captured_unified_tensors
        }
        replay_fp4_tensors = {
            "fp4_k_positions": replay_metadata.fp4_k_write_metadata[0],
            "fp4_k_slots": replay_metadata.fp4_k_write_metadata[1],
            "fp4_q_positions": replay_metadata.fp4_q_positions,
        }
        expected_fp4_tensors = {
            name: tensor.clone() for name, tensor in replay_fp4_tensors.items()
        }

        capture_metadata.refresh_for_breakable_cuda_graph_replay_(replay_metadata)

        for field_name, captured_tensor in captured_core_tensors.items():
            current = getattr(capture_core, field_name)
            self.assertIs(current, captured_tensor, field_name)
            self.assertTrue(
                torch.equal(current, expected_core_tensors[field_name]), field_name
            )
            self.assertTrue(
                torch.equal(
                    getattr(replay_core, field_name),
                    expected_core_tensors[field_name],
                ),
                f"{field_name} replay source",
            )
        for field_name, captured_tensor in captured_unified_tensors.items():
            current = getattr(capture_core.unified, field_name)
            self.assertIs(current, captured_tensor, field_name)
            self.assertTrue(
                torch.equal(current, expected_unified_tensors[field_name]),
                field_name,
            )
            self.assertTrue(
                torch.equal(
                    getattr(replay_core.unified, field_name),
                    expected_unified_tensors[field_name],
                ),
                f"{field_name} replay source",
            )

        current_fp4_tensors = {
            "fp4_k_positions": capture_metadata.fp4_k_write_metadata[0],
            "fp4_k_slots": capture_metadata.fp4_k_write_metadata[1],
            "fp4_q_positions": capture_metadata.fp4_q_positions,
        }
        for name, captured_tensor in captured_fp4_tensors.items():
            self.assertIs(current_fp4_tensors[name], captured_tensor)
            self.assertTrue(
                torch.equal(captured_tensor, expected_fp4_tensors[name]), name
            )
            self.assertTrue(
                torch.equal(replay_fp4_tensors[name], expected_fp4_tensors[name]),
                f"{name} replay source",
            )
        self.assertIs(capture_metadata.fp4_prefill_workspace, capture_workspace)

    def test_replay_refreshes_captured_metadata_and_workspace(self):
        capture_metadata = DSV4Metadata(object(), indexer_metadata=None)
        replay_metadata = DSV4Metadata(object(), indexer_metadata=None)
        capture_metadata.refresh_for_breakable_cuda_graph_replay_ = mock.Mock()

        backend = object.__new__(DeepseekV4HipRadixBackend)
        backend.low_ratios = ()
        backend.has_c4 = True
        backend.has_c128 = True
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 4096
        backend._build_forward_metadata = mock.Mock(return_value=replay_metadata)
        backend.init_forward_metadata_in_graph = mock.Mock()
        backend._refresh_fp4_prefill_workspace = mock.Mock()

        forward_batch = SimpleNamespace(name="live")
        static_forward_batch = SimpleNamespace(name="static")
        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            capture_metadata,
            forward_batch,
            static_forward_batch=static_forward_batch,
        )

        backend._build_forward_metadata.assert_called_once_with(
            static_forward_batch,
            max_seq_len_override=backend.MAX_SEQ_LEN_FOR_CAPTURE,
            use_prefill_cuda_graph=True,
        )
        backend.init_forward_metadata_in_graph.assert_called_once_with(
            static_forward_batch
        )
        capture_metadata.refresh_for_breakable_cuda_graph_replay_.assert_called_once_with(
            replay_metadata
        )
        backend._refresh_fp4_prefill_workspace.assert_called_once_with(
            static_forward_batch
        )
        self.assertIs(backend.forward_metadata, capture_metadata)

    def test_multistep_backend_forwards_bcg_metadata_hooks(self):
        backend = object.__new__(DeepseekV4MultiStepBackend)
        backend.speculative_num_steps = 3
        backend.attn_backends = [mock.Mock(), mock.Mock(), mock.Mock()]
        forward_batch = SimpleNamespace(name="live")
        static_forward_batch = SimpleNamespace(name="static")
        capture_metadata = [object(), object()]

        for index, child in enumerate(backend.attn_backends[:-1]):
            child.init_forward_metadata_for_breakable_cuda_graph_capture.return_value = f"capture-{index}"

        captured = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
            forward_batch
        )
        self.assertEqual(captured, ["capture-0", "capture-1"])

        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            capture_metadata,
            forward_batch,
            static_forward_batch=static_forward_batch,
        )
        for index, child in enumerate(backend.attn_backends[:-1]):
            child.init_forward_metadata_for_breakable_cuda_graph_capture.assert_called_once_with(
                forward_batch
            )
            child.prepare_forward_metadata_for_breakable_cuda_graph_replay.assert_called_once_with(
                capture_metadata[index],
                forward_batch,
                static_forward_batch=static_forward_batch,
            )
        backend.attn_backends[
            -1
        ].init_forward_metadata_for_breakable_cuda_graph_capture.assert_not_called()
        backend.attn_backends[
            -1
        ].prepare_forward_metadata_for_breakable_cuda_graph_replay.assert_not_called()


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestDsparkDraftBlockWindowHip(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")

    def test_graph_capture_and_replay_route_the_draft_through_the_block_window(self):
        """A TARGET_VERIFY capture on the DSpark draft must build the block window, and
        a replay must refresh it in place for new lengths and slots, whether or not the
        replay batch carries a CPU mirror of the lengths (the draft-window bucket must
        not read seq_lens_cpu)."""
        for cpu_mirror in (True, False):
            with self.subTest(cpu_mirror=cpu_mirror):
                self._check_block_window_replay(cpu_mirror=cpu_mirror)

    def _check_block_window_replay(self, *, cpu_mirror):
        block = 3
        bs = 2
        backend = _make_backend(block_size=block, device=self.device)
        backend.init_cuda_graph_state(max_bs=bs, max_num_tokens=bs * block)

        def make_batch(prefix_list, out_base, cpu_extended):
            prefix = torch.tensor(prefix_list, dtype=torch.int32, device=self.device)
            out_loc = (
                torch.arange(bs * block, device=self.device, dtype=torch.int64)
                + out_base
            )
            seq_lens_cpu = torch.tensor(prefix_list, dtype=torch.int64)
            if cpu_extended:
                # The worker hands the draft a seq_lens_cpu already + block.
                seq_lens_cpu = seq_lens_cpu + block
            return SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                req_pool_indices=torch.tensor(
                    [4, 1], dtype=torch.int32, device=self.device
                ),
                seq_lens=prefix,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=int(seq_lens_cpu.sum()),
                positions=torch.zeros(
                    bs * block, dtype=torch.int64, device=self.device
                ),
                out_cache_loc=out_loc,
                spec_info=SimpleNamespace(draft_token_num=block),
            )

        capture_batch = make_batch([1, 1], OUT_LOC_BASE, cpu_extended=False)
        backend.init_forward_metadata_out_graph(capture_batch, in_capture=True)
        captured = backend.forward_metadata
        core = captured.core_attn_metadata
        width = core.swa_page_indices.shape[1]

        replay_batch = make_batch([150, 7], OUT_LOC_BASE + 100, cpu_extended=True)
        if not cpu_mirror:
            replay_batch.seq_lens_cpu = None
            replay_batch.seq_lens_sum = None
        backend.init_forward_metadata_out_graph(replay_batch)
        self.assertIs(backend.forward_metadata, captured)
        for b, p in enumerate([150, 7]):
            row, ctx = _expected_block_row(
                backend,
                req_slot=int(replay_batch.req_pool_indices[b]),
                prefix=p,
                out_loc_block=replay_batch.out_cache_loc[b * block : (b + 1) * block],
                width=width,
            )
            for j in range(block):
                r = b * block + j
                self.assertTrue(torch.equal(core.swa_page_indices[r], row), (b, j))
                self.assertEqual(int(core.swa_topk_lengths[r]), ctx + block)
                # Derived from the device prefix, not the pre-extended CPU copy.
                self.assertEqual(int(core.positions_casual[r]), p + j)


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestAiterSparseLengthFoldPerStep(CustomTestCase):
    """The length-fold cache must not outlive the forward: a TARGET_VERIFY bucket keeps
    one core object across steps, so the warmup's lists would be replayed."""

    def test_graph_bucket_refold_follows_the_new_lengths(self):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            _fold_lengths_for_aiter_sparse,
        )

        device = torch.device("cuda")
        block, bs = 3, 2
        backend = _make_backend(block_size=block, device=device)
        backend.init_cuda_graph_state(max_bs=bs, max_num_tokens=bs * block)

        def make_batch(prefix_list, out_base):
            prefix = torch.tensor(prefix_list, dtype=torch.int32, device=device)
            seq_lens_cpu = torch.tensor(prefix_list, dtype=torch.int64)
            return SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                req_pool_indices=torch.tensor([4, 1], dtype=torch.int32, device=device),
                seq_lens=prefix,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=int(seq_lens_cpu.sum()),
                positions=torch.zeros(bs * block, dtype=torch.int64, device=device),
                out_cache_loc=torch.arange(bs * block, device=device, dtype=torch.int64)
                + out_base,
                spec_info=SimpleNamespace(draft_token_num=block),
            )

        def fold(core):
            masked, _ = _fold_lengths_for_aiter_sparse(
                core,
                0,
                core.swa_page_indices.unsqueeze(1),
                core.swa_topk_lengths,
                None,
                None,
            )
            return masked.squeeze(1)

        # Warmup-like forward at the capture lengths (prefix 1: 1 + block valid).
        backend.init_forward_metadata_out_graph(
            make_batch([1, 1], OUT_LOC_BASE), in_capture=True
        )
        core = backend.forward_metadata.core_attn_metadata
        warm = fold(core).clone()
        self.assertTrue(
            torch.equal(
                (warm >= 0).sum(1), torch.full((bs * block,), 1 + block, device=device)
            )
        )

        # Next step: replay copies longer contexts into the same bucket object.
        replay = make_batch([150, 7], OUT_LOC_BASE + 100)
        backend.init_forward_metadata_out_graph(replay)
        self.assertIs(backend.forward_metadata.core_attn_metadata, core)
        backend.init_forward_metadata_in_graph(replay)
        got = fold(core)
        expect_valid = torch.tensor(
            [SWA_WINDOW + block] * block + [7 + block] * block, device=device
        )
        self.assertTrue(torch.equal((got >= 0).sum(1), expect_valid))
        self.assertTrue(
            torch.equal(
                got,
                core.swa_page_indices.clone().masked_fill(
                    torch.arange(got.shape[1], device=device)[None, :]
                    >= core.swa_topk_lengths[:, None],
                    -1,
                ),
            )
        )
        self.assertFalse(torch.equal(got, warm))


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestLowRatioTargetVerifyHip(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        import sglang.srt.layers.attention.deepseek_v4_backend_hip_radix as module

        self.module = module

    def test_target_verify_device_lengths_match_cpu_and_replay(self):
        backend = _make_backend(block_size=4, device=self.device, is_dspark_draft=False)
        backend.has_c128 = True
        backend.token_to_kv_pool.swa_page_size = SWA_WINDOW
        backend.token_to_kv_pool._unified_kv = False
        backend.token_to_kv_pool.get_ring_size = lambda **kwargs: 128
        prefix = torch.tensor([125, 255], dtype=torch.int32, device=self.device)
        slots = torch.tensor([1, 4], dtype=torch.int32, device=self.device)
        count = 2 * backend.target_verify_num_draft_tokens
        out_loc = torch.arange(count, device=self.device) + OUT_LOC_BASE

        def build(cpu_lengths):
            return backend.init_forward_metadata_target_verify_old(
                max_seq_len=MAX_CONTEXT - backend.target_verify_num_draft_tokens,
                req_pool_indices=slots,
                seq_lens=prefix,
                seq_lens_cpu=cpu_lengths,
                out_cache_loc=out_loc,
                use_prefill_cuda_graph=True,
            )

        # Build the full compressor plan on device, not just the raw wrapper.
        build(None)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = build(None)
        for lengths, request_slots in (([125, 255], [1, 4]), ([254, 381], [4, 2])):
            prefix.copy_(torch.tensor(lengths, device=self.device))
            slots.copy_(torch.tensor(request_slots, device=self.device))
            out_loc.add_(count)
            graph.replay()
            reference = build(lengths)
            for name in ("seq_lens_casual", "swa_page_indices", "swa_topk_lengths"):
                self.assertTrue(
                    torch.equal(
                        getattr(captured.core_metadata, name),
                        getattr(reference.core_metadata, name),
                    ),
                    name,
                )
            for name in ("plan_c", "plan_w"):
                self.assertTrue(
                    torch.equal(
                        getattr(captured.c128_compress_metadata, name),
                        getattr(reference.c128_compress_metadata, name),
                    ),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
