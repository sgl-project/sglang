"""CPU contracts for the NPU compact port; NPU operator execution is separate."""

import ast
import copy
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    scatter_compact_to_strided,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cache_location_kernel_cpu_reference(
    *,
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    batch_size,
    draft_token_num,
    device,
):
    """Replace only the external NPU cache-loc op in CPU-only integration tests.

    Native cache_loc_update packs variable intervals into a maximum-width output;
    unwritten tail values must not become valid compact cache locations.
    """
    output = torch.full(
        (batch_size * draft_token_num,), -123, dtype=torch.int32, device=device
    )
    rows = [
        req_to_token[
            int(req_pool_indices[i]), int(start_offset[i]) : int(end_offset[i])
        ]
        for i in range(batch_size)
    ]
    packed = torch.cat(rows)
    output[: packed.numel()].copy_(packed)
    return output


def layout_for(lens):
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=lens, device=torch.device("cpu"), grid=[sum(lens)]
    )


def _backend_method(name="init_forward_metadata", **extra):
    # Execute the production method with CPU tensors while avoiding imports of
    # torch_npu / sgl_kernel_npu on CPU CI. Only the device boundary is replaced;
    # no copy of the metadata implementation is maintained in this test.
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    )
    tree = ast.parse(path.read_text())
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttnBackend"
    )
    method = copy.deepcopy(
        next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    method.decorator_list = []
    for arg in method.args.args:
        arg.annotation = None
    for arg in method.args.kwonlyargs:
        arg.annotation = None
    method.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(
        torch=torch,
        np=np,
        ForwardMetadata=lambda **fields: SimpleNamespace(
            **({"seq_lens_cpu_int": None, "block_tables": None} | fields)
        ),
        get_parallel=lambda: SimpleNamespace(dcp_enabled=False, attn_dp_enabled=False),
    )
    namespace.update(extra)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _bind_metadata_methods(backend):
    for name in (
        "_get_ragged_verify_layout",
        "_gather_block_tables",
        "_allocate_padded_verify_metadata",
        "_update_ragged_verify_metadata",
        "_init_cuda_graph_metadata",
        "_apply_cuda_graph_metadata",
    ):
        setattr(backend, name, MethodType(_backend_method(name), backend))


class TestNpuCompactMetadata(CustomTestCase):
    def test_dp_eager_backend_uses_sink_metadata_for_idle_and_busy(self):
        backend = SimpleNamespace(
            graph_metadata={},
            req_to_token=torch.arange(256).view(4, 64),
            page_size=8,
            device="cpu",
            _dspark_dsa_ragged_verify=True,
            ragged_verify_requires_cpu_seq_lens=False,
            ragged_verify_idle_seq_len=0,
        )
        _bind_metadata_methods(backend)
        init = _backend_method(
            get_parallel=lambda: SimpleNamespace(attn_dp_enabled=True)
        )
        for lens, prefixes in (([0], [0]), ([1, 3], [7, 15])):
            layout = RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=torch.tensor(lens), graph_num_tokens=8
            )
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=len(lens),
                seq_lens=torch.tensor(prefixes),
                req_pool_indices=torch.arange(len(lens)),
                spec_info=SimpleNamespace(ragged_verify_layout=layout),
            )
            init(backend, batch)
            metadata = backend.forward_metadata
            expected_q = torch.tensor(lens).cumsum(0).tolist() + [8]
            self.assertEqual(metadata.actual_seq_lengths_q.tolist(), expected_q)
            self.assertFalse(metadata.block_tables[-1].any())
            self.assertFalse(backend.graph_mode)
            self.assertFalse(backend.graph_metadata)
            if lens == [0]:
                self.assertFalse(metadata.block_tables.any())

    def test_compact_metadata_dispatch_is_glm_verify_only(self):
        layout = layout_for([1, 3])
        select = _backend_method("_get_ragged_verify_layout")
        for enabled, mode, request_layout in (
            (False, ForwardMode.TARGET_VERIFY, layout),
            (True, ForwardMode.EXTEND, layout),
            (True, ForwardMode.DECODE, layout),
            (True, ForwardMode.TARGET_VERIFY, None),
            (True, ForwardMode.TARGET_VERIFY, layout),
        ):
            backend = SimpleNamespace(_dspark_dsa_ragged_verify=enabled)
            batch = SimpleNamespace(
                forward_mode=mode,
                spec_info=SimpleNamespace(ragged_verify_layout=request_layout),
            )
            expected = request_layout if enabled and mode.is_target_verify() else None
            self.assertIs(select(backend, batch), expected)

    def test_eager_fallback_keeps_selected_graph_tier_without_dp(self):
        backend = SimpleNamespace(
            _dspark_dsa_ragged_verify=True,
            req_to_token=torch.arange(8, 264).view(4, 64),
            page_size=8,
            device="cpu",
        )
        _bind_metadata_methods(backend)
        # The planner has selected T=8, but the target may reject graph replay
        # later. The eager input still has eight rows, only four of them live.
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([1, 3]), graph_num_tokens=8
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            spec_info=SimpleNamespace(ragged_verify_layout=layout),
            seq_lens=torch.tensor([10, 20]),
            req_pool_indices=torch.tensor([2, 0]),
            input_ids=torch.zeros(8, dtype=torch.int64),
        )
        _backend_method()(backend, batch)
        metadata = backend.forward_metadata
        self.assertEqual(metadata.actual_seq_lengths_q.tolist(), [1, 4, 8])
        self.assertEqual(metadata.actual_seq_lengths_kv.tolist(), [11, 23, 4])
        self.assertEqual(
            metadata.block_tables.tolist(),
            [
                [17, 18, 0, 0, 0, 0, 0, 0],
                [1, 2, 3, 0, 0, 0, 0, 0],
                [0] * 8,
            ],
        )
        self.assertFalse(backend.graph_mode)
        # Outputs for sink queries must not leak into per-request acceptance.
        compact = torch.tensor([10, 20, 21, 22, 999, 999, 999, 999])[:, None]
        strided = scatter_compact_to_strided(
            compact=compact, layout=layout, fill_value=-1, verify_num_draft_tokens=4
        )
        self.assertEqual(strided.flatten().tolist(), [10, -1, -1, -1, 20, 21, 22, -1])
        self.assertEqual(batch.seq_lens.tolist(), [10, 20])

    def test_graph_capture_and_replay_refresh_token_tier_buffers(self):
        backend = SimpleNamespace(
            graph_metadata={},
            req_to_token=torch.arange(8, 8 + 4 * 64).view(4, 64),
            page_size=8,
            device="cpu",
            _dspark_dsa_ragged_verify=True,
            ragged_verify_requires_cpu_seq_lens=False,
            ragged_verify_idle_seq_len=0,
            needs_cpu_seq_lens=True,
        )
        _bind_metadata_methods(backend)
        init = _backend_method("init_forward_metadata_out_graph")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=3,
            req_pool_indices=torch.tensor([2, 1, 0]),
            seq_lens=torch.tensor([7, 15, 1]),
            out_cache_loc=None,
            spec_info=SimpleNamespace(),
        )

        def set_layout(lens, tokens):
            batch.spec_info.ragged_verify_layout = (
                RaggedVerifyLayout.from_verify_lens_device(
                    verify_lens=torch.tensor(lens), graph_num_tokens=tokens
                )
            )

        fields = ("actual_seq_lengths_q", "actual_seq_lengths_kv", "block_tables")
        set_layout([3, 3, 2], 8)
        init(backend, batch, in_capture=True)
        tier8 = backend.forward_metadata
        captured_ptrs = [getattr(tier8, field).data_ptr() for field in fields]
        set_layout([6, 5, 5], 16)
        init(backend, batch, in_capture=True)
        tier16 = backend.forward_metadata
        for field, ptr in zip(fields, captured_ptrs):
            self.assertNotEqual(getattr(tier16, field).data_ptr(), ptr)

        # A live batch can use fewer request rows than its captured tier.
        set_layout([1, 3], 8)
        with (
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host sync")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("host read")
            ),
        ):
            init(backend, batch)
        self.assertIs(backend.forward_metadata, tier8)
        self.assertEqual(
            [getattr(tier8, field).data_ptr() for field in fields], captured_ptrs
        )
        self.assertEqual(tier8.actual_seq_lengths_q.tolist(), [1, 4, 4, 8])
        self.assertEqual(tier8.actual_seq_lengths_kv.tolist(), [8, 18, 0, 4])
        self.assertEqual(tier8.block_tables[0].tolist(), [17, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(tier8.block_tables[1].tolist(), [9, 10, 11, 0, 0, 0, 0, 0])
        self.assertFalse(tier8.block_tables[2:].any())
        self.assertEqual(tier16.actual_seq_lengths_q.tolist(), [6, 11, 16, 16])

        set_layout([0], 8)
        init(backend, batch)
        self.assertIs(backend.forward_metadata, tier8)
        self.assertEqual(tier8.actual_seq_lengths_q.tolist(), [0, 0, 0, 8])
        self.assertEqual(tier8.actual_seq_lengths_kv.tolist(), [0, 0, 0, 8])
        self.assertFalse(tier8.block_tables.any())
        self.assertEqual(batch.seq_lens.tolist(), [7, 15, 1])

    def test_real_backend_metadata_does_not_double_extend_cpu_lengths(self):
        method = _backend_method()
        table = torch.arange(4 * 1024).view(4, 1024)
        backend = SimpleNamespace(
            graph_mode=False,
            _dspark_dsa_ragged_verify=True,
            ragged_verify_requires_cpu_seq_lens=False,
            ragged_verify_idle_seq_len=0,
            supports_ragged_verify_graph=True,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            req_to_token=table,
            page_size=128,
            is_hybrid_swa=False,
            use_mla=True,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        _bind_metadata_methods(backend)
        for lens in ([1, 3, 8], [8, 8, 8], [1, 1, 1]):
            with self.subTest(lens=lens):
                prefix = torch.tensor([127, 255, 513])
                expanded_cpu = prefix + torch.tensor(lens)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    batch_size=len(lens),
                    spec_info=SimpleNamespace(
                        ragged_verify_layout=layout_for(lens), draft_token_num=8
                    ),
                    spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
                    seq_lens=prefix,
                    seq_lens_cpu=expanded_cpu,
                    input_ids=torch.zeros(sum(lens), dtype=torch.int64),
                    global_num_token_non_padded_cpu=sum(lens),
                    req_pool_indices=torch.tensor([2, 0, 3]),
                    extend_seq_lens=None,
                )
                method(backend, batch)
                metadata = backend.forward_metadata
                self.assertEqual(
                    metadata.seq_lens.tolist(), expanded_cpu.tolist() + [1]
                )
                self.assertEqual(
                    metadata.actual_seq_lengths_kv.tolist(), expanded_cpu.tolist() + [1]
                )
                self.assertEqual(
                    metadata.actual_seq_lengths_q.tolist(),
                    torch.tensor(lens).cumsum(0).tolist() + [sum(lens)],
                )
                self.assertEqual(metadata.block_tables.shape, (4, 8))
                self.assertEqual(batch.seq_lens.tolist(), [127, 255, 513])
                self.assertEqual(batch.seq_lens_cpu.tolist(), expanded_cpu.tolist())

    def test_noncompact_metadata_preserves_target_branch_host_lengths(self):
        # Keep the merged release's DFlash-family host-length behavior for
        # batches without a ragged layout, including page-boundary sizing.
        method = _backend_method()
        backend = SimpleNamespace(
            supports_ragged_verify_graph=False,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(4 * 1024).view(4, 1024)
            ),
            page_size=128,
            is_hybrid_swa=False,
            use_mla=True,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        backend.req_to_token = backend.req_to_token_pool.req_to_token
        backend.dsa_dcp_target = False
        backend.use_paged_mtp = False
        backend.use_flash_mla = False
        backend.use_dsa = True
        backend.use_quant_flash_attn = False
        backend.dense_mla_dcp_target = False
        backend.needs_cpu_seq_lens = True
        backend._dspark_dsa_ragged_verify = False
        _bind_metadata_methods(backend)
        for algorithm, expected, pages in (
            (SpeculativeAlgorithm.DSPARK, [128, 256], 3),
            (SpeculativeAlgorithm.DFLASH, [128, 256], 2),
            (SpeculativeAlgorithm.EAGLE, [136, 264], 3),
        ):
            with self.subTest(algorithm=algorithm):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    spec_info=SimpleNamespace(draft_token_num=8),
                    spec_algorithm=algorithm,
                    seq_lens=torch.tensor([120, 248]),
                    seq_lens_cpu=torch.tensor([128, 256]),
                    req_pool_indices=torch.tensor([2, 0]),
                    extend_seq_lens=None,
                )
                method(backend, batch)
                metadata = backend.forward_metadata
                self.assertEqual(metadata.seq_lens_cpu_int.tolist(), expected)
                self.assertEqual(metadata.block_tables.shape, (2, pages))
                self.assertEqual(metadata.actual_seq_lengths_q.tolist(), [8, 16])
                self.assertEqual(batch.seq_lens_cpu.tolist(), [128, 256])


class TestNpuCompactExecution(CustomTestCase):
    def test_two_round_executor_handoff(self):
        # Run the production compact executor and accept/commit routing. Replace
        # ForwardBatch construction and the target model at their boundaries;
        # the Ascend metadata method, packing and acceptance execute on CPU.
        metadata_method = _backend_method()
        table = torch.arange(3 * 128, dtype=torch.int32).view(3, 128)
        backend = SimpleNamespace(
            graph_mode=False,
            _dspark_dsa_ragged_verify=True,
            ragged_verify_requires_cpu_seq_lens=False,
            ragged_verify_idle_seq_len=0,
            supports_ragged_verify_graph=True,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            req_to_token=table,
            page_size=16,
            is_hybrid_swa=False,
            use_mla=True,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        _bind_metadata_methods(backend)
        runner = SimpleNamespace(
            req_to_token_pool=backend.req_to_token_pool,
            attn_backend=backend,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
            ),
        )
        seen_lengths = []

        def prepare(verify_input, batch, target_worker):
            return (
                SimpleNamespace(
                    batch_size=batch.seq_lens.numel(),
                    input_ids=verify_input.draft_token,
                    positions=verify_input.positions,
                    spec_info=verify_input,
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    seq_lens=batch.seq_lens,
                    seq_lens_cpu=batch.seq_lens_cpu,
                    req_pool_indices=batch.req_pool_indices,
                    extend_seq_lens=None,
                    global_num_token_non_padded_cpu=verify_input.draft_token.numel(),
                    spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
                ),
                False,
            )

        def target_forward(*, batch, forward_batch, is_verify, skip_attn_backend_init):
            self.assertTrue(is_verify)
            self.assertIsNone(skip_attn_backend_init)
            metadata_method(backend, forward_batch)
            lens = forward_batch.spec_info.ragged_verify_layout.verify_lens
            self.assertEqual(
                backend.forward_metadata.actual_seq_lengths_kv.tolist(),
                (forward_batch.seq_lens + lens).tolist() + [1],
            )
            seen_lengths.append(
                backend.forward_metadata.actual_seq_lengths_q[:-1].tolist()
            )
            ids = forward_batch.input_ids
            logits = torch.full((ids.numel(), 32), -10.0)
            logits.scatter_(1, (ids + 1).unsqueeze(1), 10.0)
            return SimpleNamespace(
                can_run_cuda_graph=False,
                logits_output=SimpleNamespace(
                    next_token_logits=logits,
                    hidden_states=torch.stack(
                        [ids, forward_batch.positions], dim=1
                    ).float(),
                ),
            )

        injector = Mock()
        executor = dspark_verify.TargetVerifyExecutor(
            target_worker=SimpleNamespace(
                forward_batch_generation=target_forward, model_runner=runner
            ),
            gamma=3,
            verify_num_draft_tokens=4,
            model_runner=runner,
            kv_injector=injector,
            tp_sync=Mock(),
        )
        prefix, anchors = torch.tensor([15, 31]), torch.tensor([1, 5])
        for lengths in ([1, 3], [4, 1]):
            layout = layout_for(lengths)
            candidates = anchors[:, None] + torch.arange(4)[None, :]
            batch = SimpleNamespace(
                seq_lens=prefix,
                seq_lens_cpu=prefix.clone(),
                seq_lens_sum=int(prefix.sum()),
                req_pool_indices=torch.tensor([2, 0]),
            )
            original_cpu = batch.seq_lens_cpu
            with (
                get_parallel().override(attn_dp_enabled=False),
                patch.object(dspark_verify, "_is_npu", True),
                patch.object(
                    dspark_verify.DFlashVerifyInput, "prepare_for_verify", prepare
                ),
                patch(
                    "sglang.kernels.ops.speculative.dspark.dspark_verify_window.assign_extend_cache_locs_func",
                    side_effect=_cache_location_kernel_cpu_reference,
                ),
            ):
                result, hidden = executor.run_compact(
                    batch=batch,
                    layout=layout,
                    draft_block_ids=candidates,
                    draft_tokens=candidates[:, 1:],
                    bs=2,
                    device="cpu",
                    sampling_info=None,
                    verify_window=SimpleNamespace(kv_loc_plan=None, iteration_ids=None),
                )
            self.assertIs(batch.seq_lens_cpu, original_cpu)
            self.assertEqual(batch.seq_lens_sum, int(prefix.sum()))
            self.assertEqual(batch.out_cache_loc.numel(), sum(lengths))
            accepted = executor.accept_and_finalize(
                folded_accept=False,
                bs=2,
                verify_ids_2d=candidates,
                target_logits=result.logits_output.next_token_logits,
                draft_block=SimpleNamespace(
                    greedy_mask=torch.ones(2, dtype=torch.bool)
                ),
                sampling_info=None,
                draft_input=None,
                layout=layout,
                prefix_lens=prefix,
                draft_tokens=candidates[:, 1:],
            )
            self.assertEqual(accepted.commit_lens.tolist(), lengths)
            self.assertEqual(
                accepted.new_seq_lens.tolist(),
                (prefix + torch.tensor(lengths)).tolist(),
            )
            executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden,
                verify_window=SimpleNamespace(kv_loc_plan=None, iteration_ids=None),
                logits_output=result.logits_output,
                commit_lens=accepted.commit_lens,
                bs=2,
                run_compact=True,
            )
            committed = injector.inject_ragged.call_args.kwargs
            self.assertIs(committed["layout"], layout)
            self.assertIs(committed["commit_lens"], accepted.commit_lens)
            for row, length in enumerate(lengths):
                self.assertEqual(
                    hidden.view(2, 4, 2)[row, :length, 0].tolist(),
                    candidates[row, :length].float().tolist(),
                )
                self.assertEqual(
                    accepted.out_tokens[row, :length].tolist(),
                    list(range(int(anchors[row]) + 1, int(anchors[row]) + length + 1)),
                )
            prefix, anchors = accepted.new_seq_lens, accepted.bonus
        self.assertEqual(seen_lengths, [[1, 4], [4, 5]])
        self.assertEqual(prefix.tolist(), [20, 35])
        self.assertEqual(anchors.tolist(), [6, 9])
        self.assertEqual(injector.inject_ragged.call_count, 2)

    def test_sparse_operator_receives_compact_boundaries(self):
        op = Mock(return_value=(torch.zeros(4, 1, 4), None, None))
        method = _backend_method(
            "forward_sparse",
            torch_npu=SimpleNamespace(npu_sparse_flash_attention=op),
            _expand_dsa_sparse_indices=lambda x: x.unsqueeze(-2),
        )
        metadata = SimpleNamespace(
            actual_seq_lengths_q=torch.tensor([1, 4], dtype=torch.int32),
            actual_seq_lengths_kv=torch.tensor([16, 34], dtype=torch.int32),
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
        )
        backend = SimpleNamespace(
            dsa_dcp_target=False,
            forward_metadata=metadata,
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=SimpleNamespace(
                get_kv_buffer=lambda _: (torch.zeros(4, 4), torch.zeros(4, 2))
            ),
            _pad_topk_indices=lambda x, _: x,
        )
        method(
            backend,
            torch.zeros(4, 1, 4),
            None,
            None,
            SimpleNamespace(layer_id=0, scaling=0.5),
            SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
            save_kv_cache=False,
            q_rope=torch.zeros(4, 1, 2),
            topk_indices=torch.zeros(4, 2, dtype=torch.int32),
        )
        args = op.call_args.kwargs
        self.assertEqual(args["actual_seq_lengths_query"].tolist(), [1, 4])
        self.assertEqual(args["actual_seq_lengths_kv"].tolist(), [16, 34])
        self.assertEqual(args["sparse_mode"], 3)
        self.assertEqual(args["sparse_indices"].shape, (4, 1, 2))


if __name__ == "__main__":
    unittest.main()
