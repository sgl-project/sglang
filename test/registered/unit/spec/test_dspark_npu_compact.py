"""CPU contracts for the NPU compact port; NPU operator execution is separate."""

import ast
import copy
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    build_ragged_verify_window,
    scatter_compact_to_strided,
)
from sglang.srt.hardware_backend.npu.attention import dspark_compact
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import is_npu
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
    path = Path(dspark_compact.__file__).with_name("ascend_backend.py")
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
        ForwardMetadata=lambda: SimpleNamespace(seq_lens_cpu_int=None),
        build_compact_verify_metadata=dspark_compact.build_compact_verify_metadata,
        get_parallel=lambda: SimpleNamespace(dcp_enabled=False, attn_dp_enabled=False),
    )
    namespace.update(extra)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class TestNpuCompactMetadata(CustomTestCase):
    def test_dp_eager_backend_uses_sink_metadata_for_idle_and_busy(self):
        backend = SimpleNamespace(
            graph_metadata={},
            req_to_token=torch.arange(256).view(4, 64),
            page_size=8,
            device="cpu",
            _dspark_dsa_ragged_verify=True,
        )
        backend._glm_compact_layout = MethodType(
            _backend_method("_glm_compact_layout"), backend
        )
        backend._init_compact_graph_metadata = MethodType(
            _backend_method(
                "_init_compact_graph_metadata",
                NpuCompactGraphMetadata=dspark_compact.NpuCompactGraphMetadata,
            ),
            backend,
        )
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
            if lens == [0]:
                self.assertFalse(metadata.block_tables.any())

    def test_compact_metadata_dispatch_is_glm_verify_only(self):
        layout = layout_for([1, 3])
        select = _backend_method("_glm_compact_layout")
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

    def test_mixed_windows_and_page_boundary(self):
        prefix = torch.tensor([127, 255, 513])
        layout = layout_for([1, 3, 8])
        result = dspark_compact.build_compact_verify_metadata(
            prefix_lens=prefix, layout=layout, max_verify_len=8, num_tokens=12
        )
        self.assertEqual(result.query_ends.tolist(), [1, 4, 12])
        self.assertEqual(result.kv_lens.tolist(), [128, 258, 521])
        self.assertEqual(result.max_kv_len, 521)
        self.assertEqual(prefix.tolist(), [127, 255, 513])

    def test_device_only_layout(self):
        layout = layout_for([2, 1])
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=layout.verify_lens, graph_num_tokens=3
        )
        result = dspark_compact.build_compact_verify_metadata(
            prefix_lens=torch.tensor([8, 16]),
            layout=layout,
            max_verify_len=4,
            num_tokens=3,
        )
        self.assertEqual(result.query_ends.tolist(), [2, 3])

    def test_reject_invalid_layout(self):
        for lens, prefixes, token_count, max_len in (
            ([2], [10, 20], 2, 4),
            ([5], [10], 5, 4),
            ([1, 3], [10, 20], 8, 4),
        ):
            with self.subTest(lens=lens, token_count=token_count):
                with self.assertRaises(ValueError):
                    dspark_compact.build_compact_verify_metadata(
                        prefix_lens=torch.tensor(prefixes),
                        layout=layout_for(lens),
                        max_verify_len=max_len,
                        num_tokens=token_count,
                    )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1, 3], device=torch.device("cpu"), grid=[8]
        )
        with self.assertRaisesRegex(ValueError, "padded"):
            dspark_compact.build_compact_verify_metadata(
                prefix_lens=torch.tensor([10, 20]),
                layout=layout,
                max_verify_len=4,
                num_tokens=8,
            )

    def test_real_backend_metadata_does_not_double_extend_cpu_lengths(self):
        method = _backend_method()
        table = torch.arange(4 * 1024).view(4, 1024)
        backend = SimpleNamespace(
            graph_mode=False,
            _dspark_dsa_ragged_verify=True,
            supports_ragged_verify_graph=True,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            req_to_token=table,
            page_size=128,
            is_hybrid_swa=False,
            use_mla=True,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        backend._glm_compact_layout = MethodType(
            _backend_method("_glm_compact_layout"), backend
        )
        for lens in ([1, 3, 8], [8, 8, 8], [1, 1, 1]):
            with self.subTest(lens=lens):
                prefix = torch.tensor([127, 255, 513])
                expanded_cpu = prefix + torch.tensor(lens)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
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
                self.assertEqual(metadata.seq_lens.tolist(), expanded_cpu.tolist())
                self.assertEqual(
                    metadata.actual_seq_lengths_kv.tolist(), expanded_cpu.tolist()
                )
                self.assertEqual(
                    metadata.actual_seq_lengths_q.tolist(),
                    torch.tensor(lens).cumsum(0).tolist(),
                )
                self.assertEqual(metadata.block_tables.shape, (3, 5))
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
        backend.dsa_dcp_target = False
        backend.use_paged_mtp = False
        backend.use_flash_mla = False
        backend.use_dsa = True
        backend.use_quant_flash_attn = False
        backend.dense_mla_dcp_target = False
        backend.needs_cpu_seq_lens = True
        backend._dspark_dsa_ragged_verify = False
        backend._glm_compact_layout = MethodType(
            _backend_method("_glm_compact_layout"), backend
        )
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
            supports_ragged_verify_graph=True,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            req_to_token=table,
            page_size=16,
            is_hybrid_swa=False,
            use_mla=True,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        backend._glm_compact_layout = MethodType(
            _backend_method("_glm_compact_layout"), backend
        )
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
                (forward_batch.seq_lens + lens).tolist(),
            )
            seen_lengths.append(backend.forward_metadata.actual_seq_lengths_q.tolist())
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

    @unittest.skipUnless(is_npu(), "requires an Ascend NPU")
    def test_npu_tensor_smoke(self):
        # Run explicitly on an NPU in addition to CPU CI. This exercises host
        # scheduling transfer, searchsorted/gather/scatter and metadata copies.
        cfg = DSparkScheduleConfig(gamma=3)
        lens = ScheduleVerifyLensTopk.execute(
            confidence=torch.tensor([[0.9, 0.8, 0.7], [0.2, 0.1, 0.1]], device="npu"),
            budget=2,
            cfg=cfg,
        )
        self.assertEqual(lens.cpu().tolist(), [3, 1])
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[3, 1], device=torch.device("npu"), grid=[4]
        )
        prefix = torch.tensor([15, 31], device="npu")
        table = torch.arange(3 * 64, dtype=torch.int32, device="npu").view(3, 64)
        candidates = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="npu")
        packed = build_ragged_verify_window(
            batch=SimpleNamespace(
                seq_lens=prefix, req_pool_indices=torch.tensor([2, 0], device="npu")
            ),
            layout=layout,
            draft_block_ids=candidates,
            draft_tokens=candidates[:, 1:],
            bs=2,
            device="npu",
            verify_num_draft_tokens=4,
            model_runner=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(req_to_token=table)
            ),
        )
        self.assertEqual(packed.verify_ids.cpu().tolist(), [1, 2, 3, 5])
        self.assertEqual(packed.verify_cache_loc.cpu().tolist(), [143, 144, 145, 31])
        result = dspark_compact.build_compact_verify_metadata(
            prefix_lens=prefix, layout=layout, max_verify_len=4, num_tokens=4
        )
        self.assertEqual(result.kv_lens.tolist(), [18, 32])
        rows = scatter_compact_to_strided(
            compact=packed.verify_ids[:, None],
            layout=layout,
            fill_value=-1,
            verify_num_draft_tokens=4,
        )
        self.assertEqual(rows.cpu().flatten().tolist(), [1, 2, 3, -1, 5, -1, -1, -1])

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
