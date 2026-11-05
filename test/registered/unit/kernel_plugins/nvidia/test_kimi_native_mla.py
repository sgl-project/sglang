# SPDX-License-Identifier: Apache-2.0
"""CPU adapter checks; kernels and communication are mocked."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.kernel_plugins.nvidia import binding as mb
from sglang.srt.kernel_plugins.nvidia.kimi_mla import native_mla_decode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestKimiNativeMla(unittest.TestCase):
    def setUp(self):
        logits = ModuleType("sglang.srt.layers.logits_processor")
        self.autotune_dummy = Mock(return_value=False)
        logits.get_in_autotune_dummy_run = self.autotune_dummy
        previous = sys.modules.get(logits.__name__)
        sys.modules[logits.__name__] = logits

        def restore():
            # Restore this one fake, without unloading real torch modules that
            # lazily registered C++ operators while an adapter test was running.
            if previous is None:
                sys.modules.pop(logits.__name__, None)
            else:
                sys.modules[logits.__name__] = previous

        self.addCleanup(restore)

    def test_native_dcp_autotune_dummy_skips_attention_collectives(self):
        parallel, query, output, _, kw = self.fixture(8)
        self.autotune_dummy.return_value = True
        output.fill_(123)
        with patch("sglang.srt.runtime_context.get_parallel", return_value=parallel):
            native_mla_decode(query, output, **kw)
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        kw["backend"]._run_decode_kernel.assert_not_called()

    def fixture(self, dcp=1):
        parallel = SimpleNamespace(
            dcp_enabled=dcp > 1,
            attn_dcp_size=dcp,
            dcp_comm_backend="a2a",
            dcp_group=object(),
        )
        query = (
            torch.arange(2 * 12 * 576, dtype=torch.int64)
            .to(torch.uint8)
            .view(2, 12, 576)
        )
        output = torch.empty(2, 12, 512, dtype=torch.bfloat16)
        result = torch.full((2, 1, 12, 512), 3.0, dtype=torch.bfloat16)
        backend = SimpleNamespace(_run_decode_kernel=Mock(return_value=result))
        attention = SimpleNamespace(attn_mqa=object(), attn_mqa_for_dcp_decode=object())
        metadata = SimpleNamespace(
            block_kv_indices=torch.zeros(2, 2048, dtype=torch.int32),
            seq_lens_k=torch.tensor([0, 131072], dtype=torch.int32),
            max_seq_len_k=131072 if dcp > 1 else 1048576,
        )
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
        kwargs = dict(
            backend=backend,
            cache=torch.zeros(1, 64, 576, dtype=torch.uint8),
            attention=attention,
            metadata=metadata,
            forward_batch=batch,
        )
        return parallel, query, output, result, kwargs

    def test_native_decode_keeps_query_bytes_and_exact_metadata(self):
        parallel, query, output, result, kw = self.fixture()
        with patch("sglang.srt.runtime_context.get_parallel", return_value=parallel):
            native_mla_decode(query, output, **kw)
        torch.testing.assert_close(output, result.view_as(output))
        call = kw["backend"]._run_decode_kernel.call_args.kwargs
        self.assertTrue(torch.equal(call["query"].view(torch.uint8).squeeze(1), query))
        self.assertIs(call["block_tables"], kw["metadata"].block_kv_indices)
        self.assertIs(call["seq_lens"], kw["metadata"].seq_lens_k)
        self.assertEqual(call["max_seq_len"], 1048576)
        self.assertNotIn("return_lse", call)

    def test_dcp_reuses_native_neutralization_and_log2_merge(self):
        for comm in ("a2a", "fi_a2a", "allgather"):
            with self.subTest(comm=comm):
                parallel, query, output, _, kw = self.fixture(8)
                parallel.dcp_comm_backend = comm
                gathered = torch.cat([query + rank for rank in range(8)], dim=1)
                partial = torch.zeros(2, 96 * 512, dtype=torch.bfloat16)
                lse = torch.zeros(2, 96)
                kw["backend"]._forward_decode_dcp = Mock(return_value=(partial, lse))
                expected = (
                    torch.arange(2 * 12 * 512, dtype=torch.float32)
                    .to(torch.bfloat16)
                    .view(2, 12, 512)
                )
                dcp_module = ModuleType("sglang.srt.layers.dcp")
                dcp_module.all_gather_q_for_mla_decode = Mock(
                    return_value=(gathered[..., :512], gathered[..., 512:])
                )
                dcp_module.dcp_a2a_lse_reduce = Mock(return_value=expected)
                dcp_module.cp_lse_ag_out_rs_mla = Mock(
                    return_value=expected.transpose(0, 1)
                )
                with (
                    patch(
                        "sglang.srt.runtime_context.get_parallel", return_value=parallel
                    ),
                    patch.dict(sys.modules, {dcp_module.__name__: dcp_module}),
                ):
                    native_mla_decode(query, output, **kw)
                torch.testing.assert_close(output, expected)
                kw["backend"]._run_decode_kernel.assert_not_called()
                args = kw["backend"]._forward_decode_dcp.call_args.args
                self.assertEqual(args[0].shape, (2, 1, 96, 576))
                self.assertTrue(
                    torch.equal(args[0].view(torch.uint8).squeeze(1), gathered)
                )
                self.assertIs(args[2], kw["metadata"])
                self.assertIs(args[3], kw["attention"].attn_mqa_for_dcp_decode)
                merge = (
                    dcp_module.cp_lse_ag_out_rs_mla
                    if comm == "allgather"
                    else dcp_module.dcp_a2a_lse_reduce
                )
                self.assertFalse(merge.call_args.kwargs["is_lse_base_on_e"])

    def test_step_keeps_global_lengths_and_live_slots_and_calls_restore_barriers(self):
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        mla = mb.MlaPoolView(
            reserved_capacity=128,
            serving_capacity=lambda: 128,
            post_capture_active=False,
            page_size=64,
            logical_dtype=torch.float8_e4m3fn,
            storage_dtype=torch.uint8,
            layer_mapping={1: 0},
            caches={},
            write_shard_size=8,
            write_shard_rank=3,
        )
        conv, temporal = torch.zeros(3), torch.zeros(4)
        state = mb.KdaStateView(conv=(conv,), temporal=temporal)
        binding._preparation = mb.PreparationView(
            hybrid=mb.HybridPoolView(
                mla=mla,
                kda=mb.KdaPoolView(capacity=4, layer_mapping={2: 0}, states={2: state}),
            )
        )
        metadata = SimpleNamespace(
            block_kv_indices=torch.zeros(2, 4, dtype=torch.int32),
            seq_lens_k=torch.tensor([0, 1], dtype=torch.int32),
            global_seq_lens_k=torch.tensor([1, 8], dtype=torch.int32),
        )
        token_pool = SimpleNamespace(get_key_buffer=Mock())
        request_pool = SimpleNamespace()
        linear = SimpleNamespace(
            req_to_token_pool=request_pool,
            state_layer_cache=Mock(return_value=state),
            forward_metadata=SimpleNamespace(
                mamba_cache_indices=torch.tensor([3, 1], dtype=torch.int32)
            ),
            _track_mamba_state_decode=Mock(),
        )
        backend = object()
        sources = SimpleNamespace(
            backend=backend,
            token_pool=token_pool,
            request_pool=request_pool,
            linear_backend=linear,
            full_backend=SimpleNamespace(forward_decode_metadata=metadata),
        )
        sources.mla_backend = sources.full_backend
        batch = SimpleNamespace(
            out_cache_loc=torch.tensor([11, 19]),
            out_cache_loc_is_physical=True,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        )
        with patch.object(binding, "_current_hybrid_sources", return_value=sources):
            step = binding._hybrid_step(backend, batch)
            step.prepare_layer(1)
            step.prepare_layer(2)
            step.kda.track_decode_state(2)
            self.assertIs(step.mla.global_sequence_lengths, metadata.global_seq_lens_k)
            self.assertIs(step.mla.sequence_lengths, metadata.seq_lens_k)
            self.assertIs(step.mla.write_slots, batch.out_cache_loc)
            self.assertIs(
                step.kda.state_indices, linear.forward_metadata.mamba_cache_indices
            )
            token_pool.get_key_buffer.assert_called_once_with(1)
            linear.state_layer_cache.assert_called_once_with(2)
            self.assertIs(
                linear._track_mamba_state_decode.call_args.args[3],
                step.kda.state_indices,
            )
            linear.forward_metadata.mamba_cache_indices = torch.tensor(
                [1, 2], dtype=torch.int32
            )
            self.assertIs(
                binding._hybrid_step(backend, batch).kda.state_indices,
                linear.forward_metadata.mamba_cache_indices,
            )
            metadata.global_seq_lens_k = None
            with self.assertRaisesRegex(RuntimeError, "global lengths"):
                binding._hybrid_step(backend, batch)


if __name__ == "__main__":
    unittest.main()
