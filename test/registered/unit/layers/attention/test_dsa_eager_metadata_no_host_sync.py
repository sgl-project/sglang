import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import dsa_backend as module
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.model_executor import cuda_graph_config
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsaEagerMetadataNoHostSync(CustomTestCase):
    def check_metadata(
        self,
        mode,
        host_mirror,
        *,
        flashmla=False,
        packing=False,
        graph=False,
        tbo=False,
        custom=False,
    ):
        table = torch.arange(2 * 32, dtype=torch.int32).reshape(2, 32)
        seq_lens = torch.tensor([5, 7])
        prefix = torch.tensor([2, 4])
        batch = SimpleNamespace(
            batch_size=2,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone() if host_mirror else None,
            req_pool_indices=torch.tensor([0, 1]),
            forward_mode=mode,
            extend_seq_lens_cpu=[3, 3],
            extend_seq_lens=torch.tensor([3, 3]),
            extend_prefix_lens_cpu=[2, 4] if host_mirror else None,
            extend_prefix_lens=prefix,
            extend_num_tokens=6,
            seq_lens_sum=12 if host_mirror else 0,
        )
        backend = SimpleNamespace(
            speculative_num_draft_tokens=3,
            req_to_token=table,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            _kv_shard_pool=None,
            set_dsa_prefill_impl=lambda _: None,
            dsa_decode_impl="flashmla_kv" if flashmla else "triton",
            dsa_prefill_impl="flashmla_kv" if flashmla else "triton",
            flashmla_prefill_token_packing=packing,
            requires_eager_metadata=custom,
            flashmla_kv_num_q_heads_prefill=64,
            _compute_flashmla_metadata=Mock(
                return_value=module.DSAFlashMLAMetadata(
                    torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
                )
            ),
            use_mha=False,
            get_topk_transform_method=lambda _: TopkTransformMethod.PAGED,
            dsa_index_kpool=1,
            dsa_shards_index_k=False,
            dsa_index_topk=4,
            physical_page_size=64 if flashmla else 1,
            get_device_int32_arange=lambda n: torch.arange(n, dtype=torch.int32),
            _cal_indexer_k_start_end=lambda *args: (None, None),
            _transform_table_1_to_real=lambda table: table,
            _build_topk_v2_plan=lambda _: None,
            _dcp_topk_length_buffer=lambda *args: None,
            init_kernel_metadata=lambda _: None,
            _init_kpool_metadata=lambda metadata, *args, **kwargs: metadata,
            # A static pool: no id translation, so the gather reads the table.
            runner_ids=None,
        )
        for name in ("_read_v2p", "_gather_token_table"):
            method = getattr(module.DeepseekSparseAttnBackend, name)
            setattr(backend, name, MethodType(method, backend))

        def cumulative(lengths):
            return torch.cat(
                (torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32))
            )

        exec_config = SimpleNamespace(
            overlap=SimpleNamespace(enable_two_batch_overlap=tbo),
            graph=SimpleNamespace(
                cuda_graph_config=cuda_graph_config.CudaGraphConfig(
                    prefill=cuda_graph_config.PhaseConfig(
                        backend=(
                            cuda_graph_config.Backend.BREAKABLE
                            if graph
                            else cuda_graph_config.Backend.DISABLED
                        )
                    )
                )
            ),
        )
        with (
            patch.object(
                seq_lens, "max", side_effect=AssertionError("device max/item sync")
            ),
            patch.object(
                seq_lens, "cpu", side_effect=AssertionError("device seq_lens copy")
            ),
            patch.object(
                prefix, "cpu", side_effect=AssertionError("device prefix copy")
            ),
            patch.object(module, "compute_cu_seqlens", side_effect=cumulative),
            patch.object(
                module,
                "seqlens_expand_triton",
                side_effect=lambda extend, lens, *args: lens.repeat_interleave(extend),
            ),
            patch.object(
                module,
                "compute_dsa_seqlens",
                side_effect=lambda original_seq_lens, **kwargs: original_seq_lens.clamp(
                    max=4
                ),
            ),
            patch.object(
                module, "pad_dsa_cache_seqlens", side_effect=lambda batch, lens: lens
            ),
            patch.object(module, "is_cuda", return_value=False),
            patch.object(module, "can_dsa_prefill_cp_interleave", return_value=False),
            patch.object(module, "get_exec", return_value=exec_config),
            patch.object(cuda_graph_config, "get_exec", return_value=exec_config),
        ):
            module.DeepseekSparseAttnBackend.init_forward_metadata(backend, batch)
        width = 7 + (3 if mode.is_target_verify() else 0) if host_mirror else 32
        repeats = 3 if mode.is_draft_extend_v2() or mode.is_target_verify() else 1
        self.assertEqual(backend.forward_metadata.max_seq_len_k, width)
        torch.testing.assert_close(
            backend.forward_metadata.page_table_1,
            table[:, :width].repeat_interleave(repeats, dim=0),
        )
        if not host_mirror:
            self.assertIsNone(batch.seq_lens_cpu)
            self.assertIsNone(batch.extend_prefix_lens_cpu)
            self.assertEqual(batch.seq_lens_sum, 0)
        return backend, batch

    def test_decode_without_host_mirror(self):
        self.check_metadata(ForwardMode.DECODE, False)

    def test_draft_extend_without_host_mirror(self):
        self.check_metadata(ForwardMode.DRAFT_EXTEND_V2, False)

    def test_existing_host_mirror_keeps_narrow_table(self):
        for mode in (ForwardMode.DECODE, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                self.check_metadata(mode, True)

    def test_packed_prefill_defers_and_reuses_unpacked_schedule(self):
        """A packed layer needs no legacy schedule; rejected layers share one."""
        backend, batch = self.check_metadata(
            ForwardMode.EXTEND, True, flashmla=True, packing=True
        )
        initial_metadata = backend.forward_metadata
        self.assertIsNone(initial_metadata.flashmla_metadata)
        compute = backend._compute_flashmla_metadata
        compute.assert_not_called()

        cache = torch.empty(1, 64, 1, 656, dtype=torch.uint8)
        backend.kv_layers = SimpleNamespace(
            read=lambda _: SimpleNamespace(key=lambda: cache)
        )
        backend.kv_cache_dim = 656
        backend.use_fused_topk = True
        backend.hisparse_coordinator = None
        backend._resolve_kpool_tail_backend = lambda _, impl: impl
        backend._check_kpool_tail_backend = lambda *args: None
        backend._pad_topk_indices = lambda indices, _: indices
        backend._get_fused_topk_page_table = lambda indices: indices
        backend._flashmla_kv_q = lambda q, rope: torch.cat((q, rope), dim=-1)
        packed_out = torch.ones(6, 1, 12, 512, dtype=torch.bfloat16)
        fallback_out = torch.zeros_like(packed_out)
        pack = Mock(side_effect=[packed_out, None, None])
        backend._forward_flashmla_kv = Mock(return_value=fallback_out)
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=12,
            v_head_dim=512,
            head_dim=576,
            scaling=0.5,
        )
        q = torch.zeros(6, 12, 512, dtype=torch.bfloat16)
        rope = torch.zeros(6, 12, 64, dtype=torch.bfloat16)
        indices = torch.zeros(6, 4, dtype=torch.int32)

        def forward():
            return module.DeepseekSparseAttnBackend.forward_extend(
                backend,
                q,
                None,
                None,
                layer,
                batch,
                False,
                q_rope=rope,
                topk_indices=indices,
            )

        with patch.dict(
            "sys.modules",
            {
                "sglang.kernels.ops.attention.dsa.flashmla_token_pack": SimpleNamespace(
                    flashmla_kv_token_packed=pack
                )
            },
        ):
            self.assertIs(forward(), packed_out)
            self.assertIs(backend.forward_metadata, initial_metadata)
            compute.assert_not_called()
            self.assertIs(forward(), fallback_out)
            fallback_metadata = backend.forward_metadata
            self.assertIsNotNone(fallback_metadata.flashmla_metadata)
            self.assertIs(
                fallback_metadata.dsa_cache_seqlens_int32,
                initial_metadata.dsa_cache_seqlens_int32,
            )
            compute.assert_called_once_with(
                cache_seqlens=initial_metadata.dsa_cache_seqlens_int32,
                seq_len_q=1,
                num_heads_q=64,
            )
            self.assertIs(forward(), fallback_out)
            self.assertIs(backend.forward_metadata, fallback_metadata)
            self.assertIs(
                backend._forward_flashmla_kv.call_args.kwargs["metadata"],
                fallback_metadata,
            )
            self.assertEqual(compute.call_count, 1)

    def test_unpacked_schedule_stays_eager_outside_packed_eager_extend(self):
        for mode, options in (
            (ForwardMode.EXTEND, {"packing": False}),
            (ForwardMode.EXTEND, {"graph": True}),
            (ForwardMode.EXTEND, {"tbo": True}),
            (ForwardMode.EXTEND, {"custom": True}),
            (ForwardMode.MIXED, {}),
            (ForwardMode.DECODE, {}),
            (ForwardMode.TARGET_VERIFY, {}),
            (ForwardMode.DRAFT_EXTEND_V2, {}),
        ):
            with self.subTest(mode=mode, options=options):
                backend, _ = self.check_metadata(
                    mode, True, flashmla=True, **({"packing": True} | options)
                )
                self.assertIsNotNone(backend.forward_metadata.flashmla_metadata)


if __name__ == "__main__":
    unittest.main()
