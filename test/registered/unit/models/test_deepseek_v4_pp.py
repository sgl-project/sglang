import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner import flashinfer_autotune
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepSeekV4PP(unittest.TestCase):
    def test_pp_skips_isolated_flashinfer_autotune(self):
        runner = SimpleNamespace(
            device="cuda",
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41")
            ),
        )
        with (
            patch.object(
                flashinfer_autotune,
                "get_parallel",
                return_value=SimpleNamespace(pp_size=4, enable_kv_cache_sharding=False),
            ),
            patch.object(
                flashinfer_autotune,
                "_kernels_take_flashinfer_tactics",
                return_value=True,
            ),
            patch.object(
                flashinfer_autotune,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(disable_flashinfer_autotune=False),
                    deterministic=SimpleNamespace(enable_deterministic_inference=False),
                ),
            ),
        ):
            self.assertFalse(flashinfer_autotune.should_run_flashinfer_autotune(runner))

    def test_full_page_ids_follow_each_requests_local_mapping(self):
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 0]), seq_lens_cpu=[5, 9]
        )
        with (
            patch.object(
                deepseek_v4,
                "get_req_to_token_pool",
                return_value=SimpleNamespace(
                    req_to_token=torch.arange(4, 36).reshape(2, 16)
                ),
            ),
            patch.object(
                deepseek_v4,
                "get_token_to_kv_pool",
                return_value=SimpleNamespace(page_size=4),
            ),
        ):
            pages = deepseek_v4.DeepseekV4Model._pp_full_page_ids(None, batch)
        torch.testing.assert_close(pages, torch.tensor([5, 6, 1, 2, 3]))

    def test_pp_gate_allows_only_eager_pd_prefill(self):
        cfg = SimpleNamespace(
            disaggregation_mode="prefill",
            language_model_only=True,
            enable_prefill_cp=False,
            attn_cp_size=1,
            dcp_size=1,
            attn_dp_size=1,
            enable_dp_attention=False,
            tp_size=1,
            ep_join_mode=None,
            speculative_algorithm=None,
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=False,
            disable_radix_cache=True,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED)
            ),
        )
        with patch.object(
            deepseek_v4_hook,
            "get_platform",
            return_value=SimpleNamespace(is_cuda=True),
        ):
            deepseek_v4_hook.validate_deepseek_v41_pp(cfg)
            cfg.disaggregation_mode = "decode"
            with self.assertRaisesRegex(ValueError, "PD prefill"):
                deepseek_v4_hook.validate_deepseek_v41_pp(cfg)

    def test_handoff_preserves_logical_picks_and_remaps_physical_slots(self):
        raw = torch.tensor([[0, 3, -1], [1, 2, -1]], dtype=torch.int32)
        core = SimpleNamespace(
            c1_sparse_raw_indices=raw,
            c2_sparse_raw_indices=None,
            c1_sparse_topk_lengths=torch.tensor([2, 2]),
            page_table=torch.tensor([[7], [9]], dtype=torch.int32),
            c1_sparse_page_indices=torch.zeros_like(raw),
        )
        core.sparse_raw_indices = lambda ratio: getattr(
            core, f"c{ratio}_sparse_raw_indices"
        )
        core.sparse_page_indices = lambda ratio: getattr(
            core, f"c{ratio}_sparse_page_indices"
        )
        blocks = torch.tensor([[0], [0]], dtype=torch.int32)
        metadata = SimpleNamespace(
            core_metadata=core,
            candidate_metadata=BlockIds(blocks, [1, 1]),
        )
        backend = SimpleNamespace(forward_metadata=metadata)
        pool = SimpleNamespace(page_size=4)
        model = SimpleNamespace(
            end_layer=25,
            config=SimpleNamespace(
                compress_ratios=[0] * 20 + [1] * 20,
                kv_source_layer_ids=[20],
                index_source_layer_ids=[20, 24, 28],
                candidate_source_layer_id=20,
            ),
            _pp_full_page_ids=lambda batch: torch.tensor([1, 2]),
        )
        pool.export_source_pages = lambda source, pages: {"kv": torch.tensor([7])}
        installed = []
        pool.install_source_pages = lambda *args: installed.append(args)
        batch = SimpleNamespace(extend_seq_lens_cpu=[1, 1])
        tensors = {}
        with (
            patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            patch.object(deepseek_v4, "get_token_to_kv_pool", return_value=pool),
        ):
            deepseek_v4.DeepseekV4Model._export_pp_state(model, tensors, batch)
            core.c1_sparse_raw_indices = torch.zeros_like(raw)
            metadata.candidate_metadata = None
            deepseek_v4.DeepseekV4Model._install_pp_state(
                model, PPProxyTensors(tensors), batch
            )
        torch.testing.assert_close(core.c1_sparse_raw_indices, raw)
        torch.testing.assert_close(
            core.c1_sparse_page_indices,
            torch.tensor([[28, 31, -1], [37, 38, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(metadata.candidate_metadata.blocks, blocks)
        self.assertEqual(installed[0][0], 20)


if __name__ == "__main__":
    unittest.main()
