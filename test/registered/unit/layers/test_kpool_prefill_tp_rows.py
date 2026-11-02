"""K-pool prefill row sharding on real TP ranks, compared with the replicated path."""

import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as indexer_module
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa.kpool_plan import init_kpool_extend_metadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=60, stage="extra-b", runner_config="4-gpu-b200")

WORLD = 4


@dataclass
class _Metadata:
    topk_indices_offset: torch.Tensor
    kpool_extend_plan: object = None


def _run_rank(rank, init_method):
    torch.cuda.set_device(rank)
    init_distributed_environment(WORLD, rank, init_method, local_rank=rank)
    publish_build_topology(tp_size=WORLD, world_rank=rank)
    initialize_model_parallel()
    device = torch.device(f"cuda:{rank}")

    # 12289 rows behind a cached prefix do not divide evenly across TP ranks,
    # and q_fp8 has padding rows past them.
    prefix, length, pad_rows = 20480, 32769, 5
    n_real, pages = length - prefix, length // 256
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        req_pool_indices=torch.tensor([1], device=device),
        seq_lens=torch.tensor([length], device=device),
        seq_lens_cpu=torch.tensor([length]),
        extend_seq_lens_cpu=[n_real],
        attn_cp_metadata=None,
    )
    metadata = SimpleNamespace(
        attn_metadata=init_kpool_extend_metadata(
            _Metadata(torch.zeros(n_real, device=device, dtype=torch.int32)),
            batch,
            kpool=4,
            physical_page_size=64,
            index_page_size=64,
            topk_transform_method=TopkTransformMethod.RAGGED,
            full_real_page_table=torch.arange(
                pages * 4 + 4, device=device, dtype=torch.int32
            )[None],
            full_seqlens_expanded=torch.arange(
                prefix + 1, length + 1, device=device, dtype=torch.int32
            ),
        ),
        topk_transform_method=TopkTransformMethod.RAGGED,
    )
    # The same seed on every rank, because attention TP replicates inputs.
    torch.manual_seed(7)
    keys = torch.randn((pages + 1, 64, 128)).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = (torch.rand((pages + 1, 64)) + 0.5).view(torch.uint8)
    buf = torch.cat([keys.flatten(1), scales.flatten(1)], dim=1).to(device)
    q_fp8 = torch.randn((n_real + pad_rows, 32, 128)).to(device, torch.float8_e4m3fn)
    weights = torch.rand((n_real + pad_rows, 32, 1)).to(device)
    pool = SimpleNamespace(
        index_page_size=64, get_index_k_with_scale_buffer=lambda layer_id: buf
    )
    indexer = object.__new__(indexer_module.IndexerKPool)
    indexer.index_kpool, indexer.index_topk = 4, 2048

    def topk(enabled, q_rows):
        with envs.SGLANG_OPT_USE_DSA_KPOOL_PREFILL_TP.override(enabled):
            return indexer._get_topk_ragged_kpool_plan(
                batch, 0, q_fp8[:q_rows], weights[:q_rows], metadata
            )

    # Without padding rows the gather needs more rows than the output. The small
    # logits budget splits each rank's rows across row chunks.
    for q_rows, budget_bytes in ((n_real, 1 << 40), (n_real + pad_rows, 1 << 26)):
        with (
            patch.object(indexer_module, "get_token_to_kv_pool", return_value=pool),
            patch.object(
                indexer_module, "mqa_logits_budget_bytes", return_value=budget_bytes
            ),
            patch.object(
                indexer_module.IndexerKPool,
                "_fp8_mqa_logits",
                wraps=indexer_module.IndexerKPool._fp8_mqa_logits,
            ) as logits,
            patch.object(
                GroupCoordinator,
                "all_gather_into_tensor",
                autospec=True,
                side_effect=GroupCoordinator.all_gather_into_tensor,
            ) as gather,
        ):
            replicated = topk(False, q_rows)
            gather.assert_not_called()
            logits.reset_mock()
            sharded = topk(True, q_rows)
            scored = sum(c.kwargs["q_fp8"].shape[0] for c in logits.call_args_list)
            assert scored <= (n_real + WORLD - 1) // WORLD, f"scored {scored} rows"
            with get_forward().scoped(attn_input_scattered=True):
                topk(True, q_rows)
            gather.assert_called_once()

        # Top-k order is not stable, and an equal score at the cutoff can swap a
        # selected group, so a few rows may differ. A misplaced shard moves thousands.
        same = (sharded.sort(dim=-1).values == replicated.sort(dim=-1).values).all(
            dim=-1
        )
        assert (~same).sum() < 5, f"{(~same).sum()} rows differ"
        # A tie keeps the row's width, while a row no rank scored stays all -1.
        assert ((sharded < 0).sum(-1) == (replicated < 0).sum(-1)).all()


class TestPrefillTPRows(CustomTestCase):
    def test_sharded_path_matches_replicated_on_tp_ranks(self):
        with tempfile.TemporaryDirectory(prefix="kpool-tp-") as tmp:
            mp.spawn(_run_rank, args=(f"file://{tmp}/init",), nprocs=WORLD)


if __name__ == "__main__":
    unittest.main()
