"""Hybrid DSA MTP state transfer, including CP8 prefill / TP8 EP8 decode."""

import ctypes
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import build_kv_layer_ids, setup_state_kv_args
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _CPUStatePool:
    """CPU-backed KDA state descriptors; allocation is the test boundary."""

    def __init__(self, heads, state_shard_on_attn_cp=False):
        self.heads = heads
        self.state_shard_on_attn_cp = state_shard_on_attn_cp
        self.conv = torch.zeros((4, 3, heads, 2), dtype=torch.int32)
        self.ssm = torch.zeros((4, heads, 2, 2), dtype=torch.int32)

    def get_contiguous_buf_infos(self):
        tensors = (self.conv, self.ssm)
        return (
            [t.data_ptr() for t in tensors],
            [t.nbytes for t in tensors],
            [t[0].nbytes for t in tensors],
        )

    def get_state_dim_per_tensor(self):
        return [self.heads, self.heads]

    def get_state_slice_outer_counts(self):
        return [3, 1]

    def get_state_conv_shard_groups(self):
        return [None, None]

    def get_state_layer_ids(self):
        return [0, 0]


def _dsa_pool(layers, *, compress=False):
    return DSATokenToKVPool(
        size=1024,
        page_size=64,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        kv_cache_dim=16,
        dtype=torch.bfloat16,
        layer_num=layers,
        index_head_dim=128,
        device="cpu",
        enable_memory_saver=False,
        index_kpool=2 if compress else 1,
        index_kpool_compress=compress,
        max_running_requests=4,
    )


def _hybrid_pool(ids=(3, 7), *, heads=8, compress=False, state_shard_on_attn_cp=False):
    return HybridLinearKVPool(
        size=1024,
        dtype=torch.bfloat16,
        page_size=64,
        head_num=heads,
        head_dim=16,
        full_attention_layer_ids=list(ids),
        device="cpu",
        mamba_pool=_CPUStatePool(heads, state_shard_on_attn_cp),
        use_mla=True,
        use_dsa=True,
        full_kv_pool=_dsa_pool(len(ids), compress=compress),
    )


def _register(target, draft):
    args = KVArgs()
    setup_state_kv_args(args, target, draft, total_kv_layers=45)
    args.engine_rank = 0
    return args


def _manager(args):
    manager = object.__new__(MooncakeKVManager)
    manager.kv_args = args
    manager._mamba_state_cp_sharded = args.mamba_state_shard_on_attn_cp
    manager.attn_tp_size = 1
    manager.attn_cp_size = 8
    manager.attn_cp_rank = 0
    manager.pp_size = 1
    manager.is_mla_backend = False
    manager.is_hybrid_mla_backend = True
    manager.enable_custom_mem_pool = False
    manager.max_transfer_batch_indices = 0

    # Replace only RDMA I/O; block construction/pairing and reslicing stay real.
    def transfer(session, blocks):
        for src, dst, size in blocks:
            ctypes.memmove(dst, src, size)
        return 0

    manager._transfer_data = transfer
    return manager


class TestGLM5PDMTP(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_context().override_server_args(tp_size=8, attn_cp_size=8, ep_size=1)
        )

    def test_draft_index_k_is_registered_with_its_own_layer_id(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                target, draft = _hybrid_pool(), _dsa_pool(1)
                raw_draft = draft
                if wrapped:
                    draft = _hybrid_pool((0,))
                    raw_draft = draft.full_kv_pool
                args = _register(target, draft)
                index = args.state_types.index(StateType.DSA)
                self.assertEqual(
                    args.state_data_ptrs[index],
                    target.full_kv_pool.get_state_buf_infos()[0]
                    + raw_draft.get_state_buf_infos()[0],
                )
                self.assertEqual(args.state_layer_ids[index], [3, 7, 45])
                self.assertEqual(
                    build_kv_layer_ids(
                        token_to_kv_pool=target,
                        draft_token_to_kv_pool=draft,
                        num_draft_entries=1,
                        num_hidden_layers=45,
                    ),
                    [3, 7, 45],
                )

    def test_without_draft_keeps_existing_state_registration(self):
        target = _hybrid_pool()
        args = _register(target, None)
        self.assertEqual(args.state_types, [StateType.MAMBA, StateType.DSA])
        self.assertEqual(args.state_layer_ids, [[0, 0], []])
        self.assertEqual(
            args.state_data_ptrs[1], target.full_kv_pool.get_state_buf_infos()[0]
        )

    def test_plain_dsa_draft_registration_keeps_existing_layout(self):
        target, draft = _dsa_pool(2), _dsa_pool(1)
        args = _register(target, draft)
        self.assertEqual(args.state_types, [StateType.DSA])
        self.assertEqual(args.state_layer_ids, [[]])
        self.assertEqual(
            args.state_data_ptrs[0],
            target.get_state_buf_infos()[0] + draft.get_state_buf_infos()[0],
        )

    def test_compressed_tail_includes_draft_buffers(self):
        target, draft = _hybrid_pool(compress=True), _dsa_pool(1, compress=True)
        args = _register(target, draft)
        self.assertEqual(
            args.state_types, [StateType.MAMBA, StateType.DSA, StateType.DSA_TAIL]
        )
        for field, position in (
            ("state_data_ptrs", 0),
            ("state_data_lens", 1),
            ("state_item_lens", 2),
        ):
            self.assertEqual(
                getattr(args, field)[2],
                target.full_kv_pool.get_compress_tail_buf_infos()[position]
                + draft.get_compress_tail_buf_infos()[position],
            )

    def test_pp_stage_pairs_draft_index_k_with_the_decode_draft(self):
        source, draft = _hybrid_pool((7,)), _dsa_pool(1)
        destination, destination_draft = _hybrid_pool((3, 7, 11)), _dsa_pool(1)
        source.full_kv_pool.index_k_with_scale_buffer[0].fill_(17)
        draft.index_k_with_scale_buffer[0].fill_(91)
        src, dst = _register(source, draft), _register(destination, destination_draft)
        manager = _manager(src)
        manager.pp_size = 2
        manager.kv_args.prefill_start_layer = 1
        index = src.state_types.index(StateType.DSA)
        self.assertEqual(
            manager._send_kvcache_generic(
                "session",
                src.state_data_ptrs[index],
                dst.state_data_ptrs[index],
                src.state_item_lens[index],
                np.array([1]),
                np.array([2]),
                None,
                state_type=StateType.DSA,
                src_layer_ids=src.state_layer_ids[index],
                dst_layer_ids=dst.state_layer_ids[index],
            ),
            0,
        )
        self.assertTrue(
            torch.all(destination.full_kv_pool.index_k_with_scale_buffer[0] == 0)
        )
        self.assertTrue(
            torch.all(destination.full_kv_pool.index_k_with_scale_buffer[1][2] == 17)
        )
        self.assertTrue(
            torch.all(destination_draft.index_k_with_scale_buffer[0][2] == 91)
        )

    def test_nixl_pairs_target_and_draft_index_k_by_layer_id(self):
        source, draft = _hybrid_pool((7,)), _dsa_pool(1)
        destination, destination_draft = _hybrid_pool((3, 7, 11)), _dsa_pool(1)
        source.full_kv_pool.index_k_with_scale_buffer[0].fill_(17)
        draft.index_k_with_scale_buffer[0].fill_(91)
        src, dst = _register(source, draft), _register(destination, destination_draft)
        src.gpu_id = 0
        src.kv_data_ptrs = []
        src.prefill_start_layer = 1

        class Agent:
            def get_xfer_descs(self, requests, mem_kind):
                return requests

            def initialize_xfer(self, op, source_descs, dest_descs, peer, notif):
                return source_descs, dest_descs

            def transfer(self, handle):
                for source_row, dest_row in zip(*handle):
                    ctypes.memmove(
                        int(dest_row[0]), int(source_row[0]), int(source_row[1])
                    )
                return "DONE"

        manager = object.__new__(NixlKVManager)
        manager.kv_args = src
        manager.attn_tp_size = 1
        manager.pp_size = 2
        manager.is_mla_backend = False
        manager.prep_handles = {}
        manager.agent = Agent()
        handles = manager.maybe_send_extra(
            "peer",
            [None, [1]],
            dst.state_data_ptrs,
            [None, [2]],
            0,
            "notif",
            8,
            dst_state_item_lens=dst.state_item_lens,
            dst_state_dim_per_tensor=dst.state_dim_per_tensor,
            dst_state_layer_ids=dst.state_layer_ids,
        )
        self.assertEqual(len(handles), 1)
        self.assertTrue(
            torch.all(destination.full_kv_pool.index_k_with_scale_buffer[0] == 0)
        )
        self.assertTrue(
            torch.all(destination.full_kv_pool.index_k_with_scale_buffer[1][2] == 17)
        )
        self.assertTrue(
            torch.all(destination.full_kv_pool.index_k_with_scale_buffer[2] == 0)
        )
        self.assertTrue(
            torch.all(destination_draft.index_k_with_scale_buffer[0][2] == 91)
        )

    def test_replicated_cp_state_to_tp8_transfers_kv_index_k_and_kda_slices(self):
        parallel = get_parallel()
        self.assertEqual(
            (
                parallel.attn_tp_size,
                parallel.attn_cp_size,
                parallel.moe_tp_size,
                parallel.moe_ep_size,
            ),
            (1, 8, 8, 1),
        )
        source, draft = _hybrid_pool(), _dsa_pool(1)
        src = _register(source, draft)
        for i, pool in enumerate((source.full_kv_pool, draft)):
            for layer, buf in enumerate(pool.kv_buffer):
                buf.fill_(10 + 10 * i + layer)
            for layer, buf in enumerate(pool.index_k_with_scale_buffer):
                buf.fill_(50 + 10 * i + layer)
        for buf in (source.mamba_pool.conv, source.mamba_pool.ssm):
            buf.copy_(torch.arange(buf.numel()).reshape(buf.shape))
        src_kv = source.get_contiguous_buf_infos()
        draft_kv = draft.get_contiguous_buf_infos()
        src_kv_ids = build_kv_layer_ids(
            token_to_kv_pool=source,
            draft_token_to_kv_pool=draft,
            num_draft_entries=len(draft_kv[0]),
            num_hidden_layers=45,
        )
        for rank in range(8):
            with self.subTest(decode_rank=rank):
                with get_context().override_server_args(
                    tp_size=8, attn_cp_size=1, ep_size=8
                ):
                    parallel = get_parallel()
                    self.assertEqual(
                        (parallel.attn_tp_size, parallel.moe_ep_size), (8, 8)
                    )
                    destination, destination_draft = _hybrid_pool(heads=1), _dsa_pool(1)
                    dst = _register(destination, destination_draft)
                    decode_manager = _manager(dst)
                    decode_manager.attn_tp_size = parallel.attn_tp_size
                    decode_manager.attn_cp_size = 1
                    decode_manager.pp_rank = 0
                    decode_manager.enable_all_cp_ranks_for_transfer = True
                    decode_manager.kv_args.engine_rank = rank
                    info = PrefillServerInfo(1, 8, 1, 1, 64, "bfloat16", False)
                    CommonKVManager._resolve_rank_mapping(decode_manager, info)
                    self.assertEqual(info.target_cp_ranks, list(range(8)))
                    self.assertEqual(info.required_prefill_response_num, 8)
                    self.assertEqual(info.required_dst_info_num, 8)
                dst_kv = destination.get_contiguous_buf_infos()
                dst_draft_kv = destination_draft.get_contiguous_buf_infos()
                manager = _manager(src)
                for cp_rank in range(8):
                    manager.attn_cp_rank = cp_rank
                    rc = manager._send_kvcache_generic(
                        "session",
                        src_kv[0] + draft_kv[0],
                        dst_kv[0] + dst_draft_kv[0],
                        src_kv[2] + draft_kv[2],
                        np.array([cp_rank + 1]),
                        np.array([cp_rank + 5]),
                        None,
                        src_layer_ids=src_kv_ids,
                        dst_layer_ids=src_kv_ids,
                    )
                    self.assertEqual(rc, 0)
                    skip_kv, skip_state = manager._get_dsa_cache_transfer_skip_flags(
                        None
                    )
                    self.assertFalse(skip_kv)
                    self.assertEqual(skip_state, cp_rank != 0)
                    if skip_state:
                        continue
                    dsa_index = src.state_types.index(StateType.DSA)
                    self.assertEqual(
                        manager._send_kvcache_generic(
                            "session",
                            src.state_data_ptrs[dsa_index],
                            dst.state_data_ptrs[dsa_index],
                            src.state_item_lens[dsa_index],
                            np.arange(1, 9),
                            np.arange(5, 13),
                            None,
                            state_type=StateType.DSA,
                            src_layer_ids=src.state_layer_ids[dsa_index],
                            dst_layer_ids=dst.state_layer_ids[dsa_index],
                        ),
                        0,
                    )
                    self.assertEqual(
                        manager._send_mamba_state_slice(
                            SimpleNamespace(mooncake_session_id="session"),
                            [1],
                            src.state_data_ptrs[0],
                            src.state_item_lens[0],
                            src.state_dim_per_tensor[0],
                            dst.state_data_ptrs[0],
                            [2],
                            dst.state_item_lens[0],
                            dst.state_dim_per_tensor[0],
                            rank,
                            8,
                            src.state_conv_shard_groups[0],
                            src.state_slice_outer_counts[0],
                            src.state_layer_ids[0],
                            dst.state_layer_ids[0],
                        ),
                        0,
                    )
                for src_pool, dst_pool in (
                    (source.full_kv_pool, destination.full_kv_pool),
                    (draft, destination_draft),
                ):
                    for src_buf, dst_buf in zip(src_pool.kv_buffer, dst_pool.kv_buffer):
                        self.assertTrue(torch.equal(src_buf[64:576], dst_buf[320:832]))
                    for src_buf, dst_buf in zip(
                        src_pool.index_k_with_scale_buffer,
                        dst_pool.index_k_with_scale_buffer,
                    ):
                        self.assertTrue(torch.equal(src_buf[1:9], dst_buf[5:13]))
                self.assertTrue(
                    torch.equal(
                        destination.mamba_pool.conv[2],
                        source.mamba_pool.conv[1, :, rank : rank + 1],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        destination.mamba_pool.ssm[2],
                        source.mamba_pool.ssm[1, rank : rank + 1],
                    )
                )

    def test_glm5_cp8_kda_shards_reach_matching_decode_ranks(self):
        destinations = [_hybrid_pool(heads=1) for _ in range(8)]
        registrations = [_register(pool, None) for pool in destinations]
        for cp_rank in range(8):
            source = _hybrid_pool(heads=1, state_shard_on_attn_cp=True)
            source.mamba_pool.conv[1].fill_(100 + cp_rank)
            source.mamba_pool.ssm[1].fill_(200 + cp_rank)
            manager = _manager(_register(source, None))
            manager.attn_cp_rank = cp_rank
            self.assertFalse(manager._get_dsa_cache_transfer_skip_flags(None)[1])
            for rank, dst in enumerate(registrations):
                info = SimpleNamespace(
                    dst_attn_tp_size=8,
                    dst_tp_rank=rank,
                    dst_state_data_ptrs=dst.state_data_ptrs,
                    dst_state_item_lens=dst.state_item_lens,
                    dst_state_dim_per_tensor=dst.state_dim_per_tensor,
                    dst_state_layer_ids=dst.state_layer_ids,
                )
                req = SimpleNamespace(
                    mooncake_session_id="session",
                    dst_state_indices=[[2], None],
                )
                self.assertEqual(
                    manager.maybe_send_extra(req, [[1], None], None, info), 0
                )
        for rank, pool in enumerate(destinations):
            self.assertTrue(torch.all(pool.mamba_pool.conv[2] == 100 + rank))
            self.assertTrue(torch.all(pool.mamba_pool.ssm[2] == 200 + rank))

    def test_glm5_cp_nonleader_skips_replicated_index_state(self):
        from unittest.mock import Mock

        source = _hybrid_pool(state_shard_on_attn_cp=True)
        manager = _manager(_register(source, _dsa_pool(1)))
        manager.attn_cp_rank = 3
        manager._send_kvcache_generic = Mock(return_value=0)
        manager.maybe_send_extra(
            SimpleNamespace(dst_state_indices=[None, [1]]),
            [None, [1]],
            None,
            SimpleNamespace(),
        )
        manager._send_kvcache_generic.assert_not_called()

    def test_cache_params_declare_state_group_and_physical_head_count(self):
        from unittest.mock import patch

        from sglang.srt.configs.glm5_next import Glm5NextTextConfig
        from sglang.srt.mem_cache.memory_pool import MambaPool

        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            linear_attn_config={
                "num_heads": 8,
                "head_dim": 2,
                "short_conv_kernel_size": 3,
                "kda_layers": [0],
            },
        )
        for cp, local_heads in [(True, 1), (False, 8)]:
            with (
                self.subTest(cp=cp),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                    return_value=cp,
                ),
            ):
                params = config.mamba2_cache_params
                self.assertIs(params.state_shard_on_attn_cp, cp)
                self.assertEqual(params.shape.temporal[0], local_heads)
                pool = MambaPool(
                    size=4,
                    spec_state_size=0,
                    cache_params=params,
                    mamba_layer_ids=[0],
                    device="cpu",
                )
                self.assertIs(pool.state_shard_on_attn_cp, cp)

    def test_registration_uses_declared_group_and_preserves_tp_default(self):
        for shard_on_cp in [False, True]:
            with self.subTest(shard_on_cp=shard_on_cp):
                pool = _hybrid_pool(state_shard_on_attn_cp=shard_on_cp)
                args = _register(pool, None)
                self.assertIs(args.mamba_state_shard_on_attn_cp, shard_on_cp)
                self.assertEqual(_manager(args)._mamba_state_cp_sharded, shard_on_cp)

    def test_request_pool_layout_metadata_is_optional(self):
        for shard_on_cp in [None, True]:
            with self.subTest(shard_on_cp=shard_on_cp):
                req_pool = SimpleNamespace(get_state_buf_infos=lambda: ([1], [4], [4]))
                if shard_on_cp is not None:
                    req_pool.state_shard_on_attn_cp = shard_on_cp
                args = KVArgs()
                setup_state_kv_args(
                    args,
                    _dsa_pool(1),
                    None,
                    req_to_token_pool=req_pool,
                    total_kv_layers=45,
                )
                self.assertIs(args.mamba_state_shard_on_attn_cp, shard_on_cp is True)

    def test_gdn_keeps_attention_tp_layout_when_cp_is_configured(self):
        from sglang.srt.configs.qwen3_next import Qwen3NextConfig

        config = Qwen3NextConfig(
            num_hidden_layers=2,
            linear_num_value_heads=8,
            linear_num_key_heads=8,
            linear_value_head_dim=2,
            linear_key_head_dim=2,
            linear_conv_kernel_dim=3,
            full_attention_interval=4,
        )
        params = config.mamba2_cache_params
        self.assertEqual(get_parallel().attn_cp_size, 8)
        self.assertEqual(get_parallel().attn_tp_size, 1)
        self.assertFalse(params.state_shard_on_attn_cp)
        self.assertEqual(params.shape.temporal[0], 8)


if __name__ == "__main__":
    unittest.main()
