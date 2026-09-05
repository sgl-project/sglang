import unittest

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.decode_kv_broadcast import (
    DecodeKVBroadcaster,
    default_chunk_bytes,
    resolve_decode_kv_broadcast,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci

# CUDA-only: the relay's gather/scatter is a Triton kernel.
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class _PairedComm:
    """Two-rank comm stub: the source's relay bytes reach the receiver.

    Starts disabled like a real PyNcclCommunicator, and refuses a broadcast
    while it is: a disabled one returns silently, which would leave the
    receiver scattering from an uninitialized relay buffer.
    """

    def __init__(self, rank: int, wire: list, available: bool = True):
        self.rank = rank
        self.world_size = 2
        self.wire = wire
        self.available = available
        self.disabled = True
        self.broadcasts = 0

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        if self.disabled:
            raise AssertionError("relay broadcast on a disabled communicator")
        self.broadcasts += 1
        if self.rank == src:
            self.wire.append(tensor.clone())
        else:
            tensor.copy_(self.wire.pop(0))


def _make_pool(num_tokens: int, layer_num: int) -> MLATokenToKVPool:
    return MLATokenToKVPool(
        size=num_tokens,
        page_size=1,
        dtype=torch.bfloat16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=layer_num,
        device="cuda",
        enable_memory_saver=False,
    )


def _make_dsa_pool(num_tokens: int, layer_num: int) -> DSATokenToKVPool:
    return DSATokenToKVPool(
        size=num_tokens,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=layer_num,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=512 + 64,
    )


# Stated independently of `RELAYABLE_POOLS` on purpose: relaying a new state
# type has to be a deliberate change here too, not something a table edit
# silently satisfies.
_EXPECTED_RELAYED_STATE_TYPES = frozenset({StateType.DSA})


class _UncoveredPool(MLATokenToKVPool):
    """Stands in for a subclass with side buffers the relay does not cover."""


def _fill(buffers, seed: int):
    for layer_id, buffer in enumerate(buffers):
        buffer.copy_(
            torch.randint(
                1, 127, buffer.shape, device=buffer.device, dtype=torch.uint8
            ).to(buffer.dtype)
            + seed
            + layer_id
        )


class TestDecodeKVBroadcaster(unittest.TestCase):
    def _make_pair(
        self, src_pool, dst_pool, chunk_bytes, src_draft=None, dst_draft=None
    ):
        wire = []
        src = DecodeKVBroadcaster(
            token_to_kv_pool=src_pool,
            draft_token_to_kv_pool=src_draft,
            relay_comm=_PairedComm(0, wire),
            attn_tp_rank=0,
            attn_tp_size=2,
            forward_stream=torch.cuda.current_stream(),
            chunk_bytes=chunk_bytes,
        )
        dst = DecodeKVBroadcaster(
            token_to_kv_pool=dst_pool,
            draft_token_to_kv_pool=dst_draft,
            relay_comm=_PairedComm(1, wire),
            attn_tp_rank=1,
            attn_tp_size=2,
            forward_stream=torch.cuda.current_stream(),
            chunk_bytes=chunk_bytes,
        )
        return src, dst, wire

    def _run_relay(self, indices_list, chunk_bytes):
        src_pool = _make_pool(num_tokens=32, layer_num=3)
        dst_pool = _make_pool(num_tokens=32, layer_num=3)
        _fill(src_pool.kv_buffer, seed=1)

        src, dst, wire = self._make_pair(src_pool, dst_pool, chunk_bytes)
        src.broadcast(kv_indices_list=indices_list, state_indices_list=[])
        dst.broadcast(kv_indices_list=indices_list, state_indices_list=[])
        self.assertEqual(wire, [])
        return src, src_pool, dst_pool

    def _assert_relayed(self, src_buffers, dst_buffers, relayed, num_rows):
        untouched = [i for i in range(num_rows) if i not in set(relayed)]
        for src_buffer, dst_buffer in zip(src_buffers, dst_buffers):
            torch.testing.assert_close(dst_buffer[relayed], src_buffer[relayed])
            self.assertTrue(torch.all(dst_buffer[untouched] == 0))

    def test_multiple_reqs_and_chunks(self):
        indices = [
            torch.tensor([3, 4], dtype=torch.int32, device="cuda"),
            torch.tensor([20, 9, 10, 11, 12], dtype=torch.int32, device="cuda"),
        ]
        # 3 layers * 24 B per row, so a chunk holds 2 of the 7 tokens.
        src, src_pool, dst_pool = self._run_relay(indices, chunk_bytes=3 * 24 * 2)
        # Asserted, not just commented: a chunk sized without the layer count
        # would relay the same bytes correctly in one round and go unnoticed.
        self.assertEqual(src.relay_comm.broadcasts, 4)
        self._assert_relayed(
            src_pool.kv_buffer,
            dst_pool.kv_buffer,
            [3, 4, 20, 9, 10, 11, 12],
            dst_pool.size,
        )

    def test_indexer_k_cache_is_relayed_by_page(self):
        """The DSA indexer's K cache is page-indexed, unlike the token-indexed KV.

        It rides on the same transfer, so skipping the network pull without
        relaying it leaves the peers' indexer reading stale rows.
        """
        src_pool = _make_dsa_pool(num_tokens=256, layer_num=2)
        dst_pool = _make_dsa_pool(num_tokens=256, layer_num=2)
        _fill(src_pool.kv_buffer, seed=1)
        _fill(src_pool.index_k_with_scale_buffer, seed=2)

        kv_indices = [torch.arange(64, 192, dtype=torch.int32, device="cuda")]
        state_indices = [torch.tensor([1, 2], dtype=torch.int32, device="cuda")]
        # 2 layers * 8448 B per index row -> 1 page per chunk, so the KV and
        # index groups each split across several broadcasts.
        src, dst, wire = self._make_pair(src_pool, dst_pool, chunk_bytes=2 * 8448)
        for broadcaster in (src, dst):
            broadcaster.broadcast(
                kv_indices_list=kv_indices, state_indices_list=state_indices
            )
        self.assertEqual(wire, [])
        # 128 KV rows at 7 per chunk, then 2 index pages at 1 per chunk.
        self.assertEqual(src.relay_comm.broadcasts, 19 + 2)

        self._assert_relayed(
            src_pool.kv_buffer,
            dst_pool.kv_buffer,
            list(range(64, 192)),
            dst_pool.size,
        )
        self._assert_relayed(
            src_pool.index_k_with_scale_buffer,
            dst_pool.index_k_with_scale_buffer,
            [1, 2],
            dst_pool.index_k_with_scale_buffer[0].shape[0],
        )

    def test_rejects_pool_without_a_relay_group(self):
        """A pool with side buffers this relay does not cover must not start.

        Silently relaying only kv_buffer would corrupt the peers' KV.
        """
        pool = _UncoveredPool(
            size=8,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=8,
            qk_rope_head_dim=4,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        with self.assertRaises(RuntimeError):
            DecodeKVBroadcaster(
                token_to_kv_pool=pool,
                draft_token_to_kv_pool=None,
                relay_comm=_PairedComm(0, []),
                attn_tp_rank=0,
                attn_tp_size=2,
                forward_stream=torch.cuda.current_stream(),
            )

    def test_rejects_an_unavailable_communicator(self):
        """A relay whose NCCL comm failed to init must not start.

        PyNcclCommunicator.broadcast returns silently when unavailable, which
        would leave the peers scattering from an uninitialized relay buffer.
        """
        pool = _make_pool(num_tokens=8, layer_num=2)
        with self.assertRaises(RuntimeError):
            DecodeKVBroadcaster(
                token_to_kv_pool=pool,
                draft_token_to_kv_pool=None,
                relay_comm=_PairedComm(0, [], available=False),
                attn_tp_rank=0,
                attn_tp_size=2,
                forward_stream=torch.cuda.current_stream(),
            )

    def test_rejects_a_comm_that_is_not_the_attn_tp_group(self):
        """The relay group must be the one the KV manager gated and pulls on.

        The network-side skip picks the puller by the KV manager's attn TP
        rank, so a communicator over any other group would put the broadcast
        source on a rank holding no transferred KV.
        """
        pool = _make_pool(num_tokens=8, layer_num=2)
        for attn_tp_rank, attn_tp_size in ((0, 4), (1, 2)):
            with self.subTest(attn_tp_rank=attn_tp_rank, attn_tp_size=attn_tp_size):
                with self.assertRaises(RuntimeError):
                    DecodeKVBroadcaster(
                        token_to_kv_pool=pool,
                        draft_token_to_kv_pool=None,
                        relay_comm=_PairedComm(0, []),
                        attn_tp_rank=attn_tp_rank,
                        attn_tp_size=attn_tp_size,
                        forward_stream=torch.cuda.current_stream(),
                    )

    def test_draft_pool_is_relayed_with_the_target_pool(self):
        """Under MTP the draft pool holds its own copy of the same rows.

        It is filled by the same transfer and indexed by the same pool slots,
        so relaying only the target pool leaves the draft model reading stale
        KV on the peer ranks.
        """
        src_pool = _make_pool(num_tokens=32, layer_num=3)
        dst_pool = _make_pool(num_tokens=32, layer_num=3)
        src_draft = _make_pool(num_tokens=32, layer_num=1)
        dst_draft = _make_pool(num_tokens=32, layer_num=1)
        _fill(src_pool.kv_buffer, seed=1)
        _fill(src_draft.kv_buffer, seed=5)

        indices = [torch.tensor([3, 4, 5], dtype=torch.int32, device="cuda")]
        src, dst, wire = self._make_pair(
            src_pool,
            dst_pool,
            chunk_bytes=default_chunk_bytes(),
            src_draft=src_draft,
            dst_draft=dst_draft,
        )
        for broadcaster in (src, dst):
            broadcaster.broadcast(kv_indices_list=indices, state_indices_list=[])
        self.assertEqual(wire, [])

        for src_buffers, dst_buffers in (
            (src_pool.kv_buffer, dst_pool.kv_buffer),
            (src_draft.kv_buffer, dst_draft.kv_buffer),
        ):
            self._assert_relayed(src_buffers, dst_buffers, [3, 4, 5], dst_pool.size)

    def test_empty_is_noop(self):
        wire = []
        pool = _make_pool(num_tokens=8, layer_num=2)
        broadcaster = DecodeKVBroadcaster(
            token_to_kv_pool=pool,
            draft_token_to_kv_pool=None,
            relay_comm=_PairedComm(0, wire),
            attn_tp_rank=0,
            attn_tp_size=2,
            forward_stream=torch.cuda.current_stream(),
        )
        broadcaster.broadcast(kv_indices_list=[], state_indices_list=[])
        broadcaster.broadcast(
            kv_indices_list=[torch.tensor([], dtype=torch.int32, device="cuda")],
            state_indices_list=[],
        )
        self.assertEqual(wire, [])


class TestResolveDecodeKVBroadcast(unittest.TestCase):
    """The startup gate for configurations a single pulled copy cannot serve.

    The relay writes only the buffers `_build_relay_groups` covers, so anything
    else the transfer would have written has to be rejected here instead --
    reaching the relay with it enabled corrupts the peers' caches silently.
    """

    @staticmethod
    def _resolve(enabled=True, **overrides):
        kwargs = dict(
            state_types=[],
            is_mla_backend=True,
            attn_tp_size=2,
            supports_decode_kv_broadcast=True,
            enable_hisparse=False,
            enable_staging=False,
            enable_decode_hicache=False,
        )
        kwargs.update(overrides)
        with envs.SGLANG_ENABLE_DISAGG_MLA_DECODE_KV_BROADCAST.override(enabled):
            return resolve_decode_kv_broadcast(**kwargs)

    def test_off_unless_the_env_var_is_set(self):
        self.assertFalse(self._resolve(enabled=False))

    def test_accepts_exactly_the_relayed_state_types(self):
        """A state type added without a relay group must not silently pass."""
        for state_type in StateType:
            with self.subTest(state_type=state_type):
                if state_type in _EXPECTED_RELAYED_STATE_TYPES:
                    self.assertTrue(self._resolve(state_types=[state_type]))
                else:
                    with self.assertRaises(RuntimeError):
                        self._resolve(state_types=[state_type])

    def test_rejects_a_relayed_and_unrelayed_state_mix(self):
        """One relayed state type must not license an unrelayed one alongside it."""
        with self.assertRaises(RuntimeError):
            self._resolve(state_types=[StateType.DSA, StateType.MAMBA])

    def test_rejects_unsupported_paths(self):
        for overrides in (
            {"is_mla_backend": False},
            {"attn_tp_size": 1},
            {"enable_hisparse": True},
            {"enable_staging": True},
            {"enable_decode_hicache": True},
            {"supports_decode_kv_broadcast": False},
        ):
            with self.subTest(**overrides):
                with self.assertRaises(RuntimeError):
                    self._resolve(**overrides)


class TestDefaultChunkBytes(unittest.TestCase):
    def test_rejects_a_non_positive_chunk(self):
        """A zero chunk would size the relay buffer to nothing."""
        for chunk_mib in (0, -1):
            with self.subTest(chunk_mib=chunk_mib):
                with envs.SGLANG_DISAGG_MLA_DECODE_KV_BROADCAST_CHUNK_MIB.override(
                    chunk_mib
                ):
                    with self.assertRaises(RuntimeError):
                        default_chunk_bytes()


if __name__ == "__main__":
    unittest.main()
