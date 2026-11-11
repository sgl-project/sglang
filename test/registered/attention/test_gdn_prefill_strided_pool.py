"""GDN prefill on a state pool laid out like unified memory's: each slot of a
layer is a slice of a wide arena entry holding every layer's conv and SSM state.

`GDNAttnBackend.forward_extend` on the Triton prefill kernels must give the same
outputs and states, bit for bit, whether it runs in place on that pool, on packed
copies of its rows, or on a packed pool holding the same values: for fresh
requests, cached prefixes and a follow-up chunk of several mixed-length requests.
In place it must also leave every other byte of the arena alone. The highest slots
start past 2**31 elements of the arena, where a 32-bit slot offset wraps.
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.state_pool_io import (
    PoolStaging,
    stage_pool_states,
)
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.srt.mem_cache.kv.decl import STATE_PART, TensorPartDecl
from sglang.srt.mem_cache.kv.layout.state import state_entry_layout
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

HEADS, V_HEADS, HEAD_DIM, CONV = 2, 4, 128, 4
DIM = (2 * HEADS + V_HEADS) * HEAD_DIM
LAYERS = (0, 1)
# Slot pitch of the arena, wider than the two layers' states as when other
# layers share the entry; 4 KiB past a power of two. Arbitrary otherwise.
PITCH_BYTES = (64 << 20) + 4096
NUM_SLOTS = 136
SENTINEL = 0x5A
# Two mixed-length chunks of six requests: fresh requests, cached prefixes, and
# lengths around the 64-token chunk of the delta rule.
FIRST_CHUNK = ([1, 37, 64, 129, 300, 7], [0, 0, 48, 0, 1000, 5])
SECOND_CHUNK = ([5, 64, 1, 200, 33, 70], [1, 37, 112, 129, 1300, 12])


def _layer(layer_id):
    g = torch.Generator(device="cuda").manual_seed(100 + layer_id)

    def randn(*shape, dtype=torch.bfloat16, scale=0.2):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)

    return SimpleNamespace(
        layer_id=layer_id,
        num_q_heads=HEADS,
        num_k_heads=HEADS,
        num_v_heads=V_HEADS,
        head_q_dim=HEAD_DIM,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        q_dim=HEADS * HEAD_DIM,
        k_dim=HEADS * HEAD_DIM,
        v_dim=V_HEADS * HEAD_DIM,
        conv_weights=randn(DIM, CONV),
        bias=randn(DIM),
        activation="silu",
        A_log=randn(V_HEADS, dtype=torch.float32),
        dt_bias=randn(V_HEADS, dtype=torch.float32),
    )


class _Arena:
    """Per-layer conv and SSM views over one strided arena, built by the layout
    the unified state pool uses; ``slots`` are the slots requests use."""

    def __init__(self, slots):
        self.slots = slots
        parts = [
            TensorPartDecl(
                name="conv.0",
                shape=(DIM, CONV - 1),
                dtype=torch.bfloat16,
                layer_ids=LAYERS,
            ),
            TensorPartDecl(
                name="temporal",
                shape=(V_HEADS, HEAD_DIM, HEAD_DIM),
                dtype=torch.float32,
                layer_ids=LAYERS,
            ),
        ]
        self.layout = state_entry_layout(parts)
        self.parts = {part.name: part for part in parts}
        self.raw = torch.empty(
            NUM_SLOTS * PITCH_BYTES, dtype=torch.uint8, device="cuda"
        )
        views = self.layout.views(self.raw, NUM_SLOTS, entry_bytes=PITCH_BYTES)
        self.conv, self.ssm = views["conv.0"], views["temporal"]

    def fill(self, conv, ssm):
        """Every byte the sentinel, then the given per-layer rows at ``slots``."""
        self.raw.fill_(SENTINEL)
        for layer in LAYERS:
            self.conv[layer][self.slots] = conv[layer][self.slots]
            self.ssm[layer][self.slots] = ssm[layer][self.slots]

    def bytes_outside_slots_untouched(self):
        rows = self.raw.view(NUM_SLOTS, PITCH_BYTES)
        for name, part in self.parts.items():
            row_bytes = math.prod(part.shape) * part.dtype.itemsize
            for layer in LAYERS:
                start = (
                    self.layout.offsets[name] + layer * self.layout.layer_strides[name]
                )
                rows[self.slots, start : start + row_bytes] = SENTINEL
        return bool((self.raw == SENTINEL).all())


def _backend(caches, slot_of_request):
    backend = GDNAttnBackend.__new__(GDNAttnBackend)
    backend.state_pool = None
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda req_pool_indices: slot_of_request[req_pool_indices]
    )
    backend._state_ids = IdSpace(key=("state", None))
    pool = SimpleNamespace(layer_transfer_counter=None, start_layer=0)
    pool.layer_router = KVLayerRouter.from_layer_maps(pool, (), owner="full")
    pool.layer_router.bind_part(
        sub_pool="state",
        part=STATE_PART,
        view=SimpleNamespace(mamba2_layer_cache=lambda layer_id: caches[layer_id]),
        indices={layer: layer for layer in LAYERS},
    )
    backend.token_to_kv_pool = pool
    backend.device = torch.device("cuda")
    backend._mamba_chunk_size = 64
    backend.conv_states_shape = caches[0].conv[0].shape
    backend.accept_lens_pool = None
    backend.prefill_graph_metadata = None
    backend.kernel_dispatcher = GDNKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.TRITON,
    )
    return backend


def _batch(extend_lens, prefix_lens):
    bs = len(extend_lens)
    starts = [sum(extend_lens[:i]) for i in range(bs)]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        logical_forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        _original_batch_size=None,
        req_pool_indices=torch.arange(bs, device="cuda"),
        extend_seq_lens=torch.tensor(extend_lens, device="cuda"),
        extend_prefix_lens=torch.tensor(prefix_lens, device="cuda"),
        extend_start_loc=torch.tensor(starts, device="cuda"),
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens_cpu=list(prefix_lens),
        tbo_parent_token_range=None,
        multi_item_delimiter_indices=None,
        mamba_track_mask=None,
        mamba_track_indices=None,
        mamba_track_seqlens=None,
        mamba_prefill_track_mask_cpu=None,
        mamba_track_seqlens_cpu=None,
    )


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestGDNPrefillStridedPool(CustomTestCase):
    def setUp(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        self.layers = [_layer(layer) for layer in LAYERS]
        num_requests = len(FIRST_CHUNK[0])
        # Distinct random slots, the highest slot among them; slot 0 is the sink.
        slots = torch.randperm(NUM_SLOTS - 2, generator=g, device="cuda")[
            : num_requests - 1
        ]
        self.slots = torch.cat(
            [slots + 1, torch.tensor([NUM_SLOTS - 1], device="cuda")]
        )[torch.randperm(num_requests, generator=g, device="cuda")].int()
        # Packed pools: the values every run starts from, zero for fresh requests.
        self.conv = (
            torch.randn(
                len(LAYERS), NUM_SLOTS, DIM, CONV - 1, generator=g, device="cuda"
            )
            * 0.3
        ).to(torch.bfloat16)
        self.ssm = (
            torch.randn(
                len(LAYERS),
                NUM_SLOTS,
                V_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                generator=g,
                device="cuda",
            )
            * 0.05
        )
        for request, prefix in enumerate(FIRST_CHUNK[1]):
            if prefix == 0:
                self.conv[:, self.slots[request]] = 0
                self.ssm[:, self.slots[request]] = 0
        self.inputs = []
        for lens, _ in (FIRST_CHUNK, SECOND_CHUNK):
            tokens = sum(lens)
            self.inputs.append(
                tuple(
                    torch.randn(tokens, width, generator=g, device="cuda").to(
                        torch.bfloat16
                    )
                    for width in (DIM, V_HEADS, V_HEADS)
                )
            )
        self.arena = _Arena(self.slots.long())

    def _prefill(self, conv, ssm):
        """Both chunks through both layers; the outputs of every forward."""
        caches = [
            SimpleNamespace(conv=[conv[layer]], temporal=ssm[layer]) for layer in LAYERS
        ]
        slot_of_request = self.slots
        outputs = []
        for (lens, prefixes), (mixed_qkv, a, b) in zip(
            (FIRST_CHUNK, SECOND_CHUNK), self.inputs
        ):
            backend = _backend(caches, slot_of_request)
            batch = _batch(lens, prefixes)
            backend.init_forward_metadata(batch)
            for layer in self.layers:
                outputs.append(
                    backend.forward_extend(
                        layer, batch, mixed_qkv.clone(), a.clone(), b.clone()
                    )
                )
        return outputs

    def _assert_same(self, outputs, conv, ssm, ref_outputs, ref_conv, ref_ssm):
        for output, ref_output in zip(outputs, ref_outputs, strict=True):
            self.assertTrue(torch.equal(output, ref_output), "outputs differ")
        for layer in LAYERS:
            self.assertTrue(
                torch.equal(
                    conv[layer][self.arena.slots], ref_conv[layer][self.arena.slots]
                ),
                "conv states differ",
            )
            self.assertTrue(
                torch.equal(
                    ssm[layer][self.arena.slots], ref_ssm[layer][self.arena.slots]
                ),
                "ssm states differ",
            )

    def test_in_place_and_staged_prefill_match_the_packed_pool(self):
        arena = self.arena
        self.assertFalse(arena.ssm[0].is_contiguous())
        self.assertGreater(
            int(self.slots.max()) * arena.ssm[0].stride(0), torch.iinfo(torch.int32).max
        )
        ref_conv, ref_ssm = self.conv.clone(), self.ssm.clone()
        ref_outputs = self._prefill(ref_conv, ref_ssm)

        arena.fill(self.conv, self.ssm)
        with patch.object(
            gdn_backend,
            "stage_pool_states",
            side_effect=AssertionError("staged a pool the kernels can address"),
        ):
            outputs = self._prefill(arena.conv, arena.ssm)
        self._assert_same(
            outputs, arena.conv, arena.ssm, ref_outputs, ref_conv, ref_ssm
        )
        self.assertTrue(arena.bytes_outside_slots_untouched())

        # A prefill kernel that cannot follow the pitch runs on packed copies.
        arena.fill(self.conv, self.ssm)
        with (
            patch.object(TritonGDNKernel, "extend_honors_slot_pitch", False),
            patch.object(
                gdn_backend, "stage_pool_states", wraps=stage_pool_states
            ) as stage,
            patch.object(
                PoolStaging, "from_indices", wraps=PoolStaging.from_indices
            ) as plan,
        ):
            outputs = self._prefill(arena.conv, arena.ssm)
        # Two forwards of two layers each; one set of rows per forward.
        self.assertEqual(stage.call_count, 4)
        self.assertEqual(plan.call_count, 2)
        self._assert_same(
            outputs, arena.conv, arena.ssm, ref_outputs, ref_conv, ref_ssm
        )
        self.assertTrue(arena.bytes_outside_slots_untouched())


if __name__ == "__main__":
    unittest.main()
