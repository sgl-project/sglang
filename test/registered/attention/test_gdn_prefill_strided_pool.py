"""GDN prefill on a state pool laid out like unified memory's: each slot of a
layer is a slice of a wide arena entry holding every layer's conv and SSM state,
and requests name slots by virtual ids that the allocator's table translates.

`GDNAttnBackend.forward_extend` on the Triton or the FlashInfer prefill kernel
must give the same outputs, states and prefix-cache track slots, bit for bit,
whether it runs in place on that pool, on packed copies of its rows, or on a
packed pool holding the same values: for fresh requests, cached prefixes and a
follow-up chunk of several mixed-length requests, tracked on and off a chunk
boundary. In place it must also leave every other byte of the arena alone, and
launch the same kernels as on the packed pool. The highest slots start past
2**31 elements of the arena, where a 32-bit slot offset wraps.
"""

import collections
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
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
    is_flashinfer_gdn_prefill_available,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.state_pool_io import (
    PoolStaging,
    stage_pool_states,
)
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, UnitIds
from sglang.srt.mem_cache.kv.decl import STATE_PART, TensorPartDecl
from sglang.srt.mem_cache.kv.layout.state import state_entry_layout
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

HEADS, V_HEADS, HEAD_DIM, CONV = 2, 4, 128, 4
DIM = (2 * HEADS + V_HEADS) * HEAD_DIM
LAYERS = (0, 1)
# Slot pitch of the arena, wider than the two layers' states as when other
# layers share the entry; 4 KiB past a power of two. Arbitrary otherwise.
PITCH_BYTES = (64 << 20) + 4096
NUM_SLOTS = 136
SENTINEL = 0x5A
_SM_MAJOR = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
_CUDA_MAJOR = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
# The SSM state dtype FlashInfer prefill runs on by default: fp32 on SM90, bf16
# on SM100 (CUDA 13 or newer); None where it is not the default.
_FLASHINFER_SSM_DTYPE = {9: torch.float32, 10: torch.bfloat16}.get(_SM_MAJOR)
if _SM_MAJOR == 10 and _CUDA_MAJOR < 13:
    _FLASHINFER_SSM_DTYPE = None
CHUNK = 64
# Two mixed-length chunks of six requests: fresh requests, cached prefixes, and
# lengths around the 64-token chunk of the delta rule. The last element maps a
# tracked request to the tokens past its prefix that its track slot snapshots:
# a chunk multiple takes the final state, any other length a chunk checkpoint.
FIRST_CHUNK = (
    [1, 37, 64, 129, 300, 7],
    [0, 0, 48, 0, 1000, 5],
    {2: 64, 3: 129, 4: 300},
)
SECOND_CHUNK = (
    [5, 64, 1, 200, 33, 70],
    [1, 37, 112, 129, 1300, 12],
    {1: 64, 3: 200, 5: 70},
)


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

    def __init__(self, slots, ssm_dtype):
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
                dtype=ssm_dtype,
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


def _unified_slot_ids():
    """The state allocator's id space: virtual slot v sits at physical slot
    NUM_SLOTS - v; slot 0 is the sink in both."""
    ids = UnitIds(
        num_virtual_ids=NUM_SLOTS,
        num_physical_units=NUM_SLOTS,
        unit_tokens=1,
        shards_under_dcp=False,
        device="cuda",
    )
    ids.reset()
    ids.virtual_to_physical[1:NUM_SLOTS] = torch.arange(NUM_SLOTS - 1, 0, -1)
    return ids.slot_space(("state", None))


def _backend(caches, slot_of_request, prefill_backend, state_ids):
    backend = GDNAttnBackend.__new__(GDNAttnBackend)
    backend.state_pool = None
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda req_pool_indices: slot_of_request[req_pool_indices]
    )
    backend._state_ids = state_ids
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
    backend._mamba_chunk_size = CHUNK
    backend.conv_states_shape = caches[0].conv[0].shape
    backend.accept_lens_pool = None
    backend.prefill_graph_metadata = None
    backend.kernel_dispatcher = GDNKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        prefill_backend,
        LinearAttnKernelBackend.TRITON,
    )
    return backend


def _batch(extend_lens, prefix_lens, track, track_slots):
    """A tracked extend batch; ``track_slots`` (int64, as the scheduler hands
    them) holds each request's track slot."""
    bs = len(extend_lens)
    starts = [sum(extend_lens[:i]) for i in range(bs)]
    mask = [row in track for row in range(bs)]
    track_lens = [prefix_lens[row] + track.get(row, 0) for row in range(bs)]
    batch = SimpleNamespace(
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
        mamba_track_mask=torch.tensor(mask, device="cuda"),
        mamba_track_indices=track_slots,
        mamba_track_seqlens=torch.tensor(track_lens, device="cuda"),
        mamba_prefill_track_mask_cpu=mask,
        mamba_track_seqlens_cpu=track_lens,
    )

    def aligned_lens():
        lens = batch.mamba_track_seqlens - batch.extend_prefix_lens
        return (lens // CHUNK) * CHUNK

    batch.mamba_track_aligned_lens = aligned_lens
    return batch


def _kernel_counts(fn):
    """``fn()`` and the names of the CUDA kernels it launched, counted."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        out = fn()
        torch.cuda.synchronize()
    names = (
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )
    return out, collections.Counter(names)


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestGDNPrefillStridedPool(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(_mamba_cache_chunk_size=CHUNK)
        override.install()
        self.addCleanup(override.restore)
        g = torch.Generator(device="cuda").manual_seed(0)
        self.layers = [_layer(layer) for layer in LAYERS]
        num_requests = len(FIRST_CHUNK[0])
        # Distinct random slots, the highest slot among them; slot 0 is the sink.
        # One track slot per request follows them.
        slots = (
            torch.randperm(NUM_SLOTS - 2, generator=g, device="cuda")[
                : 2 * num_requests - 1
            ]
            + 1
        )
        self.slots = torch.cat(
            [slots[: num_requests - 1], torch.tensor([NUM_SLOTS - 1], device="cuda")]
        )[torch.randperm(num_requests, generator=g, device="cuda")].int()
        self.track_slots = slots[num_requests - 1 :]
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
        for lens, _, _ in (FIRST_CHUNK, SECOND_CHUNK):
            tokens = sum(lens)
            self.inputs.append(
                tuple(
                    torch.randn(tokens, width, generator=g, device="cuda").to(
                        torch.bfloat16
                    )
                    for width in (DIM, V_HEADS, V_HEADS)
                )
            )

    def _prefill(self, conv, ssm, prefill_backend, unified):
        """Both chunks through both layers; the outputs of every forward, and the
        kernels the forwards launched. ``unified`` names slots by virtual ids, as
        the unified pool's requests do; otherwise by the pool's own."""
        caches = [
            SimpleNamespace(conv=[conv[layer]], temporal=ssm[layer]) for layer in LAYERS
        ]
        if unified:
            state_ids = _unified_slot_ids()
            slot_of_request = NUM_SLOTS - self.slots
            track_slots = NUM_SLOTS - self.track_slots
        else:
            state_ids = IdSpace(key=("state", None))
            slot_of_request, track_slots = self.slots, self.track_slots
        outputs, kernels = [], collections.Counter()
        for (lens, prefixes, track), (mixed_qkv, a, b) in zip(
            (FIRST_CHUNK, SECOND_CHUNK), self.inputs
        ):
            backend = _backend(caches, slot_of_request, prefill_backend, state_ids)
            batch = _batch(lens, prefixes, track, track_slots.clone())
            backend.init_forward_metadata(batch)
            for layer in self.layers:
                output, launched = _kernel_counts(
                    lambda: backend.forward_extend(
                        layer, batch, mixed_qkv.clone(), a.clone(), b.clone()
                    )
                )
                outputs.append(output)
                kernels += launched
        return outputs, kernels

    def _assert_same(self, outputs, conv, ssm, ref_outputs, ref_conv, ref_ssm):
        slots = torch.cat([self.slots.long(), self.track_slots])
        for output, ref_output in zip(outputs, ref_outputs, strict=True):
            self.assertTrue(torch.equal(output, ref_output), "outputs differ")
        for layer in LAYERS:
            self.assertTrue(
                torch.equal(conv[layer][slots], ref_conv[layer][slots]),
                "conv states differ",
            )
            self.assertTrue(
                torch.equal(ssm[layer][slots], ref_ssm[layer][slots]),
                "ssm states differ",
            )

    def _check_in_place_and_staged(self, prefill_backend, kernel_cls, ssm_dtype):
        arena = _Arena(torch.cat([self.slots.long(), self.track_slots]), ssm_dtype)
        self.assertFalse(arena.ssm[0].is_contiguous())
        self.assertGreater(
            int(self.slots.max()) * arena.ssm[0].stride(0), torch.iinfo(torch.int32).max
        )
        start_ssm = self.ssm.to(ssm_dtype)
        # Each run once first, so neither counts compile or autotune launches.
        self._prefill(self.conv.clone(), start_ssm.clone(), prefill_backend, False)
        arena.fill(self.conv, start_ssm)
        self._prefill(arena.conv, arena.ssm, prefill_backend, True)
        ref_conv, ref_ssm = self.conv.clone(), start_ssm.clone()
        ref_outputs, ref_kernels = self._prefill(
            ref_conv, ref_ssm, prefill_backend, False
        )

        arena.fill(self.conv, start_ssm)
        with patch.object(
            gdn_backend,
            "stage_pool_states",
            side_effect=AssertionError("staged a pool the kernels can address"),
        ):
            outputs, kernels = self._prefill(
                arena.conv, arena.ssm, prefill_backend, True
            )
        self._assert_same(
            outputs, arena.conv, arena.ssm, ref_outputs, ref_conv, ref_ssm
        )
        self.assertTrue(arena.bytes_outside_slots_untouched())
        # Per-layer work, such as an index cast in a track write, repeats in
        # every layer of a model; in place it must match the packed pool's.
        extra = kernels - ref_kernels
        self.assertFalse(extra, f"kernels past the packed pool's: {extra}")

        # A prefill kernel that cannot follow the pitch runs on packed copies.
        arena.fill(self.conv, start_ssm)
        with (
            patch.object(kernel_cls, "extend_honors_slot_pitch", False),
            patch.object(
                gdn_backend, "stage_pool_states", wraps=stage_pool_states
            ) as stage,
            patch.object(
                PoolStaging, "from_indices", wraps=PoolStaging.from_indices
            ) as plan,
        ):
            outputs, _ = self._prefill(arena.conv, arena.ssm, prefill_backend, True)
        # Two forwards of two layers each; one set of rows per forward.
        self.assertEqual(stage.call_count, 4)
        self.assertEqual(plan.call_count, 2)
        self._assert_same(
            outputs, arena.conv, arena.ssm, ref_outputs, ref_conv, ref_ssm
        )
        self.assertTrue(arena.bytes_outside_slots_untouched())

    def test_triton_prefill_in_place_and_staged_match_the_packed_pool(self):
        self._check_in_place_and_staged(
            LinearAttnKernelBackend.TRITON, TritonGDNKernel, torch.float32
        )

    @unittest.skipUnless(
        _FLASHINFER_SSM_DTYPE is not None and is_flashinfer_gdn_prefill_available(),
        "FlashInfer GDN prefill needs SM90, or SM100 with CUDA 13",
    )
    def test_flashinfer_prefill_in_place_and_staged_match_the_packed_pool(self):
        """FlashInfer prefill gathers and scatters states by pool index, so it
        runs on the strided pool without packed copies."""
        self._check_in_place_and_staged(
            LinearAttnKernelBackend.FLASHINFER,
            FlashInferGDNKernel,
            _FLASHINFER_SSM_DTYPE,
        )


if __name__ == "__main__":
    unittest.main()
