"""Weekly layout contract for LoRAMemoryPool buffer shapes.

`get_lora_A_shape` / `get_lora_B_shape` must size each per-rank buffer the way
the base layer is actually sharded: routed experts by `moe_tp_size`, attention
projections by `attn_tp_size`, dense and shared-expert projections by the base
linear's real partition; and `load_lora_weight_to_buffer` must hand the MoE-TP
rank to the expert slicers. A wrong size trips the load-time
`buffer_view.shape == weight.shape` assert in `load_lora_weight_to_buffer`,
but only for a geometry that is actually launched, so the supported geometries
are pinned here at weekly cadence.

Usage:
    python3 test/registered/unit/lora/test_mem_pool_layouts.py
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, stage="weekly", runner_config="cpu")

import types
import unittest
import unittest.mock as mock
from typing import Dict, Optional, Tuple

import torch


class _FakeBaseLayerWithLoRA:
    pass


class _FakeFusedMoEWithLoRA(_FakeBaseLayerWithLoRA):
    pass


# Same stub as test_mem_pool_ep_unit.py: `sglang.srt.lora.layers` pulls in the
# LoRA kernels, so stand in the three names the pool reads from it.
_LORA_LAYERS_STUB = types.ModuleType("sglang.srt.lora.layers")
_LORA_LAYERS_STUB.BaseLayerWithLoRA = _FakeBaseLayerWithLoRA
_LORA_LAYERS_STUB.FusedMoEWithLoRA = _FakeFusedMoEWithLoRA
_LORA_LAYERS_STUB.unwrap_lora_layer = lambda module: (
    module.base_layer if isinstance(module, _FakeBaseLayerWithLoRA) else module
)
_LORA_ADAPTER_STUB = types.ModuleType("sglang.srt.lora.lora")
_LORA_ADAPTER_STUB.LoRAAdapter = object
with mock.patch.dict(
    "sys.modules",
    {
        "sglang.srt.lora.layers": _LORA_LAYERS_STUB,
        "sglang.srt.lora.lora": _LORA_ADAPTER_STUB,
    },
):
    from sglang.srt.lora.mem_pool import LoRAMemoryPool

_HF_CONFIG = dict(
    hidden_size=64,
    num_attention_heads=8,
    num_key_value_heads=8,
    head_dim=8,
    intermediate_size=256,
    moe_intermediate_size=192,
)
# DeepSeek-style geometry for the base-linear probe: layer 0 is dense (MLP
# width 256), layer 1 is MoE with one shared expert of width 32.
_PROBE_CONFIG = dict(
    _HF_CONFIG,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    moe_intermediate_size=32,
    n_shared_experts=1,
)

_Shapes = Dict[Tuple[str, str], Tuple[int, ...]]


def _make_pool(
    *,
    tp_size: int,
    attn_tp_size: int,
    moe_tp_size: int,
    moe_ep_size: int = 1,
    num_experts: int = 1,
    max_loras: int = 2,
    hf_config: Optional[dict] = None,
) -> LoRAMemoryPool:
    """Pool with only the fields the shape pickers read (`__init__` needs a
    real model); the EP fields follow the `__init__` rule."""
    pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
    pool.max_loras_per_batch = max_loras
    pool.tp_size = tp_size
    pool.tp_rank = 0
    pool.attn_tp_size = attn_tp_size
    pool.moe_tp_size = moe_tp_size
    pool.moe_tp_rank = 0
    pool.moe_ep_size = moe_ep_size
    pool.moe_ep_rank = 0
    pool.moe_use_local_expert_ids = moe_ep_size > 1 and num_experts % moe_ep_size == 0
    pool._num_experts_local = (
        num_experts // moe_ep_size if pool.moe_use_local_expert_ids else num_experts
    )
    pool.experts_shared_outer_loras = False
    pool.base_hf_config = types.SimpleNamespace(**(hf_config or _HF_CONFIG))
    return pool


def _shapes(pool: LoRAMemoryPool, model, expected: _Shapes, layer_idx=0) -> _Shapes:
    """Pick every `(side, module)` key of ``expected`` at LoRA rank 8."""
    pick = {"A": pool.get_lora_A_shape, "B": pool.get_lora_B_shape}
    return {
        (side, module): pick[side](module, model, 8, layer_idx)
        for side, module in expected
    }


class _HiddenDimModel(torch.nn.Module):
    """Base model exposing `get_hidden_dim` (the per-model override path)."""

    def __init__(self, num_experts: int):
        super().__init__()
        self.config = types.SimpleNamespace(num_experts=num_experts, **_HF_CONFIG)

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        cfg = self.config
        if module_name == "qkv_proj":
            return cfg.hidden_size, cfg.head_dim * (
                cfg.num_attention_heads + cfg.num_key_value_heads * 2
            )
        if module_name == "o_proj":
            return cfg.head_dim * cfg.num_attention_heads, cfg.hidden_size
        if module_name == "gate_up_proj_moe":
            return cfg.hidden_size, cfg.moe_intermediate_size * 2
        if module_name == "down_proj_moe":
            return cfg.moe_intermediate_size, cfg.hidden_size
        if module_name == "in_proj_qkvz":
            # linear-attention packed input projection (column-parallel)
            return cfg.hidden_size, 4 * cfg.hidden_size
        raise NotImplementedError(module_name)


class _FakeRowLinear(torch.nn.Module):
    """RowParallelLinear stand-in exposing only the shard bookkeeping the pool reads."""

    def __init__(self, input_size_per_partition: int):
        super().__init__()
        self.input_size_per_partition = input_size_per_partition


class _FakeColLinear(torch.nn.Module):
    """MergedColumnParallelLinear stand-in exposing only the shard bookkeeping the pool reads."""

    def __init__(self, output_partition_sizes: Tuple[int, ...]):
        super().__init__()
        self.output_partition_sizes = list(output_partition_sizes)


def _probe_model(
    down_partitions: Tuple[int, int],
    gate_up_partitions: Tuple[Tuple[int, ...], Tuple[int, ...]],
) -> torch.nn.Module:
    """Layer 0 = dense MLP, layer 1 = MoE shared experts; each `down_proj` /
    `gate_up_proj` carries the per-rank partition the pool probes. No
    `get_hidden_dim`, so the config-driven widths are the fallback."""
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for shared, down, gate_up in zip(
        (False, True), down_partitions, gate_up_partitions
    ):
        mlp = torch.nn.Module()
        mlp.down_proj = _FakeRowLinear(down)
        mlp.gate_up_proj = _FakeColLinear(gate_up)
        layer = torch.nn.Module()
        if shared:
            layer.mlp = torch.nn.Module()
            layer.mlp.shared_experts = mlp
        else:
            layer.mlp = mlp
        model.model.layers.append(layer)
    return model


class TestShardingGroups(unittest.TestCase):
    """Routed experts shard by the MoE-TP group and attention by the
    attention-TP group; neither is the outer `tp_size` once EP or attention
    DP is on. 2 slots, LoRA rank 8, hidden 64, MoE inner 192."""

    def test_routed_experts_shard_by_moe_tp_not_outer_tp(self):
        # `--tp 4 --ep 4` leaves moe_tp=1 (full-width experts on every rank);
        # `--tp 8 --ep 4` leaves moe_tp=2. 128 experts -> 32 local per rank.
        # Attention and other dense modules keep sharding by the outer tp.
        model = _HiddenDimModel(num_experts=128)
        cases = {
            (4, 1): {
                ("A", "down_proj_moe"): (2, 32, 8, 192),
                ("B", "down_proj_moe"): (2, 32, 64, 8),
                ("A", "gate_up_proj_moe"): (2, 32, 16, 64),
                ("B", "gate_up_proj_moe"): (2, 32, 384, 8),
                ("A", "o_proj"): (2, 8, 16),
                ("B", "o_proj"): (2, 64, 8),
                ("B", "qkv_proj"): (2, 48, 8),
            },
            (8, 2): {
                ("A", "down_proj_moe"): (2, 32, 8, 96),
                ("B", "down_proj_moe"): (2, 32, 64, 8),
                ("A", "gate_up_proj_moe"): (2, 32, 16, 64),
                ("B", "gate_up_proj_moe"): (2, 32, 192, 8),
                ("A", "o_proj"): (2, 8, 8),
                ("B", "o_proj"): (2, 64, 8),
                ("B", "qkv_proj"): (2, 24, 8),
            },
        }
        for (tp_size, moe_tp_size), expected in cases.items():
            with self.subTest(tp_size=tp_size, moe_tp_size=moe_tp_size):
                pool = _make_pool(
                    tp_size=tp_size,
                    attn_tp_size=tp_size,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=4,
                    num_experts=128,
                )
                self.assertEqual(_shapes(pool, model, expected), expected)

    def test_attention_shards_by_attn_tp(self):
        # `--tp 4 --attn-dp-size 4` replicates attention (attn_tp=1);
        # `--attn-dp-size 2` shards it 2-way. in_proj_qkvz (linear attention)
        # is built on the attention-TP group as well.
        model = _HiddenDimModel(num_experts=1)
        cases = {
            1: {
                ("A", "o_proj"): (2, 8, 64),
                ("B", "qkv_proj"): (2, 192, 8),
                ("B", "in_proj_qkvz"): (2, 256, 8),
            },
            2: {
                ("A", "o_proj"): (2, 8, 32),
                ("B", "qkv_proj"): (2, 96, 8),
                ("B", "in_proj_qkvz"): (2, 128, 8),
            },
        }
        for attn_tp_size, expected in cases.items():
            with self.subTest(attn_tp_size=attn_tp_size):
                pool = _make_pool(tp_size=4, attn_tp_size=attn_tp_size, moe_tp_size=4)
                self.assertEqual(_shapes(pool, model, expected), expected)


class TestDenseProjectionsProbeTheBaseLinear(unittest.TestCase):
    """Dense and shared-expert `down_proj` (row-parallel, LoRA-A input) and
    `gate_up_proj` (column-parallel, LoRA-B output) take their per-rank
    width from the base linear: under `--attn-dp-size 32 --moe-dense-tp-size 1`
    they are replicated although the outer tp is 32. 1 slot, LoRA rank 8."""

    @staticmethod
    def _pool(**config_overrides) -> LoRAMemoryPool:
        return _make_pool(
            tp_size=32,
            attn_tp_size=1,
            moe_tp_size=1,
            moe_ep_size=32,
            max_loras=1,
            hf_config={**_PROBE_CONFIG, **config_overrides},
        )

    @staticmethod
    def _dense_shapes(pool, model, layer_idx):
        """(down_proj LoRA-A shape, gate_up_proj LoRA-B shape) at a layer."""
        return (
            pool.get_lora_A_shape("down_proj", model, 8, layer_idx),
            pool.get_lora_B_shape("gate_up_proj", model, 8, layer_idx),
        )

    def test_partition_widths(self):
        # Per layer (dense, shared): the base linear's down_proj input
        # partition and gate_up_proj output partitions -> expected shapes.
        cases = (
            (
                "replicated",
                (256, 32),
                ((256, 256), (32, 32)),
                (((1, 8, 256), (1, 512, 8)), ((1, 8, 32), (1, 64, 8))),
            ),
            (
                "sharded_8_way",
                (32, 4),
                ((32, 32), (4, 4)),
                (((1, 8, 32), (1, 64, 8)), ((1, 8, 4), (1, 8, 8))),
            ),
            # Only the gate/up slices count toward the LoRA-B width; a third
            # non-LoRA output slice is ignored.
            (
                "extra_non_lora_slice",
                (256, 32),
                ((256, 256, 64), (32, 32, 8)),
                (((1, 8, 256), (1, 512, 8)), ((1, 8, 32), (1, 64, 8))),
            ),
        )
        for name, down, gate_up, expected in cases:
            model = _probe_model(down, gate_up)
            pool = self._pool()
            for layer_idx, want in enumerate(expected):
                with self.subTest(case=name, layer_idx=layer_idx):
                    self.assertEqual(self._dense_shapes(pool, model, layer_idx), want)

    def test_base_linear_width_wins_over_the_config(self):
        # With `first_k_dense_replace` unset the config calls layer 1 dense
        # (width 256) and tp=1 would make that full width; the probed 32 wins.
        model = _probe_model((256, 32), ((256, 256), (32, 32)))
        for tp_size in (32, 1):
            with self.subTest(tp_size=tp_size):
                pool = self._pool(first_k_dense_replace=None)
                pool.tp_size = tp_size
                self.assertEqual(
                    self._dense_shapes(pool, model, 1), ((1, 8, 32), (1, 64, 8))
                )

    def test_probe_miss_falls_back_to_the_outer_tp(self):
        self.assertEqual(
            self._dense_shapes(self._pool(), torch.nn.Module(), 0),
            ((1, 8, 256 // 32), (1, 512 // 32, 8)),
        )

    def test_attention_projections_ignore_the_probe(self):
        model = _probe_model((256, 32), ((256, 256), (32, 32)))
        pool = self._pool()
        pool.attn_tp_size = 4
        self.assertEqual(pool.get_lora_A_shape("o_proj", model, 8, 0), (1, 8, 16))
        self.assertEqual(pool.get_lora_B_shape("qkv_proj", model, 8, 0), (1, 48, 8))


class _FakeRoutedMoeLayer(_FakeFusedMoEWithLoRA):
    """Routed `FusedMoEWithLoRA` stand-in; the test installs the slicers."""

    def __init__(self, tp_rank: int):
        self.base_layer = types.SimpleNamespace(
            moe_tp_rank=tp_rank, is_shared_fused_moe=False
        )


def _load_lora_weight_to_buffer(pool, **kwargs):
    with mock.patch.dict("sys.modules", {"sglang.srt.lora.layers": _LORA_LAYERS_STUB}):
        return pool.load_lora_weight_to_buffer(**kwargs)


class TestLoadBufferPassesMoeTpRankToSlice(unittest.TestCase):
    """`load_lora_weight_to_buffer` must hand `moe_tp_rank`, not the outer
    `tp_rank`, to `slice_moe_lora_{a,b}_weights`.

    The two differ only when `1 < ep_size < tp_size`: at `tp=4 ep=2` outer
    rank 3 is moe_tp_rank 1 of 2, and slicing by rank 3 starts at
    `3 * moe_inter / 2`, past the weight, so the `[r, 0]` slice fails the
    shape assert in `load_lora_weight_tensor`. The mocked slicer captures the
    rank and raises before the buffer-copy phase would need real shapes.
    """

    class _StopAfterCapture(Exception):
        """Raised from the mocked slicer to short-circuit before the copy."""

    def test_moe_tp_rank_used_for_slicing_when_ep_lt_tp(self):
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.tp_size = 4
        pool.tp_rank = 3
        pool.moe_tp_size = 2
        pool.moe_tp_rank = 1
        pool.moe_ep_size = 2
        pool.moe_ep_rank = 1
        pool.moe_use_local_expert_ids = True
        pool._num_experts_local = 1
        pool.num_layer = 1
        pool.target_modules = {"gate_up_proj", "down_proj"}
        pool.experts_shared_outer_loras = False
        pool.strict_loading = False
        pool.lora_added_tokens_size = 0
        # Tiny placeholder buffers; the mocked slicer raises before any of
        # this is read in the buffer-copy phase.
        pool.A_buffer = {
            "gate_up_proj_moe": [torch.zeros(1, 1, 1, 1)],
            "down_proj_moe": [torch.zeros(1, 1, 1, 1)],
        }
        pool.B_buffer = {
            "gate_up_proj_moe": [torch.zeros(1, 1, 1, 1)],
            "down_proj_moe": [torch.zeros(1, 1, 1, 1)],
        }
        pool.embedding_A_buffer = {}
        pool.embedding_B_buffer = {}
        pool.lm_head_A_buffer = {}
        pool.lm_head_B_buffer = {}
        pool.new_embeddings_buffer = {}

        moe_mod = _FakeRoutedMoeLayer(tp_rank=1)

        for ab in ("A", "B"):
            with self.subTest(ab=ab):
                captured_ranks = []

                def capture(weights, tp_rank, target_module):
                    captured_ranks.append((ab, target_module, tp_rank))
                    raise TestLoadBufferPassesMoeTpRankToSlice._StopAfterCapture()

                moe_mod.slice_moe_lora_a_weights = (
                    capture if ab == "A" else lambda weights, *_args: weights
                )
                moe_mod.slice_moe_lora_b_weights = (
                    capture if ab == "B" else lambda weights, *_args: weights
                )

                # The per-expert key creates the A or B dictionary whose
                # production call site invokes the matching MoE slicer.
                adapter = mock.MagicMock()
                adapter.config.r = 4
                adapter.scaling = 1.0
                adapter.embedding_layers = {}
                adapter.added_tokens_embeddings = {}
                adapter.layers = [
                    types.SimpleNamespace(
                        weights={
                            f"model.layers.0.mlp.experts.0.gate_up_proj.lora_{ab}.weight": torch.zeros(
                                8, 4
                            ),
                        },
                        pinned_weights={},
                    )
                ]

                with self.assertRaises(
                    TestLoadBufferPassesMoeTpRankToSlice._StopAfterCapture
                ):
                    _load_lora_weight_to_buffer(
                        pool,
                        uid="test",
                        buffer_id=0,
                        lora_adapter=adapter,
                        lora_modules=[{"mlp.experts": moe_mod}],
                        lora_embed_tokens_module=None,
                        lora_lm_head_module=None,
                    )

                self.assertEqual(len(captured_ranks), 1, "slicing was never invoked")
                _, target_module, rank = captured_ranks[0]
                self.assertEqual(
                    rank,
                    pool.moe_tp_rank,
                    f"slice_moe_lora_{ab.lower()}_weights for {target_module} "
                    f"received rank={rank}; expected moe_tp_rank="
                    f"{pool.moe_tp_rank} (outer tp_rank is {pool.tp_rank}). "
                    "Passing the outer tp_rank slices past "
                    "intermediate_size_per_partition when ep_size < tp_size.",
                )


if __name__ == "__main__":
    unittest.main()
