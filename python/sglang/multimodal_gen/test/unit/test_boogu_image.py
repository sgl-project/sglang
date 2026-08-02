# SPDX-License-Identifier: Apache-2.0
"""Boogu-Image packed-QKV weight remapping.

The checkpoint stores q, k and v as three separate projections; the model fuses
each triple into one `MergedColumnParallelLinear`. The fusion is only correct if
every triple lands in the same packed tensor in q, k, v order -- a swapped k/v
loads without error and silently produces wrong images, so these assert the
order explicitly rather than just the shapes.
"""

import unittest

import torch

from sglang.multimodal_gen.configs.models.dits.boogu_image import BooguImageArchConfig
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)

DIM = 4
KV_DIM = 2

# (checkpoint projection prefix, fused parameter name)
QKV_FAMILIES = [
    ("layers.0.attention.to_", "layers.0.attention.to_qkv.weight"),
    ("layers.0.attention.img_to_", "layers.0.attention.img_to_qkv.weight"),
    (
        "layers.0.attention.instruct_to_",
        "layers.0.attention.instruct_to_qkv.weight",
    ),
]


def boogu_mapping():
    return get_param_names_mapping(BooguImageArchConfig().param_names_mapping)


def qkv_weights(prefix: str):
    """q / k / v filled with distinct values so concat order is observable."""
    return [
        (f"{prefix}q.weight", torch.full((DIM, DIM), 1.0)),
        (f"{prefix}k.weight", torch.full((KV_DIM, DIM), 2.0)),
        (f"{prefix}v.weight", torch.full((KV_DIM, DIM), 3.0)),
    ]


class TestBooguImagePackedQkvMapping(unittest.TestCase):

    def test_each_family_concatenates_q_k_v_in_order(self):
        for prefix, fused in QKV_FAMILIES:
            with self.subTest(prefix=prefix):
                mapped, _ = hf_to_custom_state_dict(
                    iter(qkv_weights(prefix)), boogu_mapping()
                )

                self.assertIn(fused, mapped)
                torch.testing.assert_close(
                    mapped[fused],
                    torch.cat(
                        [
                            torch.full((DIM, DIM), 1.0),
                            torch.full((KV_DIM, DIM), 2.0),
                            torch.full((KV_DIM, DIM), 3.0),
                        ],
                        dim=0,
                    ),
                )

    def test_plain_to_q_rule_does_not_capture_joint_projections(self):
        """`.to_q` must not match `.img_to_q` / `.instruct_to_q`.

        The three families are kept disjoint by the underscore before `to_q`. A
        future diff that loosens the rule to `(.*)to_q\\.weight$` would collapse
        all three into one packed tensor, which this catches.
        """
        mapping = boogu_mapping()
        for prefix, fused in QKV_FAMILIES:
            for role in ("q", "k", "v"):
                with self.subTest(prefix=prefix, role=role):
                    name, shard, total = mapping(f"{prefix}{role}.weight")
                    self.assertEqual(name, fused)
                    self.assertEqual(shard, "qkv".index(role))
                    self.assertEqual(total, 3)

    def test_feed_forward_fusion_is_unaffected(self):
        """The qkv rules must not disturb the pre-existing w13 gate/up fusion."""
        mapping = boogu_mapping()
        for role, shard in (("linear_1", 0), ("linear_3", 1)):
            with self.subTest(role=role):
                name, got_shard, total = mapping(f"layers.0.feed_forward.{role}.weight")
                self.assertEqual(name, "layers.0.feed_forward.w13.weight")
                self.assertEqual(got_shard, shard)
                self.assertEqual(total, 2)

        name, shard, total = mapping("layers.0.feed_forward.linear_2.weight")
        self.assertEqual(name, "layers.0.feed_forward.w2.weight")
        self.assertIsNone(shard)


@unittest.skipUnless(
    torch.cuda.is_available(),
    "importing the DiT module autotunes triton kernels, which needs a device",
)
class TestBooguImageQkvSplitSizes(unittest.TestCase):

    def test_split_sizes_partition_the_packed_projection(self):
        """Output-side splits must match the widths the packed linear produces."""
        from sglang.multimodal_gen.runtime.models.dits.boogu_image import (
            qkv_split_sizes,
        )

        config = BooguImageArchConfig()
        head_dim = config.dim // config.num_attention_heads
        sizes = qkv_split_sizes(
            local_num_heads=config.num_attention_heads,
            local_num_kv_heads=config.n_kv_heads,
            head_dim=head_dim,
        )

        kv_dim = config.n_kv_heads * head_dim
        self.assertEqual(sizes, [config.dim, kv_dim, kv_dim])
        self.assertEqual(sum(sizes), config.dim + 2 * kv_dim)

        packed = torch.cat(
            [torch.full((3, width), float(i)) for i, width in enumerate(sizes)], dim=-1
        )
        for i, part in enumerate(packed.split(sizes, dim=-1)):
            self.assertEqual(part.shape[-1], sizes[i])
            torch.testing.assert_close(part, torch.full((3, sizes[i]), float(i)))


if __name__ == "__main__":
    unittest.main()
