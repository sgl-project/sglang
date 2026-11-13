"""CPU coverage for image-DP across Kimi's prefill-CP peers."""

import unittest
from contextlib import nullcontext
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.models import kimi_k3
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Tower:
    device = torch.device("cpu")
    merge_kernel_size = (1, 1)
    config = SimpleNamespace(hidden_size=1)
    patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=torch.empty(1)))

    def __init__(self):
        self.calls = 0

    def __call__(self, pixels, **_kwargs):
        self.calls += 1
        return pixels.reshape(-1, 1, 1)


class _ImageGroup:
    world_size = 2
    use_symmetric_memory = staticmethod(lambda *_args, **_kwargs: nullcontext())
    is_allocation_symmetric = staticmethod(lambda: False)

    def __init__(self, rank, features, shards):
        self.rank_in_group = rank
        self.features = features
        self.shards = shards
        self.collectives = 0

    def broadcast(self, tensor, src):
        assert src == 0
        tensor.copy_(self.features[0].reshape(-1, 1, 1))
        self.collectives += 1

    def all_gather_into_tensor(self, output, tensor):
        packed = []
        for rank, indices in enumerate(self.shards):
            expected = torch.cat([self.features[i] for i in indices]).reshape(-1, 1, 1)
            if rank == self.rank_in_group:
                torch.testing.assert_close(tensor[: len(expected)], expected)
            padded = torch.zeros_like(tensor)
            padded[: len(expected)] = expected
            packed.append(padded)
        self.collectives += 1
        output.copy_(torch.cat(packed, dim=0))


class TestKimiK3PCPVisionDP(CustomTestCase):
    def test_pcp_image_group_encodes_each_image_once_and_restores_order(self):
        cases = [([4], [[0], []]), ([4, 4], [[0], [1]]), ([4, 2, 1], [[0], [1, 2]])]
        for rank, (sizes, shards) in product(range(2), cases):
            with self.subTest(rank=rank, sizes=sizes):
                features = [
                    torch.full((n, 1), float(i + 1)) for i, n in enumerate(sizes)
                ]
                group = _ImageGroup(rank, features, shards)
                tower = _Tower()
                loaded = []

                def load(indices):
                    loaded.extend(indices)
                    return torch.cat([features[i] for i in indices])

                # Each CP peer has singleton attention TP, but shares this image-DP group.
                with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
                    output = run_dp_sharded_mrope_vision_model(
                        tower,
                        None,
                        [[1, 1, n] for n in sizes],
                        rope_type="rope_2d",
                        load_local_pixel_values=load,
                        pixel_values_device=torch.device("cpu"),
                        pixel_values_dtype=torch.float32,
                        parallel_group=group,
                    )

                assert loaded == shards[rank]
                assert tower.calls == bool(shards[rank])
                assert group.collectives == 1
                torch.testing.assert_close(
                    output, torch.cat(features).reshape(-1, 1, 1)
                )

    def test_kimi_only_selects_full_tp_for_replicated_cp_images(self):
        group = object()
        for cp, dp, use_tp_group in [(2, 1, True), (1, 1, False), (2, 2, False)]:
            with (
                self.subTest(cp=cp, dp=dp),
                patch.object(
                    kimi_k3,
                    "get_parallel",
                    return_value=SimpleNamespace(
                        attn_cp_size=cp, attn_dp_size=dp, tp_group=group
                    ),
                ),
            ):
                model = kimi_k3.KimiK3ForConditionalGeneration.__new__(
                    kimi_k3.KimiK3ForConditionalGeneration
                )
                torch.nn.Module.__init__(model)
                model.use_data_parallel = True
                model.vision_tower = _Tower()
                model.mm_projector = torch.nn.Identity()
                expected = torch.ones(1, 1)

                def run_dp(*_args, parallel_group, **_kwargs):
                    assert parallel_group is (group if use_tp_group else None)
                    return expected

                item = SimpleNamespace(
                    model_specific_data={"grid_thws": torch.tensor([[1, 1, 1]])}
                )
                with patch(
                    "sglang.srt.multimodal.mm_utils.run_dp_sharded_mrope_vision_model",
                    run_dp,
                ):
                    assert model.get_image_feature([item]) is expected

    def test_replicated_cache_cold_hot_partial_and_empty_chunks(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                cache = MultiModalStaticCache(1024)
                items = [
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        hash=i,
                        offsets=[(i * 4, i * 4 + 3)],
                        feature=torch.full((4, 1), float(i + 1)),
                    )
                    for i in range(2)
                ]
                request = mm_schedule.PerImageRequestInfo(
                    0, items, [(0, 3), (4, 7)], 0, 8
                )
                encoded = []

                def encode(misses):
                    encoded.append([item.hash for item in misses])
                    features = [item.feature for item in misses]
                    shards = [[0], []] if len(misses) == 1 else [[0], [1]]
                    return run_dp_sharded_mrope_vision_model(
                        _Tower(),
                        torch.cat(features),
                        [[1, 1, 4]] * len(misses),
                        rope_type="rope_2d",
                        parallel_group=_ImageGroup(rank, features, shards),
                    )

                with (
                    patch.object(mm_schedule, "embedding_cache", cache),
                    get_parallel().override(attn_tp_size=1, attn_tp_rank=0),
                ):
                    cold = mm_schedule._batch_encode_per_image_misses(
                        encode, [request], torch.device("cpu")
                    )
                    hot = mm_schedule._batch_encode_per_image_misses(
                        encode, [request], torch.device("cpu")
                    )
                    assert encoded == [[0, 1]]
                    assert cold.keys() == hot.keys()
                    for key in cold:
                        torch.testing.assert_close(cold[key], hot[key])
                    cache.free(items[0].cache_key, None)
                    partial = mm_schedule._batch_encode_per_image_misses(
                        encode, [request], torch.device("cpu")
                    )
                    assert encoded == [[0, 1], [0]]
                    for key in cold:
                        torch.testing.assert_close(cold[key], partial[key])
                    request.extend_seq_len = 0
                    assert (
                        mm_schedule._batch_encode_per_image_misses(
                            encode, [request], torch.device("cpu")
                        )
                        == {}
                    )
                    assert encoded == [[0, 1], [0]]


if __name__ == "__main__":
    unittest.main()
