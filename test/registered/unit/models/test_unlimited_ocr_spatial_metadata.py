import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models.unlimited_ocr_crop_transfer import (
    RetiredTransfer,
    UnlimitedOcrCropTransfer,
)
from sglang.srt.models.unlimited_ocr_v2 import (
    UnlimitedOCRV2Attention,
    UnlimitedOCRV2ForCausalLM,
)
from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnlimitedOCRRouterTP(CustomTestCase):
    def test_each_rank_loads_and_projects_the_full_router_head(self):
        """TP shards main attention heads, but every rank needs all 128 router dimensions."""
        self.addCleanup(reset_context)
        config = SimpleNamespace(
            hidden_size=1280,
            num_attention_heads=10,
            num_key_value_heads=10,
            rms_norm_eps=1e-6,
            sparse_attention_config={"sparse_index_dim": 128},
        )
        hidden = torch.arange(2 * 1280, dtype=torch.float32).reshape(2, 1280) / 1280
        weight = torch.arange(128 * 1280, dtype=torch.float32).reshape(128, 1280)
        weight = weight / weight.numel()
        expected = torch.nn.functional.linear(hidden, weight)
        for rank in range(2):
            with self.subTest(rank=rank):
                reset_context()
                publish(
                    ServerArgs(model_path="dummy", device="cpu", tp_size=2),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                attention = UnlimitedOCRV2Attention(config, layer_id=0)
                self.assertEqual(attention.q_proj.weight.shape, (640, 1280))
                for projection in (attention.index_q_proj, attention.index_k_proj):
                    projection.weight.weight_loader(projection.weight, weight)
                    actual, _ = projection(hidden)
                    self.assertEqual(actual.shape, (2, 128))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestUnlimitedOCRSpatialMetadata(CustomTestCase):
    def test_cpu_spatial_metadata_preserves_page_assembly(self):
        model = object.__new__(UnlimitedOCRV2ForCausalLM)
        torch.nn.Module.__init__(model)
        model.vision_model = SimpleNamespace(dtype=torch.bfloat16)
        model.crop_transfer = SimpleNamespace(
            stack=lambda tensors, dtype, device: torch.ones(1, 1, 2, 3, 4, 4)
        )
        seen = []

        def encode(**kwargs):
            seen.append(kwargs)
            return [torch.ones(2, 3)]

        model._pixel_values_to_embedding = encode
        items = [
            SimpleNamespace(
                feature=torch.ones(1, 3, 4, 4),
                images_crop=torch.ones(1, 2, 3, 4, 4),
                images_spatial_crop=torch.tensor([[[2, 1]]]),
                has_local_crops=True,
            )
        ]

        actual = model._process_image_input(items)

        self.assertEqual(actual.shape, (2, 3))
        self.assertEqual(len(seen), 1)
        spatial = seen[0]["images_spatial_crop"]
        self.assertEqual(spatial.device.type, "cpu")
        self.assertEqual(spatial.dtype, torch.long)
        self.assertEqual(spatial.tolist(), [[[2, 1]]])


class TestUnlimitedOCRPageSlices(CustomTestCase):
    def test_split_pages_pickle_only_their_own_crops(self):
        """Pickling a page's crop view used to serialize the whole document's
        crop tensor, so request bytes and TTFT grew with pages squared."""
        from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
        from sglang.srt.multimodal.processors.unlimited_ocr_v2 import (
            UnlimitedOCRV2Processor,
        )

        pages = 4
        bundle = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.zeros(pages, 3, 8, 8),
            offsets=[(10 * i, 10 * i + 5) for i in range(pages)],
        )
        bundle.set("images_crop", torch.rand(pages, 2, 3, 64, 64))
        bundle.set("images_spatial_crop", torch.ones(pages, 1, 2, dtype=torch.long))
        items = get_new_expanded_mm_items([bundle])
        self.assertEqual(len(items), pages)

        processor = object.__new__(UnlimitedOCRV2Processor)
        items = processor._postprocess_mm_items_before_transport(items, images=None)

        page_bytes = 2 * 3 * 64 * 64 * 4
        for page, item in enumerate(items):
            crop = item.model_specific_data["images_crop"]
            self.assertLess(len(pickle.dumps(crop)), 2 * page_bytes)
            self.assertTrue(torch.equal(crop, bundle.images_crop[page : page + 1]))


class _Transform:
    mean = (0.5, 0.5, 0.5)

    def __call__(self, image):
        return torch.zeros(1)


class TestUnlimitedOCRTileCap(CustomTestCase):
    def test_v2_tile_cap_does_not_reach_v1(self):
        """The tile cap takes part in aspect-ratio selection: v1 keeps its 32,
        v2 its checkpoint's 16, or a tall page is tiled differently."""
        from PIL import Image

        from sglang.srt.configs.unlimited_ocr import UnlimitedOCRHFProcessor

        processor = object.__new__(UnlimitedOCRHFProcessor)
        processor.image_transform = _Transform()
        page = Image.new("RGB", (900, 3000))

        def tiles(v2_layout):
            _, crops, (w, h) = processor._preprocess_page(
                page,
                cropping=True,
                base_size=1024,
                image_size=640,
                v2_layout=v2_layout,
            )
            self.assertEqual(len(crops), w * h)
            return w * h

        self.assertGreater(tiles(False), 16)
        self.assertLessEqual(tiles(True), 16)


class TestUnlimitedOCRCropTransfer(CustomTestCase):
    def _transfer(self, *, prefetch=True):
        with envs.SGLANG_ENABLE_UOCR_CROP_PREFETCH.override(prefetch):
            transfer = UnlimitedOcrCropTransfer()
        self.addCleanup(transfer.close)
        return transfer

    def test_prefetch_preserves_source_order_and_duplicates(self):
        """A reconstructed crop batch must retain source order and duplicate slots."""
        transfer = self._transfer()
        sources = [torch.full((2, 3), value) for value in (1.0, 2.0, 3.0)]
        for source in sources[:2]:
            self.assertTrue(transfer.submit(source, torch.bfloat16))
        ordered = [sources[1], sources[0], sources[2], sources[1]]

        actual = transfer.stack(ordered, torch.bfloat16, torch.device("cpu"))
        expected = torch.stack(ordered).bfloat16()

        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )

    def test_prefetch_failure_uses_native_fallback(self):
        """A staging failure must preserve the native crop conversion result."""
        transfer = self._transfer()
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        with (
            patch(
                "sglang.srt.models.unlimited_ocr_crop_transfer._stage",
                side_effect=[RuntimeError("injected"), source.bfloat16()],
            ),
            self.assertLogs(
                "sglang.srt.models.unlimited_ocr_crop_transfer", level="WARNING"
            ),
        ):
            self.assertTrue(transfer.submit(source, torch.bfloat16))
            actual = transfer.stack([source], torch.bfloat16, torch.device("cpu"))

        self.assertTrue(torch.equal(actual, source[None].bfloat16()))
        transfer.reap()
        self.assertEqual(transfer.reserved_items, 0)

    def test_instance_owned_transfer_state(self):
        first = self._transfer(prefetch=False)
        second = self._transfer(prefetch=False)
        source = torch.ones(2, 3)
        first.pinned_slots[("sentinel",)] = []

        actual = second.stack([source], torch.bfloat16, torch.device("cpu"))

        self.assertNotIn(("sentinel",), second.pinned_slots)
        self.assertTrue(torch.equal(actual, source[None].bfloat16()))

    def test_pinned_lifetime_waits_for_completion_event(self):
        """Pinned crop storage remains leased until its H2D event completes."""
        transfer = self._transfer()
        source = torch.ones(2, 3)
        self.assertTrue(transfer.submit(source, torch.bfloat16))
        entry = next(iter(transfer.entries.values()))
        prepared = entry.future.result(timeout=5)
        event = Mock()
        event.query.return_value = False
        entry.leases = 1
        transfer.retired.append(
            RetiredTransfer(event=event, buffers=[prepared], entries=[entry])
        )

        transfer.reap()
        self.assertEqual(entry.leases, 1)
        self.assertEqual(transfer.reserved_items, 1)
        event.query.return_value = True
        transfer.reap()
        self.assertEqual(entry.leases, 0)
        self.assertEqual(transfer.reserved_items, 1)


if __name__ == "__main__":
    unittest.main()
