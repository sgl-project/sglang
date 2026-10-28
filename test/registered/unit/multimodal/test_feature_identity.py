"""Default VLM identity contract, independent of any model-specific opt-in."""

import asyncio
import itertools
import pickle
import unittest
from array import array
from dataclasses import dataclass, replace
from test.registered.unit.mem_cache.test_rust_tree_core_integration import (
    _insert,
    _key,
    _pump_insert,
    _tree_core,
)
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
import numpy as np
import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_utils import get_new_expanded_mm_items, hash_feature_digest
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
    MultimodalProcessorOutput,
    Req,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.multimodal.cache.digest import structured_digest
from sglang.srt.multimodal.cache.identity import (
    build_artifact_key,
    media_preprocess_kwargs,
)
from sglang.srt.multimodal.media_artifacts.base import MediaArtifactInput
from sglang.srt.multimodal.processors.llava import (
    LlavaImageProcessor,
    LlavaMultimodalProcessor,
)
from sglang.srt.multimodal.processors.muse_spark_omni import (
    MuseSpark1xOmniMultimodalProcessor,
)
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.multimodal.processors.unlimited_ocr import UnlimitedOCRProcessor
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _item(value, **kwargs):
    return MultimodalDataItem(
        modality=Modality.IMAGE, feature=value, offsets=[(0, 0)], **kwargs
    )


class TestFeatureIdentity(CustomTestCase):
    def test_qwen_artifacts_keep_typed_identity_without_feature_storage(self):
        processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        processor._processor = SimpleNamespace(image_processor=Mock())
        processor._tokenizer = None
        processor.image_config = {}
        processor.mm_preprocess_cache = SimpleNamespace(enabled=True)
        processor.precompute_hash_before_cpu_transfer = False
        processor.mm_feature_transport = "cpu"
        processor.use_cuda_ipc = False
        processor.vision_start_token_id, processor.vision_end_token_id = 1, 3
        processor.mm_tokens = SimpleNamespace(
            image_token_id=2, video_token_id=4, audio_token_id=5
        )
        key_a, key_b = "sha256:" + "aa" * 32, "sha256:" + "bb" * 32
        entry = MediaArtifactInput(key_a, key_a, Modality.IMAGE, None)
        feature = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        artifacts = []
        for value, grid, key in (
            (feature, [1, 2, 2], key_a),
            (feature + 1, [1, 2, 2], key_a),
            (feature.to(torch.float64), [1, 2, 2], key_a),
            (feature.reshape(4, 1, 2), [1, 2, 2], key_a),
            (feature, [1, 1, 4], key_a),
            (feature, [1, 2, 2], key_b),
        ):
            processor._processor.image_processor.return_value = {
                "pixel_values": value,
                "image_grid_thw": torch.tensor([grid]),
            }
            artifacts.extend(
                processor.prepare_artifact_batch([replace(entry, artifact_key=key)])
            )
        self.assertEqual(len({a.feature_hash for a in artifacts}), len(artifacts))
        template = ([1, 2, 2, 2, 2, 3], [(1, 4)], torch.zeros(3, 6), torch.zeros(1))
        with (
            patch.object(
                processor, "_build_image_prompt_template", return_value=template
            ),
            patch.object(
                processor, "_mark_cuda_ipc_features_for_deferred_reconstruction"
            ),
            patch.object(
                MultimodalDataItem, "_input_identity", side_effect=AssertionError
            ),
        ):
            full = processor.compose_image_artifacts([1, 2, 3], artifacts[:1])
            cached = processor.compose_image_artifacts(
                [1, 2, 3], [replace(artifacts[0], feature=None)]
            )
        self.assertEqual(full.mm_items[0].cache_key, cached.mm_items[0].cache_key)
        self.assertEqual(len(full.mm_items[0].cache_key), 32)
        self.assertEqual(full.padded_input_ids, cached.padded_input_ids)

    def test_muse_artifact_identity_is_adopted_before_transport(self):
        processor = MuseSpark1xOmniMultimodalProcessor.__new__(
            MuseSpark1xOmniMultimodalProcessor
        )
        processor.preprocess_config = SimpleNamespace(use_gpu_preprocess=False)
        processor.mm_feature_transport = "cuda_ipc"
        processor.use_cuda_ipc = True
        processor.image_start_id, processor.image_end_id = 1, 3
        processor.patch_id, processor.video_id, processor.audio_id = 2, 4, 5
        processor.audio_start_id = processor.audio_end_id = None
        processor._tokenizer = Mock()
        processor._tokenizer.encode.return_value = [1, 2, 2, 3]
        key = "sha256:" + "aa" * 32
        entry = MediaArtifactInput(key, key, Modality.IMAGE, None)
        feature = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        artifacts = []
        for value in (feature, feature + 1, feature.to(torch.float64)):
            with patch(
                "sglang.srt.multimodal.processors.muse_spark_omni.preprocess_image",
                return_value=(value, 2),
            ):
                artifacts.extend(processor.prepare_artifact_batch([entry]))
        self.assertEqual(len({a.feature_hash for a in artifacts}), len(artifacts))
        processor.preprocess_config.use_gpu_preprocess = True
        processor.prepare_media_artifacts = AsyncMock(return_value=artifacts[:1])
        with (
            patch.object(
                processor, "_wrap_tensor_for_cuda_ipc", side_effect=lambda t: t
            ),
            patch.object(
                MultimodalDataItem, "_input_identity", side_effect=AssertionError
            ),
        ):
            output = asyncio.run(
                processor._process_cached_images([b"image"], "<image>", None)
            )
        self.assertEqual(
            output.mm_items[0].cache_key, artifacts[0].feature_hash.to_bytes(32, "big")
        )

    def test_transport_preserves_padding_until_frontend_finalization(self):
        processor = LlavaImageProcessor.__new__(LlavaImageProcessor)
        processor.processor_fingerprint = "sha256:" + "12" * 32
        processor.use_cuda_ipc = True
        item = _item(torch.arange(8))
        item.offsets = [(1, 8)]
        item.set_pad_value()
        input_ids = array("q", [1] + [2] * 8 + [3])
        padded = MultimodalProcessorOutput.build_padded_input_ids(input_ids, [item])
        output = MultimodalProcessorOutput(
            input_ids=input_ids, padded_input_ids=padded, mm_items=[item]
        )
        # Model fast paths may build padding before wrapping transport storage.
        with patch.object(
            processor, "_wrap_tensor_for_cuda_ipc", side_effect=lambda t: t
        ):
            processor._prepare_mm_items_for_transport([item])
        processor.finalize_cache_identities(output)
        self.assertEqual(
            output.padded_input_ids,
            output.build_padded_input_ids(input_ids, [item]),
        )
        self.assertNotEqual(output.padded_input_ids, padded)
        expected = output.padded_input_ids
        processor.finalize_cache_identities(output)
        self.assertIs(output.padded_input_ids, expected)

    def test_composite_processor_uses_inner_identity_namespace(self):
        inner = LlavaImageProcessor.__new__(LlavaImageProcessor)
        inner.processor_fingerprint = "sha256:" + "12" * 32
        outer = LlavaMultimodalProcessor.__new__(LlavaMultimodalProcessor)
        outer.inner = inner
        item = _item(torch.arange(8))
        output = MultimodalProcessorOutput(mm_items=[item])
        original = item.cache_key
        outer.finalize_cache_identities(output)
        self.assertNotEqual(original, item.cache_key)
        self.assertEqual(item.processor_fingerprint, inner.processor_fingerprint)
        expected = item.cache_key
        outer.finalize_cache_identities(output)
        self.assertEqual(expected, item.cache_key)

    def test_dataclass_metadata_keeps_tensor_storage_and_nested_types(self):
        @dataclass(slots=True)
        class Settings:
            crop: torch.Tensor
            output_size: tuple[int, int]

        value = Settings(torch.arange(8), (224, 224))
        with patch.object(torch.Tensor, "__deepcopy__", side_effect=AssertionError):
            digest = structured_digest(value)
            kwargs = media_preprocess_kwargs(value)
            self.assertIs(kwargs["crop"], value.crop)
            artifact = build_artifact_key(
                "sha256:" + "00" * 32,
                modality="image",
                processor_fingerprint="sha256:" + "11" * 32,
                preprocess_kwargs={"settings": value},
            )
        self.assertEqual(digest, structured_digest(value))
        value.output_size = (448, 224)
        self.assertNotEqual(digest, structured_digest(value))
        self.assertNotEqual(
            artifact,
            build_artifact_key(
                "sha256:" + "00" * 32,
                modality="image",
                processor_fingerprint="sha256:" + "11" * 32,
                preprocess_kwargs={"settings": value},
            ),
        )

    def test_request_namespace_prevents_partial_media_prefix_reuse(self):
        for backend, page_size, text_len, media_len in itertools.product(
            ("python", "rust"), (1, 16), (15, 16), (1, 32)
        ):
            with self.subTest(
                backend=backend, page=page_size, text=text_len, media=media_len
            ):
                a, b = _item(torch.zeros(8)), _item(torch.ones(8))
                a.set_pad_value()
                b.set_pad_value()
                # Simulate a short sentinel collision; full identities differ.
                b.pad_values = (a.pad_values[0], *b.pad_values[1:])
                b.pad_value = b.pad_values[0]
                keys = []
                for index, item in enumerate((a, b, a)):
                    req = Req(
                        rid=str(index),
                        origin_input_text=None,
                        origin_input_ids=[1] * text_len
                        + item.padding_sequence(media_len)
                        + [2] * 32,
                        sampling_params=SamplingParams(),
                        cache_salt="tenant",
                    )
                    req.extend_image_inputs(MultimodalInputs(mm_items=[item]))
                    keys.append(
                        RadixKey(
                            array("q", req.origin_input_ids), cache_salt=req.cache_salt
                        )
                    )
                    salt = req.cache_salt
                    req.bind_multimodal_cache_namespace()
                    self.assertEqual(req.cache_salt, salt)
                self.assertNotEqual(keys[0].cache_salt, keys[1].cache_salt)
                self.assertEqual(keys[0].cache_salt, keys[2].cache_salt)
                self.assertNotEqual(
                    get_storage_hash_str(keys[0]), get_storage_hash_str(keys[1])
                )
                core = _tree_core(backend, page_size=page_size)
                cached_len = len(keys[0]) // page_size * page_size
                _pump_insert(
                    core, InsertParams(key=keys[0], value=torch.arange(cached_len))
                )
                self.assertEqual(
                    core.match_prefix(MatchPrefixParams(key=keys[1])).device_prefix_len,
                    0,
                )
                self.assertEqual(
                    core.match_prefix(MatchPrefixParams(key=keys[2])).device_prefix_len,
                    cached_len,
                )

    def test_empty_numpy_retains_shape_and_dtype(self):
        keys = [
            structured_digest(np.empty(shape, dtype=dtype))
            for shape, dtype in itertools.product(
                ((0,), (0, 3), (0, 4)), (np.uint8, np.float32)
            )
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_ocr_config_updates_finalized_identity(self):
        keys = []
        for width in (512, 1024, 512):
            item = _item(torch.arange(8))
            original = item.cache_key
            # transport lifetime is owned by encoding, not by cache lookup
            item.feature = object()
            UnlimitedOCRProcessor._mix_config_into_hash([item], {"image_size": width})
            self.assertNotEqual(item.cache_key, original)
            keys.append(item.cache_key)
        self.assertNotEqual(keys[0], keys[1])
        self.assertEqual(keys[0], keys[2])

    def test_structure_is_not_flattened(self):
        x = torch.arange(16, dtype=torch.uint8)
        pairs = [
            (x, x.reshape(4, 4)),
            (x, x.view(torch.int32)),
            ([x[:4], x[4:]], [x[:8], x[8:]]),
            ([x], [[x]]),
            ([x], (x,)),
            (
                torch.tensor([42], dtype=torch.uint8),
                torch.tensor([42, 0, 0, 0], dtype=torch.uint8),
            ),
        ]
        for a, b in pairs:
            with self.subTest(a=type(a), b=type(b)):
                self.assertNotEqual(hash_feature_digest(a), hash_feature_digest(b))
        self.assertEqual(
            structured_digest(x.reshape(4, 4).T),
            structured_digest(x.reshape(4, 4).T.contiguous()),
        )

    def test_content_not_caller_hash_is_authoritative(self):
        a, b = _item(torch.zeros(8), hash=7), _item(torch.ones(8), hash=7)
        self.assertNotEqual(a.cache_key, b.cache_key)
        self.assertNotEqual(a.padding_values(), b.padding_values())
        a.set_hash(17)
        b.set_hash(17)
        self.assertNotEqual(a.cache_key, b.cache_key)

    def test_modality_format_geometry_and_processor_are_scoped(self):
        keys = []
        for modality, fmt, grid, namespace in itertools.product(
            Modality.all(),
            (MultimodalInputFormat.NORMAL, MultimodalInputFormat.PRECOMPUTED_EMBEDDING),
            ([1, 2, 4], [1, 4, 2]),
            ("11", "22"),
        ):
            item = MultimodalDataItem(
                modality=modality,
                feature=torch.ones(8),
                format=fmt,
                model_specific_data={"image_grid_thw": torch.tensor([grid])},
            )
            item.bind_processor_fingerprint("sha256:" + namespace * 32)
            keys.append(item.cache_key)
        self.assertEqual(len(keys), len(set(keys)))

    def test_full_identity_and_combined_keys(self):
        a, b = _item(None), _item(None)
        a.set_identity(b"samehead" + b"A" * 24)
        b.set_identity(b"samehead" + b"B" * 24)
        self.assertEqual(a.hash, b.hash)  # Fault injection, not a digest collision.
        cache = MultiModalStaticCache(4096)
        cache.set(a.cache_key, EmbeddingResult(embedding=torch.tensor([17.0])))
        self.assertIsNone(cache.get_single(b.cache_key))
        cache.set(
            cache.combine_hashes([a.cache_key]),
            EmbeddingResult(embedding=torch.tensor([17.0])),
        )
        self.assertIsNone(cache.get([b.cache_key]))
        x, y = 1, 1 + (2**61 - 1)
        self.assertEqual(hash(x), hash(y))
        cache.set(
            cache.combine_hashes([x]), EmbeddingResult(embedding=torch.tensor([17.0]))
        )
        self.assertIsNone(cache.get([y]))

    def test_scheduler_same_batch_and_a_b_a(self):
        a, b = _item(torch.tensor([17.0])), _item(torch.tensor([23.0]))
        a.set_identity(b"samehead" + b"A" * 24)
        b.set_identity(b"samehead" + b"B" * 24)

        def encode(items):
            return [item.feature.reshape(1, 1) for item in items]

        with patch.object(mm_schedule, "embedding_cache", MultiModalStaticCache(4096)):
            for item in (a, b, a, b):
                output = mm_schedule._get_chunked_embedding_by_item(
                    encode, [item], 0, 1, torch.device("cpu")
                )[0]
                self.assertTrue(torch.equal(output, item.feature.reshape(1, 1)))
        with patch.object(mm_schedule, "embedding_cache", MultiModalStaticCache(4096)):
            reqs = [
                mm_schedule.PerImageRequestInfo(i, [item], [(0, 0)], 0, 1)
                for i, item in enumerate((a, b, a))
            ]
            result = mm_schedule._batch_encode_per_image_misses(
                encode, reqs, torch.device("cpu")
            )
            self.assertTrue(
                torch.equal(result[(b.cache_key, 1)], b.feature.reshape(1, 1))
            )
            self.assertEqual(len(result), 2)

    def test_split_invalidates_parent_identity(self):
        item = _item(torch.tensor([[1.0], [2.0]]))
        item.offsets = [(0, 0), (2, 2)]
        item.set_pad_value()
        parts = get_new_expanded_mm_items([item])
        self.assertEqual(len(parts), 2)
        self.assertEqual(len({item.cache_key, *(part.cache_key for part in parts)}), 3)

    def test_identity_survives_wire_without_rehash(self):
        item = _item(torch.arange(8))
        expected = item.cache_key
        for restored in (
            pickle.loads(pickle.dumps(item)),
            msgspec.msgpack.decode(
                msgspec.msgpack.encode(
                    MultimodalDataItem(
                        modality=item.modality,
                        hash=item.hash,
                        identity=item.identity,
                        pad_values=item.pad_values,
                        pad_value=item.pad_value,
                    )
                ),
                type=MultimodalDataItem,
            ),
        ):
            with patch(
                "sglang.srt.managers.mm_utils.hash_feature_digest",
                side_effect=AssertionError("already hashed"),
            ):
                self.assertEqual(restored.cache_key, expected)

    def test_python_and_rust_prefix_trees(self):
        a, b = _item(torch.zeros(32)), _item(torch.ones(32))
        a.set_pad_value()
        b.set_pad_value()
        for backend, page_size in itertools.product(("python", "rust"), (1, 16)):
            with self.subTest(backend=backend, page_size=page_size):
                core = _tree_core(backend, page_size=page_size)
                first = [1] * 16 + a.padding_sequence(32) + [2] * 16
                second = [1] * 16 + b.padding_sequence(32) + [2] * 16
                _insert(core, first, list(range(len(first))))
                match = core.match_prefix(MatchPrefixParams(key=_key(second)))
                self.assertEqual(match.device_prefix_len, 16)


if __name__ == "__main__":
    unittest.main()
