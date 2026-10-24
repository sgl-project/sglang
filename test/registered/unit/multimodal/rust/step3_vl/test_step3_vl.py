import asyncio
import base64
import json
import os
import re
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
)
from sglang.srt.multimodal.processors.step3_vl import (  # noqa: E402
    CPUToTensor,
    ImagePatcher,
    Step3VLImageProcessor,
    Step3VLProcessor,
    step3_vl_rust_spec,
)
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.rust_server.multimodal import (  # noqa: E402
    RustMmProcessor,
    RustStep3VlSpec,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import image_bytes, load_core, make_image  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

CORE = load_core()

SPECIAL = {
    "<im_start>": 900,
    "<im_patch>": 901,
    "<im_end>": 902,
    "<patch_start>": 903,
    "<patch_newline>": 904,
    "<patch_end>": 905,
}

SIZES = [
    (640, 480),
    (700, 400),
    (1000, 800),
    (1000, 400),
    (3000, 600),
    (3000, 450),
    (400, 20),
    (30, 300),
    (729, 728),
    (1512, 504),
    (1281, 719),
    (505, 1600),
]


class FakeTokenizer:
    unk_token_id = 0
    _split = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL) + ")")

    def get_vocab(self):
        return dict(SPECIAL)

    def convert_tokens_to_ids(self, token):
        return SPECIAL.get(token, self.unk_token_id)

    def __call__(self, text):
        ids = [1]
        for piece in self._split.split(text):
            ids.extend([SPECIAL[piece]] if piece in SPECIAL else map(ord, piece))
        return {"input_ids": ids}


class BosTokenizer(FakeTokenizer):
    bos_token = "<s>"
    bos_token_id = 1
    _split = re.compile("(" + "|".join(re.escape(t) for t in [*SPECIAL, "<s>"]) + ")")

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, list):
            return {
                "input_ids": [self(t, add_special_tokens)["input_ids"] for t in text]
            }
        ids = [self.bos_token_id] if add_special_tokens else []
        for piece in self._split.split(text):
            if piece == self.bos_token:
                ids.append(self.bos_token_id)
            elif piece in SPECIAL:
                ids.append(SPECIAL[piece])
            else:
                ids.extend(map(ord, piece))
        return {"input_ids": ids}


def python_pixels(proc, image):
    raw, patches, mask = proc.patcher(image)
    pv = proc._convert_images_to_pixel_values([raw])[0].float().cpu().numpy()
    ppv = [
        p.float().cpu().numpy()
        for p in proc._convert_images_to_pixel_values(patches, is_patch=True)
    ]
    size = proc.patch_size
    ppv = np.concatenate(ppv) if ppv else np.zeros((0, 3, size, size), np.float32)
    return pv, ppv, mask or []


@unittest.skipUnless(CORE, "sglang-mm extension not built")
class TestStep3VlRust(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = Step3VLProcessor(SimpleNamespace(), FakeTokenizer())
        cls.fields = step3_vl_rust_spec(cls.proc)
        cls.spec_json = json.dumps({"family": "step3_vl", **cls.fields})

    def test_spec_from_processor(self):
        self.assertEqual(self.fields["image_token_id"], SPECIAL["<im_patch>"])
        self.assertEqual(self.fields["patch_newline_id"], SPECIAL["<patch_newline>"])
        self.assertEqual(
            (self.fields["image_size"], self.fields["patch_size"]), (728, 504)
        )

    def test_spec_rejects_missing_tokens(self):
        class Missing(FakeTokenizer):
            def convert_tokens_to_ids(self, token):
                return (
                    0
                    if token == "<patch_newline>"
                    else super().convert_tokens_to_ids(token)
                )

        self.assertIsNone(
            step3_vl_rust_spec(Step3VLProcessor(SimpleNamespace(), Missing()))
        )

    def test_plan_and_token_counts_match_patcher(self):
        patcher = ImagePatcher()
        for w, h in SIZES:
            with self.subTest(size=(w, h)):
                _, _, _, windows, newlines = CORE.step3_vl.plan_windows(w, h)
                _, patches, mask = patcher(make_image(w, h))
                self.assertEqual(len(windows), len(patches))
                self.assertEqual(list(newlines), list(mask or []))
                self.assertEqual(
                    CORE.step3_vl.num_image_tokens(w, h, self.spec_json),
                    self.proc.get_num_image_tokens(w, h),
                )

    def test_expand_input_ids_matches_python_prompt(self):
        prompt = "a <im_patch> b <im_patch> c"
        sizes = [(1000, 800), (640, 480)]
        tok = self.proc.tokenizer
        repl, newlines = [], []
        for w, h in sizes:
            _, patches, mask = self.proc.patcher(make_image(w, h))
            repl.append(self.proc._get_image_repl_features(1, len(patches), mask)[0])
            newlines.append(list(mask or []))
        expected = tok(
            self.proc.replace_placeholder(prompt, self.proc.image_token, repl)
        )["input_ids"]
        ids, runs = CORE.step3_vl.expand_input_ids(
            tok(prompt)["input_ids"], newlines, self.spec_json
        )
        self.assertEqual(list(ids), expected)
        image_token = SPECIAL["<im_patch>"]
        covered = [
            i for per_image in runs for s, e in per_image for i in range(s, e + 1)
        ]
        self.assertEqual(covered, [i for i, t in enumerate(ids) if t == image_token])
        self.assertEqual([len(r) for r in runs], [5, 1])

    def test_pixels_match_python(self):
        for w, h in [(640, 480), (1000, 800), (3000, 450), (505, 1600)]:
            data = image_bytes(w, h)
            with self.subTest(size=(w, h), semantics="pil"):
                pv, ppv, nl = CORE.step3_vl.preprocess(data, self.spec_json, "pil")
                ref_pv, ref_ppv, ref_nl = python_pixels(self.proc, make_image(w, h))
                np.testing.assert_allclose(
                    np.asarray(pv).reshape(ref_pv.shape), ref_pv, atol=5e-4, rtol=0
                )
                np.testing.assert_allclose(
                    np.asarray(ppv).reshape(ref_ppv.shape), ref_ppv, atol=5e-4, rtol=0
                )
                self.assertEqual(list(nl), list(ref_nl))
            with self.subTest(size=(w, h), semantics="tensor"):
                pv, ppv, _ = CORE.step3_vl.preprocess(data, self.spec_json, "tensor")
                arr = torch.from_numpy(np.array(make_image(w, h)))
                ref_pv, ref_ppv, _ = python_pixels(
                    self.proc, arr.permute(2, 0, 1).contiguous()
                )
                ppv = np.asarray(ppv).reshape(ref_ppv.shape)
                np.testing.assert_allclose(
                    np.asarray(pv).reshape(ref_pv.shape), ref_pv, atol=5e-3, rtol=0
                )
                np.testing.assert_allclose(ppv, ref_ppv, atol=4e-2, rtol=0)
                if ppv.size:
                    self.assertLess(np.abs(ppv - ref_ppv).mean(), 1e-3)

    def test_batch_matches_single(self):
        datas = [image_bytes(1000, 800), image_bytes(640, 480, seed=1)]
        semantics = ["pil", "tensor"]
        batch = CORE.step3_vl.preprocess_batch(datas, self.spec_json, semantics)
        for data, mode, (pv, ppv, nl) in zip(datas, semantics, batch):
            spv, sppv, snl = CORE.step3_vl.preprocess(data, self.spec_json, mode)
            np.testing.assert_array_equal(np.asarray(pv), np.asarray(spv))
            np.testing.assert_array_equal(np.asarray(ppv), np.asarray(sppv))
            self.assertEqual(list(nl), list(snl))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            CORE.step3_vl.preprocess(b"junk", self.spec_json)
        with self.assertRaises(ValueError):
            CORE.step3_vl.preprocess_batch(
                [image_bytes(8, 8)], self.spec_json, ["pil", "pil"]
            )
        with self.assertRaises(ValueError):
            CORE.step3_vl.preprocess(image_bytes(8, 8), self.spec_json, "bicubic")
        with self.assertRaises(ValueError):
            CORE.step3_vl.expand_input_ids([1, 2], [[]], self.spec_json)
        with self.assertRaises(ValueError):
            CORE.step3_vl.num_image_tokens(10, 10, json.dumps({"family": "step3_vl"}))


class TestStep3VlBackendGate(CustomTestCase):
    def load(self, **kwargs):
        publish(ServerArgs(model_path="dummy", **kwargs), role="tokenizer")
        self.addCleanup(reset_context)
        processor = Step3VLProcessor(SimpleNamespace(), FakeTokenizer())
        return Step3VLImageProcessor._load_rust(processor)

    @unittest.skipUnless(CORE, "sglang-mm extension not built")
    def test_auto_backend_uses_rust(self):
        module, spec_json = self.load()
        self.assertIsNotNone(module)
        self.assertEqual(json.loads(spec_json)["family"], "step3_vl")

    def test_explicit_backends_keep_python(self):
        for backend in ("pil", "torchvision"):
            with self.subTest(backend=backend):
                self.assertEqual(
                    self.load(image_processor_backend=backend), (None, None)
                )
        self.assertEqual(self.load(disable_fast_image_processor=True), (None, None))


class TestStep3VlImageSource(CustomTestCase):
    def test_reads_file_urls_and_relative_paths(self):
        data = image_bytes(8, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a b.png"
            path.write_bytes(data)
            read = Step3VLImageProcessor._read_image_bytes
            self.assertEqual(read(str(path)), data)
            self.assertEqual(read("file://" + quote(str(path))), data)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertEqual(read("a b.png"), data)
            finally:
                os.chdir(cwd)
        self.assertEqual(read(data), data)
        self.assertEqual(read(base64.b64encode(data).decode()), data)

    def test_source_selection(self):
        source = Step3VLImageProcessor._rust_image_source
        self.assertEqual(source("x.png"), "x.png")
        self.assertEqual(source(b"x"), b"x")
        self.assertEqual(source(SimpleNamespace(url="u", preprocess_kwargs=None)), "u")
        self.assertIsNone(source(SimpleNamespace(url="u", preprocess_kwargs={"a": 1})))
        image = make_image(4, 4)
        self.assertIs(source(image), image)


class TestStep3VlWrapEncoded(CustomTestCase):
    def test_wraps_named_buffers(self):
        spec = RustStep3VlSpec(
            family="step3_vl",
            feature_shm=False,
            image_token_id=901,
            image_start_id=900,
            image_end_id=902,
            patch_start_id=903,
            patch_end_id=905,
            patch_newline_id=904,
            image_size=4,
            patch_size=2,
            image_feature_tokens=1,
            patch_feature_tokens=1,
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
        )

        def tensor(name, array, shape):
            return SimpleNamespace(name=name, feature=array, shape=shape)

        items = [
            SimpleNamespace(
                feature=np.arange(48, dtype=np.float32),
                shape=[1, 3, 4, 4],
                hash=11,
                offsets=[(2, 2), (5, 5), (8, 8)],
                model_specific_data={"num_patches": 2},
                aux=[
                    tensor("patch_pixel_values", np.ones(24, np.float32), [2, 3, 2, 2]),
                ],
            ),
            SimpleNamespace(
                feature=np.zeros(48, dtype=np.float32),
                shape=[1, 3, 4, 4],
                hash=22,
                offsets=[(12, 12)],
                model_specific_data={"num_patches": 0},
                aux=[
                    tensor("patch_pixel_values", np.zeros(0, np.float32), [0, 3, 2, 2]),
                ],
            ),
        ]
        import msgspec

        buffers = {}
        meta_items = []
        for index, item in enumerate(items):
            buffers[f"mm.feature.{index}"] = item.feature.reshape(item.shape)
            for aux in item.aux:
                buffers[f"mm.aux.{index}.{aux.name}"] = aux.feature.reshape(aux.shape)
            meta_items.append(
                dict(
                    modality="image",
                    hash=item.hash,
                    offsets=item.offsets,
                    model_specific_data=item.model_specific_data,
                )
            )
        buffers["mm.meta"] = np.frombuffer(
            msgspec.msgpack.encode(
                dict(items=meta_items, token_ids={"im_token_id": 901}, mrope_delta=None)
            ),
            dtype=np.uint8,
        )
        output = RustMmProcessor.wrap_encoded(spec, buffers)
        self.assertEqual(output.im_token_id, 901)
        first, second = output.mm_items
        self.assertEqual(tuple(first.feature.shape), (1, 3, 4, 4))
        self.assertEqual(first.hash, 11)
        self.assertEqual(first.offsets, [(2, 2), (5, 5), (8, 8)])
        self.assertEqual(first.model_specific_data["num_patches"].tolist(), [2])
        self.assertEqual(
            tuple(first.model_specific_data["patch_pixel_values"].shape), (2, 3, 2, 2)
        )
        self.assertNotIn("patch_newline_mask", first.model_specific_data)
        self.assertEqual(second.model_specific_data["num_patches"].tolist(), [0])
        self.assertEqual(
            tuple(second.model_specific_data["patch_pixel_values"].shape), (0, 3, 2, 2)
        )


class TestStep3VlSingleBos(CustomTestCase):
    """A prompt whose template already wrote BOS keeps exactly one BOS
    on both the Python and the Rust image paths."""

    PROMPT = "<s>a <im_patch> b"

    def setUp(self):
        self.proc = object.__new__(Step3VLImageProcessor)
        self.proc._processor = Step3VLProcessor(SimpleNamespace(), BosTokenizer())
        self.proc._tokenizer = self.proc._processor.tokenizer
        self.proc._tokenizer_auto_adds_specials = True

    def assert_single_bos(self, ids):
        self.assertEqual(list(ids[:2]), [BosTokenizer.bos_token_id, ord("a")])

    def test_python_image_processor_honors_add_special_tokens(self):
        output = self.proc._processor(
            text=[self.PROMPT],
            images=[make_image(640, 480)],
            add_special_tokens=False,
        )
        self.assert_single_bos(output["input_ids"][0])

    @unittest.skipUnless(CORE, "sglang-mm extension not built")
    def test_rust_image(self):
        fields = step3_vl_rust_spec(self.proc._processor)
        self.proc._rust = CORE.step3_vl
        self.proc._rust_spec_json = json.dumps({"family": "step3_vl", **fields})
        self.proc.io_executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.proc.io_executor.shutdown)
        self.proc.gpu_image_decode = False
        self.proc.mm_tokens = SimpleNamespace(image_token_id=SPECIAL["<im_patch>"])
        output = asyncio.run(
            self.proc._process_mm_data_rust([image_bytes(640, 480)], self.PROMPT)
        )
        self.assert_single_bos(output.input_ids)


class TestStep3VlPromptIdsPlaceholders(CustomTestCase):
    def test_placeholder_count_mismatch_is_rejected_before_loading(self):
        """Prompt ids whose image placeholders do not match the images are rejected
        up front; the Rust-to-Python fallback must not swallow the mismatch."""
        proc = object.__new__(Step3VLImageProcessor)
        proc.IM_TOKEN_ID = SPECIAL["<im_patch>"]
        proc._rust = None
        proc.load_mm_data = AsyncMock(side_effect=AssertionError("loaded media"))
        images = [image_bytes(8, 8), image_bytes(8, 8, seed=1)]
        with self.assertRaisesRegex(ValueError, "1 image placeholder"):
            asyncio.run(
                proc.process_mm_data_async(
                    image_data=images,
                    input_text=[1, SPECIAL["<im_patch>"], 2],
                    request_obj=SimpleNamespace(video_data=None),
                )
            )


class TestStep5PerImageItems(CustomTestCase):
    COUNTS = [2, 0, 3]

    def setUp(self):
        self.proc = object.__new__(Step3VLImageProcessor)
        self.proc.per_image_mm_items = True

    def bundled(self, counts=None, feature=None, patches=None, **extra):
        counts = self.COUNTS if counts is None else counts
        offsets, start = [], 10
        for n in counts:
            for _ in range(n):
                offsets.append((start, start + 80))
                start += 83
            offsets.append((start, start + 168))
            start += 175
        if feature is None:
            feature = torch.rand(len(counts), 3, 6, 6)
        if patches is None:
            patches = torch.rand(sum(counts), 3, 4, 4)
        return MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=offsets,
            model_specific_data={
                "num_patches": torch.tensor(counts, dtype=torch.long),
                "patch_pixel_values": patches,
                **extra,
            },
        )

    def test_splits_bundle_into_one_item_per_image(self):
        item = self.bundled()
        items = self.proc._split_per_image(item)
        self.assertEqual(len(items), 3)
        run, patch = 0, 0
        for i, (split, n) in enumerate(zip(items, self.COUNTS)):
            self.assertTrue(torch.equal(split.feature, item.feature[i : i + 1]))
            self.assertEqual(split.offsets, item.offsets[run : run + n + 1])
            self.assertEqual(split.model_specific_data["num_patches"].tolist(), [n])
            patches = split.model_specific_data.get("patch_pixel_values")
            if n:
                expected = item.model_specific_data["patch_pixel_values"]
                self.assertTrue(torch.equal(patches, expected[patch : patch + n]))
            else:
                self.assertIsNone(patches)
            run += n + 1
            patch += n

    def test_new_image_keeps_earlier_pad_values(self):
        base = self.bundled()
        grown = self.bundled(
            counts=self.COUNTS + [1],
            feature=torch.cat([base.feature, torch.rand(1, 3, 6, 6)]),
            patches=torch.cat(
                [base.model_specific_data["patch_pixel_values"], torch.rand(1, 3, 4, 4)]
            ),
        )
        first = self.proc._split_per_image(base)
        second = self.proc._split_per_image(grown)
        for item in first + second:
            item.set_pad_value()
        self.assertEqual(
            [x.pad_value for x in first], [x.pad_value for x in second[:3]]
        )
        self.assertEqual(len({x.pad_value for x in second}), 4)

    def test_keeps_bundle_when_layout_does_not_match(self):
        cases = {
            "single image": self.bundled(counts=[2]),
            "offset count": self.bundled(),
            "patch count": self.bundled(),
            "unknown field": self.bundled(image_grid_thw=torch.ones(3, 3)),
        }
        cases["offset count"].offsets = cases["offset count"].offsets[:-1]
        cases["patch count"].model_specific_data["patch_pixel_values"] = torch.rand(
            4, 3, 4, 4
        )
        for name, item in cases.items():
            with self.subTest(name):
                self.assertEqual(self.proc._split_per_image(item), [item])

    def test_disabled_for_non_step5(self):
        self.proc.per_image_mm_items = False
        item = self.bundled()
        self.assertEqual(self.proc._split_per_image(item), [item])


class TestStep3VlCpuPreprocessing(CustomTestCase):
    def test_cuda_transport_cannot_allocate_a_worker_pool(self):
        for transport in ("cpu", "cuda_ipc", "cuda_vmm"):
            with self.subTest(transport=transport):
                args = ServerArgs(
                    model_path="dummy",
                    mm_feature_transport=transport,
                    mm_process_config={},
                    image_processor_backend="pil",
                )
                publish(args, role="tokenizer")
                self.addCleanup(reset_context)
                with patch(
                    "sglang.srt.multimodal.processors.base_processor.MmItemMemoryPool",
                    side_effect=AssertionError("GPU pool allocation"),
                ):
                    proc = Step3VLImageProcessor(
                        SimpleNamespace(),
                        args,
                        BosTokenizer(),
                        transport_mode="default",
                    )
                self.addCleanup(proc.shutdown)
                self.assertEqual(proc.mm_feature_transport, "cpu")
                self.assertFalse(proc.use_cuda_ipc)
                self.assertFalse(proc.keep_mm_features_on_device)

    def test_conversion_does_not_touch_cuda(self):
        image = make_image(64, 48)
        array = np.asarray(image).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA probe")):
            outputs = [CPUToTensor()(value) for value in (image, array, tensor)]
        for output in outputs:
            self.assertEqual(output.device.type, "cpu")
            torch.testing.assert_close(output, outputs[0])
        self.assertFalse(Step3VLImageProcessor.gpu_image_decode)

    @unittest.skipUnless(CORE, "sglang-mm extension not built")
    def test_low_and_high_take_rust_path_with_cpu_pixels(self):
        proc = object.__new__(Step3VLImageProcessor)
        proc._processor = Step3VLProcessor(SimpleNamespace(), BosTokenizer())
        proc._rust = CORE.step3_vl
        proc._rust_spec_json = json.dumps(
            {"family": "step3_vl", **step3_vl_rust_spec(proc._processor)}
        )
        proc.IM_TOKEN_ID = SPECIAL["<im_patch>"]
        proc.mm_tokens = SimpleNamespace(image_token_id=proc.IM_TOKEN_ID)
        proc.io_executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(proc.io_executor.shutdown)
        proc.load_mm_data = AsyncMock(side_effect=AssertionError("Python fallback"))
        for size in [(3000, 450), (450, 3000), (1000, 800), (64, 48)]:
            with self.subTest(size=size):
                low = SimpleNamespace(
                    url=image_bytes(*size), detail="low", preprocess_kwargs=None
                )
                high = image_bytes(1000, 800)
                resolved = proc._load_low_detail_image(low)
                self.assertLessEqual(max(resolved.size), 728)
                self.assertLessEqual(max(resolved.size) / min(resolved.size), 1.5)
                with patch(
                    "torch.cuda.is_available", side_effect=AssertionError("CUDA probe")
                ):
                    output = asyncio.run(
                        proc.process_mm_data_async(
                            [low, high],
                            [1, proc.IM_TOKEN_ID, 2, proc.IM_TOKEN_ID],
                            SimpleNamespace(video_data=None),
                        )
                    )
                item = output.mm_items[0]
                self.assertEqual(item.feature.device.type, "cpu")
                self.assertEqual(
                    item.model_specific_data["num_patches"].tolist(), [0, 4]
                )
                ref, _, _ = python_pixels(proc._processor, resolved)
                np.testing.assert_allclose(
                    item.feature[0].numpy(), ref[0], atol=5e-4, rtol=0
                )
                self.assertEqual(
                    output.input_ids.count(proc.IM_TOKEN_ID),
                    sum(end - start + 1 for start, end in item.offsets),
                )
        proc.load_mm_data.assert_not_called()

    def test_low_python_fallback_retains_resized_image(self):
        proc = object.__new__(Step3VLImageProcessor)
        proc._rust = None
        proc.IM_TOKEN_ID = SPECIAL["<im_patch>"]
        proc.mm_tokens = SimpleNamespace(image_token_id=proc.IM_TOKEN_ID)
        proc.io_executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(proc.io_executor.shutdown)
        proc.load_mm_data = AsyncMock(return_value=object())
        proc.process_and_combine_mm_data_async = AsyncMock(
            return_value=([], torch.tensor([1, 2]), None)
        )
        low = SimpleNamespace(url=image_bytes(3000, 450), detail="low")
        asyncio.run(
            proc.process_mm_data_async(
                [low], [1, proc.IM_TOKEN_ID], SimpleNamespace(video_data=None)
            )
        )
        image = proc.load_mm_data.call_args.kwargs["image_data"][0]
        self.assertEqual(image.size, (728, 486))


if __name__ == "__main__":
    unittest.main()
