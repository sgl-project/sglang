"""The multimodal CPU preprocess pool stays safe while the tokenizer process keeps
HF tokenizers parallelism on.

The pool forks its workers from a process whose tokenizers (rayon) thread pool is
already live: `parallel_prompt_encode` ran first. A forked worker inherits that
pool's state but none of its threads, so a parallel encode in the worker would
wait forever. `TOKENIZERS_PARALLELISM=true` is set here so tokenizers' own
after-fork guard (it only acts when the variable is unset) cannot hide a missing
guard in the pool. The cases cover the first worker start, image requests and
worker encodes running while the parent keeps encoding, and the pool that
replaces a broken one.
"""

import asyncio
import io
import os
import threading
import unittest
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
from transformers import CLIPImageProcessor, PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai import parallel_prompt_encode as ppe
from sglang.srt.multimodal.processors.llava import LlavaImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

# A hung worker never answers; fail the case instead of the whole run.
RESULT_TIMEOUT_S = 60
SPECIALS = ["<|user|>", "<|assistant|>"]


def _backend_json():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog 0123456789"] * 50,
        trainers.BpeTrainer(
            vocab_size=400,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    backend.add_tokens(
        [AddedToken(t, normalized=False, special=True) for t in SPECIALS]
    )
    return backend.to_str()


def _encode_batch_in_worker(backend_json, texts):
    # Runs in a pool worker: encode_batch takes the rayon pool when parallelism is on.
    backend = Tokenizer.from_str(backend_json)
    return [e.ids for e in backend.encode_batch(texts, add_special_tokens=False)]


def _exit_worker_process():
    os._exit(1)


def _png_bytes(size):
    image = Image.new("RGB", size, (200, 30, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestCpuPreprocessPoolWithTokenizerParallelism(CustomTestCase):
    def setUp(self):
        super().setUp()
        env = mock.patch.dict(
            os.environ, {"TOKENIZERS_PARALLELISM": "true", "SGLANG_CPU_WORKERS": "2"}
        )
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(setattr, ppe, "_parallel_ok", ppe._parallel_ok)
        ppe._parallel_ok = None
        self.backend_json = _backend_json()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_str(self.backend_json)
        )
        self.prompt = "".join(
            f"{SPECIALS[i % 2]}the quick brown fox {i} " * 200 for i in range(64)
        )
        # The parent's rayon pool is live before the pool forks any worker.
        self.assertEqual(
            ppe.parallel_prompt_encode(self.tokenizer, self.prompt, {}),
            self.tokenizer.encode(self.prompt),
        )
        if (os.cpu_count() or 1) > 1:
            self.assertTrue(ppe._parallel_ok)
        self.processor = object.__new__(LlavaImageProcessor)
        self.processor.mm_feature_transport = "cpu"
        self.processor._cpu_executor_lock = threading.Lock()
        self.processor._processor = SimpleNamespace(
            image_processor=CLIPImageProcessor()
        )
        self.processor.cpu_executor = self.processor._create_cpu_executor()
        self.addCleanup(self._stop_pool)

    def _stop_pool(self):
        executor = self.processor.cpu_executor
        processes = list((executor._processes or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.kill()

    def _keep_encoding(self, stop):
        while not stop.is_set():
            ppe.parallel_prompt_encode(self.tokenizer, self.prompt, {})

    def _worker_encodes(self):
        texts = [f"the lazy dog {i} " * 300 for i in range(8)]
        expected = _encode_batch_in_worker(self.backend_json, texts)
        futures = [
            self.processor.cpu_executor.submit(
                _encode_batch_in_worker, self.backend_json, texts
            )
            for _ in range(4)
        ]
        for future in futures:
            self.assertEqual(future.result(timeout=RESULT_TIMEOUT_S), expected)

    def test_workers_encode_and_preprocess_while_the_parent_encodes(self):
        stop = threading.Event()
        encoder = threading.Thread(target=self._keep_encoding, args=(stop,))
        encoder.start()
        try:
            self._worker_encodes()

            async def preprocess():
                return await asyncio.gather(
                    *(
                        self.processor._process_single_image(
                            _png_bytes((64 + 16 * i, 48)), "pad", None
                        )
                        for i in range(6)
                    )
                )

            with mock.patch.dict(
                os.environ, {"REQUEST_TIMEOUT": str(RESULT_TIMEOUT_S)}
            ):
                results = asyncio.run(preprocess())
        finally:
            stop.set()
            encoder.join()
        for i, (pixel_values, _, size) in enumerate(results):
            # "pad" squares the image on its longer side before the resize.
            self.assertEqual(size, (64 + 16 * i, 64 + 16 * i))
            self.assertEqual(pixel_values.shape, (3, 224, 224))

    def test_the_replacement_pool_after_a_worker_failure_still_encodes(self):
        self._worker_encodes()
        failed = self.processor.cpu_executor
        with self.assertRaises(BrokenProcessPool):
            failed.submit(_exit_worker_process).result(timeout=RESULT_TIMEOUT_S)
        self.processor._replace_broken_cpu_executor(failed)
        self.assertIsNot(self.processor.cpu_executor, failed)
        ppe.parallel_prompt_encode(self.tokenizer, self.prompt, {})
        self._worker_encodes()


if __name__ == "__main__":
    unittest.main()
