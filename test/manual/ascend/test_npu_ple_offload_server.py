"""PLE host offload (--ple-offload-*) end to end on NPU.

The table stays on the host and only the rows a forward needs are staged to the
device. The generation checks are smoke checks: a wrong row mapping or scale
keeps every tensor well-shaped and can still produce a plausible answer.

Covers what test_npu_ple_offload.py cannot reach by constructing the embedding
directly: the CLI -> hf_text_config -> model wiring, the real checkpoint weight
loader, the prefill prefetch side stream, decode graphs split around the host
lookup, TP sharding over HCCL, and the file backend's on-disk layout.

Not covered here:
  - Numerical agreement with a known-good reference; that needs reference
    outputs or PLE embeddings for this checkpoint.
  - Restart reuse of a populated table file. Rewriting one runs at ~17 MB/s
    (~55 min for 47.7 GiB) until the upstream rewrite lands, so delete the
    directory between runs instead.
  - Device memory saved by the offload. Read it from the loader's
    "PLE table: ..." line and the reported KV pool size.

Run: python3 test/manual/ascend/test_npu_ple_offload_server.py -v
"""

import os
import re
import shutil
import tempfile
import unittest

import requests
import sympy
import torch

from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    pad_vocab_size,
)
from sglang.srt.models.qwen4_exp_ple_table import ple_table_file_name
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers import get_config
from sglang.test.ascend.test_ascend_utils import MODEL_WEIGHTS_DIR
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = os.environ.get(
    "SGLANG_TEST_PLE_MODEL",
    os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3.8-Flash-Next"),
)
TP_SIZE = os.environ.get("SGLANG_TEST_PLE_TP_SIZE", "4")

# ple_table_<dims>_<dtype>_<bytes>B_rows<start>-<end>.bin
TABLE_FILE = re.compile(
    r"^ple_table_(?P<dims>[0-9x]+)_(?P<dtype>[a-z0-9_]+)_(?P<nbytes>\d+)B"
    r"_rows(?P<start>\d+)-(?P<end>\d+)\.bin$"
)


def expected_table_files(dtype: torch.dtype, tp_size: int) -> list[str]:
    """One file per PLE layer and TP rank, derived from the checkpoint config.

    The file name carries no layer id, so two layers with equal shard shapes
    would silently share one file; a duplicate here flags that.
    """
    text_config = get_config(MODEL, trust_remote_code=True).text_config
    ngram_heads = (text_config.ngram_size - 1) * text_config.heads_per_ngram
    head_dim = text_config.ple_embed_dim // ngram_heads
    divisible_by = text_config.make_ngram_vocab_size_divisible_by
    # Head k of the i-th PLE layer (sorted by id) owns the (i * ngram_heads + k + 1)-th
    # prime above the base; see Qwen4ExpNGramEmbedding._build_head_vocab_and_offsets.
    prime = text_config.ngram_vocab_size_base - 1
    names = []
    for layer_id in sorted(set(text_config.ple_layer_ids)):
        vocab = 0
        for _ in range(ngram_heads):
            prime = int(sympy.nextprime(prime))
            vocab += prime
        vocab = -(-vocab // divisible_by) * divisible_by
        # PLE ids are 1-based; only decoder layers below num_hidden_layers exist.
        if not 1 <= layer_id <= text_config.num_hidden_layers:
            continue
        padded = pad_vocab_size(vocab)
        for rank in range(tp_size):
            shard = VocabParallelEmbedding._get_indices(
                padded, padded, vocab, vocab, rank, tp_size
            )
            names.append(
                ple_table_file_name(
                    (padded // tp_size, head_dim),
                    dtype,
                    tag=f"rows{shard.org_vocab_start_index}"
                    f"-{shard.org_vocab_end_index}",
                )
            )
    return names


class TestNpuPleOffloadServer(CustomTestCase):
    """Testcase: the server loads and answers with the PLE table offloaded on NPU.

    [Test Category] Functional
    [Test Target] --ple-offload-embedding / --ple-offload-backend / --ple-offload-dir on NPU
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(MODEL):
            raise unittest.SkipTest(f"model weights not found: {MODEL}")
        cls.base_url = DEFAULT_URL_FOR_TEST

    def setUp(self):
        self.process = None
        self.table_dir = tempfile.mkdtemp(prefix="ple_table_")
        self.addCleanup(shutil.rmtree, self.table_dir, ignore_errors=True)

    def tearDown(self):
        if self.process is not None:
            kill_process_tree(self.process.pid)

    def launch(self, backend, extra_args=None, logs=None):
        args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--tp-size",
            TP_SIZE,
            "--mem-fraction-static",
            "0.85",
            "--ple-offload-embedding",
            "--ple-offload-backend",
            backend,
        ]
        if backend == "file":
            args += ["--ple-offload-dir", self.table_dir]
        args += extra_args or []
        self.process = popen_launch_server(
            MODEL,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            return_stdout_stderr=logs,
        )

    def generate(self, prompt, max_new_tokens=32):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=120,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()["text"]

    def server_info(self):
        return requests.get(f"{self.base_url}/server_info", timeout=30).json()

    def check_serving(self, backend):
        info = self.server_info()
        self.assertTrue(
            info["ple_offload_embedding"],
            f"{backend}: server did not enable the PLE offload",
        )
        self.assertEqual(info["ple_offload_backend"], backend)

        # Smoke only: a grossly wrong mapping breaks the answer, a subtle one may not.
        self.assertIn("Paris", self.generate("The capital of France is"))

        # Prefill sizes drive a different lookup width than decode does.
        self.assertTrue(self.generate("Count from one to twenty: " * 64))

    def test_pinned_backend_serves(self):
        self.launch("pinned")
        self.check_serving("pinned")

    def test_file_backend_serves(self):
        self.launch("file")
        self.check_serving("file")
        self.assertEqual(self.server_info()["ple_offload_dir"], self.table_dir)

    def test_file_backend_table_layout(self):
        self.launch("file")
        self.check_serving("file")

        files = sorted(os.listdir(self.table_dir))
        self.assertTrue(files, f"no PLE table written under {self.table_dir}")

        dtypes = set()
        for name in files:
            match = TABLE_FILE.match(name)
            self.assertIsNotNone(match, f"unexpected file name: {name}")
            dtypes.add(match["dtype"])
            stat = os.stat(os.path.join(self.table_dir, name))
            self.assertEqual(
                stat.st_size,
                int(match["nbytes"]),
                f"{name}: apparent size does not match the name",
            )
            # Created sparse, then filled by the weight loader; a file still
            # made of holes means the shard was never written to this dir.
            self.assertGreater(
                stat.st_blocks * 512,
                stat.st_size // 2,
                f"{name}: mostly unwritten, the loader did not fill the table",
            )

        # The table dtype depends on the checkpoint's quantization, not on this test.
        self.assertEqual(len(dtypes), 1, f"mixed table dtypes: {sorted(dtypes)}")
        expected = expected_table_files(getattr(torch, dtypes.pop()), int(TP_SIZE))
        self.assertEqual(
            len(set(expected)),
            len(expected),
            "two PLE layers map to the same table file",
        )
        self.assertEqual(files, sorted(expected))

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_PLE_GRAPH_VS_EAGER"),
        "loads the model twice",
    )
    def test_decode_graph_matches_eager(self):
        prompts = ["The capital of France is", "Count from one to twenty: " * 64]
        paths = [
            os.path.join(self.table_dir, f"server_{name}.log")
            for name in ("stdout", "stderr")
        ]
        with open(paths[0], "w") as stdout, open(paths[1], "w") as stderr:
            self.launch("pinned", logs=(stdout, stderr))
            graph_texts = [self.generate(prompt) for prompt in prompts]
            # Separate handles: the dump threads keep writing to the ones above.
            log = "".join(open(path).read() for path in paths)
            kill_process_tree(self.process.pid)
            self.process = None
        # Launching proves little on its own: decode may run eagerly.
        self.assertIn("NPU PLE decode graph split into", log)

        self.launch("pinned", extra_args=["--cuda-graph-backend-decode", "disabled"])
        eager_texts = [self.generate(prompt) for prompt in prompts]
        # Replaying capture-time PLE rows would diverge within a few tokens.
        self.assertEqual(graph_texts, eager_texts)


if __name__ == "__main__":
    unittest.main()
