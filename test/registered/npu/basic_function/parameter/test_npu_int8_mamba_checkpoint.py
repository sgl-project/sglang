"""Ascend coverage for --enable-int8-mamba-checkpoint.

The codec/pool cases migrate the device-independent checks from
test/registered/mem_cache/test_int8_checkpoint_store.py to real NPU tensors.
The Llama case checks flag compatibility only: Llama has no Mamba state, so
successful generation or a KV prefix hit does NOT prove INT8 Mamba reuse.

Run from the repository root on one idle NPU:
    ASCEND_RT_VISIBLE_DEVICES=0 \
    SGLANG_TEST_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
    python test/registered/npu/basic_function/parameter/test_npu_int8_mamba_checkpoint.py -v

Set SGLANG_TEST_LOG_DIR to retain server logs and JSON evidence.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.mem_cache.mamba_checkpoint_pool import (
    Int8CheckpointStore,
    MambaCheckpointPool,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu, kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, popen_launch_server

register_npu_ci(est_time=180, suite="full-1-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Requires Ascend NPU")
class TestNpuInt8MambaCheckpointPool(CustomTestCase):
    def test_codec_error_and_zero(self):
        generator = torch.Generator().manual_seed(2026)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                state = (
                    torch.randn(4, 2, 32, 128, 128, generator=generator) * 0.06
                ).to(device="npu", dtype=dtype)
                # Include a zero channel among nonzero channels.
                state[..., 0] = 0
                q, scale = Int8CheckpointStore.quantize(state)
                self.assertEqual(q.device.type, "npu")
                self.assertEqual(q.dtype, torch.int8)
                self.assertEqual(scale.dtype, dtype)
                self.assertEqual(tuple(scale.shape), (4, 2, 32, 1, 128))
                restored = Int8CheckpointStore.dequantize(q, scale, dtype)
                self.assertTrue(torch.isfinite(restored).all().item())
                relative_error = (
                    (restored.float() - state.float()).norm() / state.float().norm()
                ).item()
                self.assertLess(relative_error, 0.01)
                self.assertEqual(torch.count_nonzero(restored[..., 0]).item(), 0)
                zeros = torch.zeros_like(state)
                zero_q, zero_scale = Int8CheckpointStore.quantize(zeros)
                self.assertTrue(
                    torch.equal(
                        Int8CheckpointStore.dequantize(zero_q, zero_scale, dtype),
                        zeros,
                    )
                )
                print(f"NPU codec: dtype={dtype}, relative_error={relative_error:.8f}")

    def test_pool_copy_on_write_memory_and_lifecycle(self):
        generator = torch.Generator().manual_seed(42)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                kwargs = dict(
                    num_layers=2,
                    num_slots=4,
                    num_heads=4,
                    head_v_dim=128,
                    head_k_dim=64,
                    conv_shapes=[(3, 64), (3, 32)],
                    conv_dtype=torch.bfloat16,
                    temporal_dtype=dtype,
                )
                pool = MambaCheckpointPool(device="npu", **kwargs)
                cache = SimpleNamespace(
                    temporal=torch.zeros(2, 5, 4, 128, 64, device="npu", dtype=dtype),
                    conv=[
                        torch.zeros(2, 5, *shape, device="npu", dtype=torch.bfloat16)
                        for shape in kwargs["conv_shapes"]
                    ],
                )
                active = SimpleNamespace(mamba_cache=cache)
                src = torch.tensor([1, 3], device="npu")
                dst = torch.tensor([2, 4], device="npu")
                original = (
                    torch.randn(2, 2, 4, 128, 64, generator=generator) * 0.06
                ).to(device="npu", dtype=dtype)
                cache.temporal[:, src] = original
                original_conv = []
                for conv in cache.conv:
                    values = torch.randn(2, 2, *conv.shape[2:], generator=generator).to(
                        device="npu", dtype=conv.dtype
                    )
                    conv[:, src] = values
                    original_conv.append(values)

                slots = pool.alloc(2)
                self.assertIsNotNone(slots)
                self.assertTrue((slots > 0).all().item())
                self.assertEqual(pool.available_size(), 2)
                pool.store_from_active(active, src, slots)
                self.assertEqual(pool.temporal.qdata.device.type, "npu")
                self.assertEqual(pool.temporal.qdata.dtype, torch.int8)
                # Reusing source slots must not change the cached checkpoint.
                cache.temporal[:, src] = 0
                for conv in cache.conv:
                    conv[:, src] = 0
                pool.load_to_active(active, slots, dst)
                restored = cache.temporal[:, dst].clone()
                relative_error = (
                    (restored.float() - original.float()).norm()
                    / original.float().norm()
                ).item()
                self.assertLess(relative_error, 0.01)
                for conv, values in zip(cache.conv, original_conv):
                    self.assertTrue(torch.equal(conv[:, dst], values))
                # Mutating one restored active copy must not corrupt another hit.
                cache.temporal[:, dst] = 1
                pool.load_to_active(active, slots, src)
                self.assertTrue(torch.equal(cache.temporal[:, src], restored))

                estimate = MambaCheckpointPool.estimate_mem_usage_bytes(**kwargs)
                self.assertEqual(estimate["total"], pool.mem_usage_bytes())
                native_temporal_bytes = (
                    pool.temporal.qdata.numel()
                    * torch.empty((), dtype=dtype).element_size()
                )
                ratio = pool.temporal.mem_usage_bytes() / native_temporal_bytes
                self.assertLess(ratio, 0.6)
                remaining = pool.alloc(2)
                self.assertIsNotNone(remaining)
                self.assertEqual(pool.available_size(), 0)
                self.assertIsNone(pool.alloc(1))
                pool.free(slots)
                recycled = pool.alloc(2)
                self.assertTrue(torch.equal(recycled, slots))
                pool.clear()
                self.assertEqual(pool.available_size(), 4)
                print(
                    f"NPU pool: dtype={dtype}, relative_error={relative_error:.8f}, "
                    f"temporal_memory_ratio={ratio:.6f}, bytes={pool.mem_usage_bytes()}"
                )


@unittest.skipUnless(is_npu(), "Requires Ascend NPU")
class TestNpuInt8MambaCheckpointLlama(CustomTestCase):
    """Parameter validation and no-op compatibility on the requested Llama model."""

    def setUp(self):
        self.model = os.environ.get(
            "SGLANG_TEST_MODEL_PATH", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        )
        self.assertEqual(AutoConfig.from_pretrained(self.model).model_type, "llama")
        log_dir = os.environ.get("SGLANG_TEST_LOG_DIR")
        if log_dir is None:
            directory = tempfile.TemporaryDirectory(prefix="npu-int8-mamba-")
            self.addCleanup(directory.cleanup)
            log_dir = directory.name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

    def test_incompatible_cache_options_rejected(self):
        for extra, message in [
            ({"enable_hierarchical_cache": True}, "host-offload path"),
            ({"radix_cache_backend": "custom"}, "built-in mamba"),
        ]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                ServerArgs(
                    model_path=self.model,
                    device="npu",
                    enable_int8_mamba_checkpoint=True,
                    **extra,
                ).resolve_once()

    def _run_server(self, enabled, prompts):
        label = "enabled" if enabled else "disabled"
        base_url = f"http://127.0.0.1:{get_open_port()}"
        args = [
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--dtype",
            "bfloat16",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.3",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "4096",
            "--max-running-requests",
            "4",
            "--random-seed",
            "0",
            "--disable-cuda-graph",
        ]
        if enabled:
            args += ["--enable-int8-mamba-checkpoint", "--int8-mamba-ckpt-size", "8"]
        log_path = self.log_dir / f"llama_{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = None
            try:
                process = popen_launch_server(
                    self.model,
                    base_url,
                    timeout=600,
                    other_args=args,
                    env={"TOKENIZERS_PARALLELISM": "false"},
                    return_stdout_stderr=(log, log),
                )
                response = self.session.get(f"{base_url}/server_info", timeout=30)
                response.raise_for_status()
                info = response.json()
                self.assertEqual(info["enable_int8_mamba_checkpoint"], enabled)
                self.assertEqual(info["device"], "npu")
                if enabled:
                    self.assertEqual(info["int8_mamba_ckpt_size"], 8)
                flush = self.session.post(f"{base_url}/flush_cache", timeout=30)
                flush.raise_for_status()
                rounds = []
                for name in ("cold", "warm"):
                    response = self.session.post(
                        f"{base_url}/generate",
                        json={
                            "text": prompts,
                            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                            "return_logprob": True,
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    results = response.json()
                    rounds.append(results)
                    (self.log_dir / f"llama_{label}.json").write_text(
                        json.dumps({"server_info": info, "rounds": rounds}, indent=2),
                        encoding="utf-8",
                    )
                    self.assertEqual(len(results), len(prompts))
                    for result in results:
                        self.assertTrue(result["text"].strip())
                        self.assertGreater(result["meta_info"]["completion_tokens"], 0)
                        if name == "warm":
                            self.assertGreater(result["meta_info"]["cached_tokens"], 0)
                    self.assertIn("paris", results[0]["text"].lower())
                    self.assertIn("4", results[1]["text"])
                self.assertEqual(
                    [r["text"] for r in rounds[0]], [r["text"] for r in rounds[1]]
                )
                log.flush()
                # This log marks pool construction. A dense Llama model must
                # not allocate a Mamba checkpoint pool just because the flag is set.
                self.assertNotIn("int8 mamba checkpoint pool:", log_path.read_text())
                return rounds
            except Exception:
                log.flush()
                print(log_path.read_text(errors="replace")[-16000:])
                raise
            finally:
                if process is not None:
                    kill_process_tree(process.pid)
                    process.wait(timeout=30)

    def test_llama_flag_on_off_is_compatible(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        # Ascend uses 128-token KV pages by default. Make the
                        # shared prefix long enough to leave a reusable page.
                        "content": "Answer the user's question briefly and accurately. "
                        * 24,
                    },
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question in (
                "What is the capital of France? Reply with only the city name.",
                "Calculate 2 + 2. Reply with only the integer answer.",
            )
        ]
        disabled = self._run_server(False, prompts)
        enabled = self._run_server(True, prompts)
        for baseline_round, enabled_round in zip(disabled, enabled):
            for baseline, actual in zip(baseline_round, enabled_round):
                self.assertEqual(actual["text"], baseline["text"])
                baseline_probs = baseline["meta_info"]["output_token_logprobs"]
                actual_probs = actual["meta_info"]["output_token_logprobs"]
                self.assertEqual(
                    [p[1] for p in actual_probs], [p[1] for p in baseline_probs]
                )
                torch.testing.assert_close(
                    torch.tensor([p[0] for p in actual_probs]),
                    torch.tensor([p[0] for p in baseline_probs]),
                    atol=1e-5,
                    rtol=1e-5,
                )
        print(
            "Llama flag off/on: matching greedy tokens/logprobs; KV cache hits verified."
        )


if __name__ == "__main__":
    unittest.main()
