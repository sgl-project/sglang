"""Exercise native gRPC tokenization without loading model weights."""

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import grpc
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
)
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.rust_extensions import load_rust_extension
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestNativeGrpcTokenize(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.native_grpc = load_rust_extension("sglang.srt.rust_extensions._grpc")
        proto_dir = (
            Path(__file__).resolve().parents[4] / "proto" / "sglang" / "runtime" / "v1"
        )
        with patch.object(sys, "path", [str(proto_dir), *sys.path]):
            cls.proto, cls.services = grpc.protos_and_services("sglang.proto")

        tokenizer_dir = TemporaryDirectory()
        cls.addClassCleanup(tokenizer_dir.cleanup)
        cls.tokenizer_path = tokenizer_dir.name
        backend = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "hello": 1, "<|": 2, "mask": 3, "|>": 4, "[BOS]": 5},
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.decoder = decoders.Fuse()
        backend.post_processor = processors.TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", 5)]
        )
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            bos_token="[BOS]",
        )
        cls.tokenizer.add_tokens([AddedToken("<|mask|>", normalized=False)])
        cls.mask_id = cls.tokenizer.convert_tokens_to_ids("<|mask|>")
        cls.tokenizer.save_pretrained(cls.tokenizer_path)

    @contextmanager
    def _server(self, *, dllm_enabled=True, tokenizer_mode="auto"):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = self.tokenizer
        manager.server_args = SimpleNamespace(
            tokenizer_path=self.tokenizer_path,
            tokenizer_mode=tokenizer_mode,
            dllm_algorithm="joint_threshold" if dllm_enabled else None,
        )
        manager.model_config = SimpleNamespace(context_len=128)
        with patch(
            "sglang.srt.managers.tokenizer_manager.get_dllm_model_params",
            return_value={"mask_id": self.mask_id},
        ):
            manager._init_dllm_prompt_handling()
        manager.context_len = 128
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.is_generation = True
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = manager
        port = find_available_port(30000)
        server = self.native_grpc.start_server("127.0.0.1", port, handle)
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                grpc.channel_ready_future(channel).result(timeout=10)
                yield self.services.SglangServiceStub(channel), handle, manager
        finally:
            server.shutdown()

    def test_tokenize_returns_normalized_reusable_ids(self):
        """Native tokenization must not return mask IDs rejected by generation."""
        for tokenizer_mode in ("auto", "slow"):
            with (
                self.subTest(tokenizer_mode=tokenizer_mode),
                self._server(tokenizer_mode=tokenizer_mode) as (stub, _, manager),
            ):
                for add_special_tokens in (None, False, True):
                    with self.subTest(add_special_tokens=add_special_tokens):
                        response = stub.Tokenize(
                            self.proto.TokenizeRequest(
                                text="<|mask|>", add_special_tokens=add_special_tokens
                            ),
                            timeout=10,
                        )
                        expected_ids = [2, 3, 4]
                        if add_special_tokens is not False:
                            expected_ids.insert(0, 5)
                        self.assertEqual(list(response.tokens), expected_ids)
                        self.assertEqual(response.count, len(expected_ids))
                        self.assertEqual(response.max_model_len, 128)
                        self.assertEqual(response.input_text, "<|mask|>")
                        manager._validate_one_request(
                            SimpleNamespace(sampling_params={}), list(response.tokens)
                        )

    def test_tokenize_keeps_native_path_when_normalization_is_unnecessary(self):
        """Neither ordinary dLLM text nor non-dLLM masks need Python fallback."""
        for dllm_enabled, text in ((True, "hello"), (False, "<|mask|>")):
            with (
                self.subTest(dllm_enabled=dllm_enabled),
                self._server(dllm_enabled=dllm_enabled) as (stub, handle, _),
            ):
                with patch.object(
                    handle,
                    "tokenize",
                    side_effect=AssertionError("Unexpected Python tokenization"),
                ):
                    response = stub.Tokenize(
                        self.proto.TokenizeRequest(text=text), timeout=10
                    )
                expected_ids = self.tokenizer.encode(text)
                self.assertEqual(list(response.tokens), expected_ids)
                self.assertEqual(response.count, len(expected_ids))
                with patch.object(
                    handle,
                    "detokenize",
                    side_effect=AssertionError("Unexpected Python detokenization"),
                ):
                    response = stub.Detokenize(
                        self.proto.DetokenizeRequest(tokens=[1]), timeout=10
                    )
                self.assertEqual(response.text, "hello")


if __name__ == "__main__":
    unittest.main()
