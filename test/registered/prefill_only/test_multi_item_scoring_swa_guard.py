"""GPU end-to-end and HTTP tests for SWA + --enable-mis guards."""

import gc
import os
import socket
import tempfile
import unittest

import requests
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Cohere2Config,
    Cohere2ForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    PreTrainedTokenizerFast,
)

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=85, stage="base-b", runner_config="1-gpu-small")

_VOCAB_SIZE = 12000
_LABELS = [10, 11, 12, 13]
_W_SWA = 32
_MEM_FRACTION_STATIC = float(os.environ.get("SGLANG_TEST_MEM_FRACTION_STATIC", "0.05"))


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _save_tokenizer(path: str) -> None:
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    for idx in range(3, _VOCAB_SIZE):
        vocab[f"t{idx}"] = idx
    tk = Tokenizer(WordLevel(vocab=vocab, unk_token="<pad>"))
    tk.pre_tokenizer = Whitespace()
    fast_tk = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    fast_tk.save_pretrained(path)


def _save_synthetic_checkpoint(model: torch.nn.Module, path: str) -> None:
    for param in model.parameters():
        if param.ndim >= 2:
            torch.nn.init.normal_(param, mean=0.0, std=0.12)
    model.to(torch.float16).save_pretrained(path)
    _save_tokenizer(path)


def _build_gemma2_swa(path: str) -> None:
    torch.manual_seed(42)
    cfg = Gemma2Config(
        vocab_size=_VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        sliding_window=_W_SWA,
        max_position_embeddings=2048,
        hidden_act="gelu_pytorch_tanh",
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    _save_synthetic_checkpoint(Gemma2ForCausalLM(cfg), path)


def _build_gemma3(path: str, *, layer_types: list[str]) -> None:
    torch.manual_seed(42)
    cfg = Gemma3TextConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        query_pre_attn_scalar=64,
        sliding_window=_W_SWA,
        max_position_embeddings=2048,
        layer_types=layer_types,
        hidden_activation="gelu_pytorch_tanh",
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    _save_synthetic_checkpoint(Gemma3ForCausalLM(cfg), path)


def _build_cohere2_swa(path: str) -> None:
    torch.manual_seed(42)
    cfg = Cohere2Config(
        vocab_size=_VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=_W_SWA,
        max_position_embeddings=2048,
        layer_types=["sliding_attention"] * 4,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    _save_synthetic_checkpoint(Cohere2ForCausalLM(cfg), path)


def _launch_engine(*, model_path: str, enable_mis: bool) -> Engine:
    return Engine(
        model_path=model_path,
        dtype="float16",
        mem_fraction_static=_MEM_FRACTION_STATIC,
        attention_backend="flashinfer",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        enable_mis=enable_mis,
        log_level="error",
    )


def _shutdown_engine(engine: Engine) -> None:
    engine.shutdown()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _allowed_regimes(
    *, expected_window: int
) -> dict[str, tuple[list[int], list[list[int]]]]:
    return {
        "regime_a": (
            [20 + (i % 50) for i in range(12)],
            [[100, 101, 102, 103], [110, 111, 112, 113]],
        ),
        "regime_b": (
            [20 + (i % 50) for i in range(16)],
            [[100 + j * 10 + k for k in range(8)] for j in range(6)],
        ),
        "boundary": (
            [20 + (i % 50) for i in range(expected_window)],
            [[101]],
        ),
    }


def _collect_pointwise_ref(
    *,
    model_dir: str,
    regimes: dict[str, tuple[list[int], list[list[int]]]],
    expected_window: int,
) -> tuple[dict[str, torch.Tensor], int]:
    engine = _launch_engine(model_path=model_dir, enable_mis=False)
    try:
        refs = {}
        for name, (query, items) in regimes.items():
            rows = []
            for item in items:
                res = engine.score(
                    query=query,
                    items=[item],
                    label_token_ids=_LABELS,
                    apply_softmax=False,
                    return_token_logprobs=True,
                )
                rows.append(res.token_logprobs[0])
            refs[name] = torch.tensor(rows, dtype=torch.float32)

        non_mis_req = GenerateReqInput(
            input_ids=[20] * (expected_window + 5),
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            multi_item_delimiter_indices=[expected_window + 1, expected_window + 3],
        )
        non_mis_out = engine.loop.run_until_complete(
            engine.tokenizer_manager.generate_request(non_mis_req, None).__anext__()
        )
        return refs, len(non_mis_out["output_ids"])
    finally:
        _shutdown_engine(engine)


class TestSWAMultiItemScoringGuardE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        root = cls._tmp_dir.name
        cls.gemma2_dir = os.path.join(root, "gemma2_swa")
        cls.gemma3_swa_dir = os.path.join(root, "gemma3_swa")
        cls.cohere2_dir = os.path.join(root, "cohere2_swa")
        cls.gemma3_full_dir = os.path.join(root, "gemma3_full")
        _build_gemma2_swa(cls.gemma2_dir)
        _build_gemma3(cls.gemma3_swa_dir, layer_types=["sliding_attention"] * 4)
        _build_cohere2_swa(cls.cohere2_dir)
        _build_gemma3(cls.gemma3_full_dir, layer_types=["full_attention"] * 4)

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    def _verify_single_swa_path(self, *, model_dir: str, expected_window: int) -> None:
        regimes = _allowed_regimes(expected_window=expected_window)
        refs, non_mis_out_len = _collect_pointwise_ref(
            model_dir=model_dir, regimes=regimes, expected_window=expected_window
        )
        self.assertEqual(non_mis_out_len, 1)
        engine = _launch_engine(model_path=model_dir, enable_mis=True)
        try:
            self.assertEqual(
                engine.tokenizer_manager.min_positive_swa_window,
                expected_window,
            )
            overflow_cases = (
                (
                    [20 + (i % 50) for i in range(64)],
                    [[100 + j * 10 + k for k in range(8)] for j in range(4)],
                ),
                ([20 + (i % 50) for i in range(expected_window)], [[101, 102]]),
                ([20 + (i % 50) for i in range(expected_window + 1)], [[101]]),
            )
            for bad_query, bad_items in overflow_cases:
                with self.assertRaises(ValueError):
                    engine.score(
                        query=bad_query,
                        items=bad_items,
                        label_token_ids=_LABELS,
                        apply_softmax=False,
                    )

            direct_bad_req = GenerateReqInput(
                input_ids=[20] * (expected_window + 4),
                sampling_params={"max_new_tokens": 0},
                return_logprob=True,
                token_ids_logprob=_LABELS,
                multi_item_delimiter_indices=[
                    expected_window + 1,
                    expected_window + 3,
                ],
            )
            with self.assertRaisesRegex(ValueError, "window_left"):
                engine.loop.run_until_complete(
                    engine.tokenizer_manager.generate_request(
                        direct_bad_req, None
                    ).__anext__()
                )

            for name, (query, items) in regimes.items():
                res = engine.score(
                    query=query,
                    items=items,
                    label_token_ids=_LABELS,
                    apply_softmax=False,
                    return_token_logprobs=True,
                )
                mis_lps = torch.tensor(res.token_logprobs, dtype=torch.float32)
                self.assertTrue(torch.isfinite(mis_lps).all())
                torch.testing.assert_close(mis_lps, refs[name], rtol=1e-2, atol=5e-3)
        finally:
            _shutdown_engine(engine)

    def test_swa_paths_boundary_reject_and_same_engine_recovery(self):
        cases = [
            ("HybridSWA-Gemma2", self.gemma2_dir, 31),
            ("AllSliding-Gemma3", self.gemma3_swa_dir, 31),
            ("LayerOnlySWA-Cohere2", self.cohere2_dir, 32),
        ]
        for label, model_dir, expected_window in cases:
            with self.subTest(model=label):
                self._verify_single_swa_path(
                    model_dir=model_dir, expected_window=expected_window
                )

    def test_all_full_gemma3_negative_control(self):
        engine = _launch_engine(model_path=self.gemma3_full_dir, enable_mis=True)
        try:
            self.assertIsNone(engine.tokenizer_manager.min_positive_swa_window)
            res = engine.score(
                query=[20] * 64,
                items=[[100, 101], [102, 103]],
                label_token_ids=_LABELS,
                apply_softmax=False,
            )
            self.assertEqual(len(res.scores), 2)
            self.assertTrue(torch.isfinite(torch.tensor(res.scores)).all())
        finally:
            _shutdown_engine(engine)

    def test_live_http_score_rejects_400_and_recovers_200(self):
        base_url = f"http://127.0.0.1:{_pick_free_port()}"
        proc = popen_launch_server(
            self.gemma3_swa_dir,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dtype",
                "float16",
                "--mem-fraction-static",
                str(_MEM_FRACTION_STATIC),
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--enable-mis",
                "--attention-backend",
                "flashinfer",
            ],
        )
        try:
            ok_payload = {
                "query": [20] * 31,
                "items": [[101]],
                "label_token_ids": _LABELS,
                "apply_softmax": False,
            }
            resp_ok = requests.post(f"{base_url}/v1/score", json=ok_payload)
            self.assertEqual(resp_ok.status_code, 200)
            baseline = torch.tensor(resp_ok.json()["scores"])
            self.assertTrue(torch.isfinite(baseline).all())

            for bad_query, bad_items in (
                ([20] * 32, [[101]]),
                ([20] * 31, [[101, 102]]),
            ):
                resp_bad = requests.post(
                    f"{base_url}/v1/score",
                    json={
                        "query": bad_query,
                        "items": bad_items,
                        "label_token_ids": _LABELS,
                        "apply_softmax": False,
                    },
                )
                self.assertEqual(resp_bad.status_code, 400)
                self.assertIn("window_left", resp_bad.text)

            gen_bad = requests.post(
                f"{base_url}/generate",
                json={
                    "input_ids": [20] * 35,
                    "sampling_params": {"max_new_tokens": 0},
                    "return_logprob": True,
                    "token_ids_logprob": _LABELS,
                    "multi_item_delimiter_indices": [32, 34],
                },
            )
            self.assertEqual(gen_bad.status_code, 400)
            self.assertIn("window_left", gen_bad.text)

            gen_ok = requests.post(
                f"{base_url}/generate",
                json={
                    "input_ids": [20] * 34,
                    "sampling_params": {"max_new_tokens": 0},
                    "return_logprob": True,
                    "token_ids_logprob": _LABELS,
                    "multi_item_delimiter_indices": [31, 33],
                },
            )
            self.assertEqual(gen_ok.status_code, 200)

            resp_recovered = requests.post(f"{base_url}/v1/score", json=ok_payload)
            self.assertEqual(resp_recovered.status_code, 200)
            torch.testing.assert_close(
                torch.tensor(resp_recovered.json()["scores"]), baseline
            )
        finally:
            terminate_and_kill_process_tree(proc)


if __name__ == "__main__":
    unittest.main()
