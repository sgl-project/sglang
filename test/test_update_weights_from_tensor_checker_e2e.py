import base64
import hashlib
import os
import pickle
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import requests
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
TRANSFORMER_MODULE = "transformer"
NUM_TENSORS_TO_UPDATE = 4
SERVER_READY_MESSAGE = "Application startup complete."


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _list_safetensor_files(model_dir: str) -> list[Path]:
    return sorted(Path(model_dir).glob("*.safetensors"))


def _iter_transformer_weights_from_disk(model_path: str):
    local_model_path = snapshot_download(
        repo_id=model_path,
        allow_patterns=[f"{TRANSFORMER_MODULE}/*"],
    )
    weights_dir = Path(local_model_path) / TRANSFORMER_MODULE
    safetensor_files = _list_safetensor_files(str(weights_dir))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No safetensors files found for module '{TRANSFORMER_MODULE}' in {weights_dir}"
        )

    for safetensor_file in safetensor_files:
        with safe_open(str(safetensor_file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def _compute_tensor_sha256(tensor: torch.Tensor) -> str:
    materialized = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(materialized.dtype).encode("utf-8"))
    hasher.update(repr(tuple(materialized.shape)).encode("utf-8"))
    hasher.update(materialized.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _build_named_tensor_sha256(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> dict[str, str]:
    return {name: _compute_tensor_sha256(tensor) for name, tensor in named_tensors}


def _select_transformer_tensors(
    model_path: str,
    max_tensors: int = NUM_TENSORS_TO_UPDATE,
) -> list[tuple[str, torch.Tensor]]:
    selected_named_tensors: list[tuple[str, torch.Tensor]] = []
    for name, tensor in _iter_transformer_weights_from_disk(model_path):
        if not tensor.is_floating_point():
            continue
        selected_named_tensors.append((name, tensor.to(torch.bfloat16).clone()))
        if len(selected_named_tensors) == max_tensors:
            break

    if not selected_named_tensors:
        raise AssertionError("Expected at least one floating-point transformer tensor")

    return selected_named_tensors


def _build_shifted_named_tensors(
    named_tensors: list[tuple[str, torch.Tensor]],
    delta: float,
) -> list[tuple[str, torch.Tensor]]:
    shifted_named_tensors: list[tuple[str, torch.Tensor]] = []
    for index, (name, tensor) in enumerate(named_tensors):
        shifted_tensor = tensor.clone()
        if index == 0:
            shifted_tensor.add_(delta)
        shifted_named_tensors.append((name, shifted_tensor))
    return shifted_named_tensors


def _serialize_named_tensors(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    return base64.b64encode(pickle.dumps(named_tensors)).decode("utf-8")


class _ServerRunner:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.port = _get_free_port()
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.log_path = Path(tempfile.gettempdir()) / (
            f"sglang_update_weight_checker_{self.port}.log"
        )
        self.log_path.unlink(missing_ok=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        command = [
            "sglang",
            "serve",
            "--model-path",
            self.model_path,
            "--port",
            str(self.port),
            "--log-level=debug",
            "--num-gpus",
            "1",
        ]
        env = os.environ.copy()
        env["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"

        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
        try:
            self._wait_for_ready()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self.process is None:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except Exception:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        self.process = None

    def _wait_for_ready(self) -> None:
        deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200"))
        start_time = time.time()

        while time.time() - start_time < deadline:
            if self.process is None:
                raise RuntimeError("Server process was not started")
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Server exited early with code {self.process.returncode}.\n"
                    f"{self._get_log_tail()}"
                )
            if self.log_path.exists():
                log_content = self.log_path.read_text(encoding="utf-8", errors="ignore")
                if SERVER_READY_MESSAGE in log_content:
                    return
            time.sleep(1)

        raise TimeoutError(
            f"Server did not become ready within {deadline}s.\n{self._get_log_tail()}"
        )

    def _get_log_tail(self, lines: int = 200) -> str:
        if not self.log_path.exists():
            return ""
        log_content = self.log_path.read_text(encoding="utf-8", errors="ignore")
        return "\n".join(log_content.splitlines()[-lines:])


class UpdateWeightFromTensorCheckerE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshot_download(repo_id=BASE_MODEL)
        cls.base_transformer_tensors = _select_transformer_tensors(BASE_MODEL)
        cls.expected_updated_tensors = _build_shifted_named_tensors(
            cls.base_transformer_tensors,
            delta=1.0,
        )
        cls.server = _ServerRunner(BASE_MODEL)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _update_weights_from_disk(self, model_path: str) -> tuple[dict, int]:
        response = requests.post(
            f"{self.server.base_url}/update_weights_from_disk",
            json={"model_path": model_path, "target_modules": [TRANSFORMER_MODULE]},
            timeout=300,
        )
        return response.json(), response.status_code

    def _update_weights_from_tensor(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
    ) -> tuple[dict, int]:
        response = requests.post(
            f"{self.server.base_url}/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [_serialize_named_tensors(named_tensors)],
                "target_modules": [TRANSFORMER_MODULE],
            },
            timeout=300,
        )
        return response.json(), response.status_code

    def _check_updated_weights_from_tensor(
        self,
        expected_transformer_sha256: dict[str, str],
    ) -> tuple[dict, int]:
        response = requests.post(
            f"{self.server.base_url}/update_weights_from_tensor_checker",
            json={"expected_transformer_sha256": [expected_transformer_sha256]},
            timeout=300,
        )
        return response.json(), response.status_code

    def test_update_weights_from_tensor_checker_success(self):
        reset_result, reset_status = self._update_weights_from_disk(BASE_MODEL)
        self.assertEqual(reset_status, 200, reset_result)
        self.assertTrue(reset_result.get("success"), reset_result)

        update_result, update_status = self._update_weights_from_tensor(
            self.expected_updated_tensors
        )
        self.assertEqual(update_status, 200, update_result)
        self.assertTrue(update_result.get("success"), update_result)

        expected_transformer_sha256 = _build_named_tensor_sha256(
            self.expected_updated_tensors
        )
        check_result, check_status = self._check_updated_weights_from_tensor(
            expected_transformer_sha256
        )
        self.assertEqual(check_status, 200, check_result)
        self.assertTrue(check_result.get("success"), check_result)
        self.assertIn("Verified transformer update", check_result.get("message", ""))

    def test_update_weights_from_tensor_checker_detects_corrupted_payload(self):
        reset_result, reset_status = self._update_weights_from_disk(BASE_MODEL)
        self.assertEqual(reset_status, 200, reset_result)
        self.assertTrue(reset_result.get("success"), reset_result)

        expected_transformer_sha256 = _build_named_tensor_sha256(
            self.expected_updated_tensors
        )
        corrupted_named_tensors = _build_shifted_named_tensors(
            self.base_transformer_tensors,
            delta=2.0,
        )

        update_result, update_status = self._update_weights_from_tensor(
            corrupted_named_tensors
        )
        self.assertEqual(update_status, 200, update_result)
        self.assertTrue(update_result.get("success"), update_result)

        check_result, check_status = self._check_updated_weights_from_tensor(
            expected_transformer_sha256
        )
        self.assertEqual(check_status, 400, check_result)
        self.assertFalse(check_result.get("success", True), check_result)
        self.assertIn("checksum mismatch", check_result.get("message", ""))
        self.assertIn(
            self.expected_updated_tensors[0][0],
            check_result.get("message", ""),
        )


if __name__ == "__main__":
    unittest.main()
