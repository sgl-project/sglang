"""DSV4 Flash PD-disagg: fp8 two-pool vs bf16, same CLI besides the env.

Default transfer probe order is mooncake_tcp, mooncake, mori, nixl.
Force with SGLANG_TEST_PD_TRANSFER_BACKEND. MoE a2a stays none unless
SGLANG_TEST_PD_MOE_A2A=mori. GSM8K n/bar: SGLANG_TEST_PD_GSM8K_N (200),
SGLANG_TEST_PD_GSM8K_BAR (0.95).
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

DSV4_FLASH_MODEL_PATH = os.environ.get(
    "SGLANG_TEST_DSV4_FLASH_MODEL",
    "sgl-project/DeepSeek-V4-Flash-FP8",
)

DSV4_FLASH_ENV = {
    # local DeepSeek-V4-Flash snapshot has expert_dtype=fp4 in config.json;
    # handbook Flash-FP8 sets this to 0. 0 allocates an fp8 w13 (4096) and
    # then copy_ from packed fp4 (2048) blows up in fused_moe_triton.
    "SGLANG_DSV4_FP4_EXPERTS": os.environ.get("SGLANG_DSV4_FP4_EXPERTS", "1"),
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "SGLANG_OPT_USE_AITER_BATCHED_GEMM": "1",
    "SGLANG_USE_AITER": "1",
    # unset: aiter CK 2stage on small-M warmup has no kernel
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SGLANG_USE_ROCM700A": "0",
}

_PD_ARGS = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--dp",
    "4",
    "--enable-dp-attention",
    "--disable-shared-experts-fusion",
    "--cuda-graph-max-bs-decode",
    "128",
    "--max-running-requests",
    "256",
    "--mem-fraction-static",
    "0.7",
    "--attention-backend",
    "dsv4",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--watchdog-timeout",
    "900",
]


def _probe_transfer(name: str) -> None:
    if name in ("mooncake", "mooncake_tcp"):
        import mooncake  # noqa: F401
    elif name == "mori":
        import mori.io  # noqa: F401
    elif name == "nixl":
        from nixl._api import nixl_agent  # noqa: F401
    else:
        raise ImportError(f"unknown PD transfer backend {name}")


def _transfer_backend_args():
    # mooncake_tcp first: single-node P+D over RDMA has failed ibv_reg_mr on the
    # NICs we have. Cross-node still wants mooncake; nixl hits the same thing.
    forced = os.environ.get("SGLANG_TEST_PD_TRANSFER_BACKEND")
    order = []
    if forced:
        order.append(forced)
    for name in ("mooncake_tcp", "mooncake", "mori", "nixl"):
        if name not in order:
            order.append(name)
    errors = []
    for name in order:
        try:
            _probe_transfer(name)
            print(f"PD transfer backend={name}")
            return ["--disaggregation-transfer-backend", name]
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("no PD transfer backend importable: " + "; ".join(errors))


def _moe_a2a_args():
    # default none = fused TP-MoE. mori EP is an overlay, not DeepEP.
    forced = os.environ.get("SGLANG_TEST_PD_MOE_A2A", "none")
    if forced in ("", "none"):
        print("PD moe-a2a-backend=none")
        return []
    if forced != "mori":
        print(f"PD moe-a2a-backend={forced} (passthrough)")
        return ["--moe-a2a-backend", forced]
    try:
        import mori  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"SGLANG_TEST_PD_MOE_A2A=mori but import failed: {e}") from e
    print("PD moe-a2a-backend=mori ep-size=4")
    return [
        "--moe-a2a-backend",
        "mori",
        "--ep-size",
        "4",
        "--deepep-mode",
        "normal",
    ]


class _DSV4FlashPDBase(PDDisaggregationServerBase):
    # pytest collects unittest.TestCase subclasses even when the name is _Foo.
    __test__ = False
    moe_a2a_args = []
    transfer_backend = []
    capture_per_side_logs = False

    @classmethod
    def _arm_env(cls):
        env = dict(DSV4_FLASH_ENV)
        env["SGLANG_DSV4_UNIFIED_KV_FP8"] = "1" if cls.fp8 else "0"
        backend = ""
        if getattr(cls, "transfer_backend", None):
            backend = cls.transfer_backend[-1]
        if backend == "nixl":
            env.setdefault("UCX_TLS", "tcp,sm,self,rocm")
        if backend == "mooncake":
            env.setdefault("MC_GID_INDEX", "1")
        if "--moe-a2a-backend" in getattr(cls, "moe_a2a_args", []):
            env.setdefault("SGLANG_MORI_RECV_BOUND", "1")
            # FP4 experts on this snapshot; cookbook: both env vars for FP4.
            env.setdefault("SGLANG_MORI_DISPATCH_DTYPE", "mxfp8")
        return env

    @classmethod
    def setUpClass(cls):
        try:
            import torch

            n = torch.cuda.device_count()
        except Exception as e:
            raise unittest.SkipTest(f"cuda unavailable: {e}") from e
        if n < 8:
            raise unittest.SkipTest(f"Flash PD E2E needs 8 GPUs, got {n}")
        super().setUpClass()
        cls.transfer_backend = _transfer_backend_args()
        cls.moe_a2a_args = _moe_a2a_args()
        rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")
        cls.rdma_devices = ["--disaggregation-ib-device", rdma_env] if rdma_env else []
        cls.model = DSV4_FLASH_MODEL_PATH
        if cls.capture_per_side_logs:
            cls._open_side_log_files()
        # P+D together has SIGKILL'd around 7 min while /health was still 503.
        ready_timeout = int(os.environ.get("SGLANG_TEST_PD_READY_TIMEOUT", "3600"))
        cls.start_prefill()
        try:
            cls.wait_server_ready(
                cls.prefill_url + "/health",
                timeout=ready_timeout,
                process=cls.process_prefill,
            )
            cls.start_decode()
            cls.wait_server_ready(
                cls.decode_url + "/health",
                timeout=ready_timeout,
                process=cls.process_decode,
            )
        except Exception:
            cls._dump_side_logs()
            raise
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--nccl-port",
            cls.prefill_nccl_port,
            "--base-gpu-id",
            "0",
            *_PD_ARGS,
            *cls.moe_a2a_args,
            *cls.transfer_backend,
            *cls.rdma_devices,
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=cls._arm_env(),
            return_stdout_stderr=(
                (cls._prefill_stdout_buf, cls._prefill_stderr_buf)
                if cls.capture_per_side_logs
                else None
            ),
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--nccl-port",
            cls.decode_nccl_port,
            "--base-gpu-id",
            "4",
            *_PD_ARGS,
            *cls.moe_a2a_args,
            *cls.transfer_backend,
            *cls.rdma_devices,
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=cls._arm_env(),
            return_stdout_stderr=(
                (cls._decode_stdout_buf, cls._decode_stderr_buf)
                if cls.capture_per_side_logs
                else None
            ),
        )

    @classmethod
    def _open_side_log_files(cls):
        # StringIO dies with SIGKILL; a real file keeps the last lines.
        logdir = os.environ.get("SGLANG_TEST_PD_LOGDIR")
        cls._log_files = []
        if not logdir:
            return
        os.makedirs(logdir, exist_ok=True)
        arm = "fp8" if getattr(cls, "fp8", False) else "bf16"
        for side in ("prefill", "decode"):
            for stream in ("stdout", "stderr"):
                path = os.path.join(logdir, f"pd_e2e_{arm}_{side}.{stream}.log")
                fh = open(path, "w", buffering=1)
                setattr(cls, f"_{side}_{stream}_buf", fh)
                cls._log_files.append(fh)

    @classmethod
    def _dump_side_logs(cls):
        logdir = os.environ.get("SGLANG_TEST_PD_LOGDIR")
        if not logdir:
            return
        arm = "fp8" if getattr(cls, "fp8", False) else "bf16"
        for fh in getattr(cls, "_log_files", []):
            try:
                fh.flush()
            except Exception:
                pass
        for side in ("prefill", "decode"):
            for stream in ("stdout", "stderr"):
                path = os.path.join(logdir, f"pd_e2e_{arm}_{side}.{stream}.log")
                try:
                    with open(path) as f:
                        text = f.read()
                except OSError:
                    text = ""
                tail = text[-4000:]
                if tail:
                    print(f"===== {side} {stream} (tail) =====\n{tail}")

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            for fh in getattr(cls, "_log_files", []):
                try:
                    fh.close()
                except Exception:
                    pass

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=int(os.environ.get("SGLANG_TEST_PD_GSM8K_N", "200")),
            max_new_tokens=512,
            parallel=64,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_gsm8k_eval(args)
        print(f"{metrics=}")
        bar = float(os.environ.get("SGLANG_TEST_PD_GSM8K_BAR", "0.95"))
        self.assertGreater(metrics["accuracy"], bar)

    def _recall_marker(self, marker, filler_tokens, marker_first):
        filler = " ".join(f"tok{i}" for i in range(filler_tokens))
        stated = f"The secret marker is {marker}."
        body = f"{stated}\n{filler}" if marker_first else f"{filler}\n{stated}"
        r = requests.post(
            f"{self.lb_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": f"{body}\nRepeat the secret marker and nothing else:",
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def test_rope_probe_recalls_marker_in_swa_window(self):
        # Recent marker: decode answers out of the SWA ring, which ships as
        # StateType.SWA_RING. GSM8K can pass with stale rope, this can't.
        text = self._recall_marker("74931", 64, marker_first=False)
        print(f"rope_probe swa text={text!r}")
        self.assertIn("74931", text)

    def test_rope_probe_recalls_marker_beyond_swa_window(self):
        # Same probe with the marker pushed out of the window, so the answer has
        # to come from the compressed pages -- the groups fp8 splits in two. Bump
        # SGLANG_TEST_PD_ROPE_PROBE_TOKENS if the model's window grows; if the
        # window probe above passes and this one fails, suspect rope, and only
        # then the model losing the needle.
        filler = int(os.environ.get("SGLANG_TEST_PD_ROPE_PROBE_TOKENS", "2048"))
        text = self._recall_marker("58207", filler, marker_first=True)
        print(f"rope_probe compressed text={text!r}")
        self.assertIn("58207", text)


class TestDSV4FlashPDDisaggFp8(_DSV4FlashPDBase):
    __test__ = True
    fp8 = True


class TestDSV4FlashPDDisaggBf16(_DSV4FlashPDBase):
    __test__ = True
    fp8 = False


if __name__ == "__main__":
    unittest.main()
