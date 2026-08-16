#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the fixed SenseNova U1 warmed-performance acceptance matrix."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from bench_sensenova_u1_interleave import DEFAULT_SYSTEM_MESSAGE
from PIL import Image

DEFAULT_BASE_URL = "http://127.0.0.1:31002"
DEFAULT_OUT_DIR = "outputs/sensenova_u1_main/warmed_perf_20260816"
DEFAULT_HF_T2I_IMAGE = "outputs/sensenova_u1_main/image_graph_20260816/hf_256/t2i.png"
DEFAULT_HF_INTERLEAVE_IMAGE = (
    "/mnt/afs/fanyijiat/sensenova_u1_sglang_omni/outputs/m6/"
    "warmed_image_speedgate_default_no_env_fixed_work_256_steps2_tokens126_"
    "20260815/hf/interleave.png"
)

TEXT_INPUT_IDS = [
    151644,
    872,
    198,
    7985,
    264,
    11682,
    48826,
    52573,
    369,
    20045,
    264,
    22670,
    44378,
    71730,
    13,
    13655,
    1817,
    1509,
    14175,
    323,
    24586,
    13,
    151645,
    198,
    151644,
    77091,
    198,
]
TEXT_EXPECTED_IDS = [
    151667,
    198,
    785,
    1196,
    374,
    10161,
    369,
    264,
    11682,
    48826,
    52573,
    369,
    20045,
    264,
    22670,
    44378,
    71730,
    13,
    1096,
    374,
    264,
    10916,
    2197,
    429,
    1265,
    387,
    14976,
    11,
    91078,
    11,
    323,
    15817,
]
TEXT_HF_TOKENS_PER_S = {
    1: 23.334656180265792,
    8: 23.277544939885093,
    16: 23.374976421265412,
}
TEXT_TARGET_TOKENS_PER_S = {
    1: 58.451309878651436,
    8: 387.15448121146744,
    16: 729.8184413695169,
}
TEXT_TARGET_SPEEDUP = {
    1: 2.504914125457907,
    8: 16.632101117678204,
    16: 31.222210804266897,
}

T2I_PROMPT = "A small red cube on a plain white table, studio lighting."
T2I_HF_SECONDS = 0.16035353764891624
T2I_TARGET_SECONDS = 0.05010858178138733
T2I_TARGET_SPEEDUP = 3.200121255646454

INTERLEAVE_PROMPT = (
    "Write a promotional copy for a beachfront villa. Interleave images of the "
    "villa's exterior, the infinity pool, and the ocean view from the master "
    "bedroom."
)
INTERLEAVE_HF_SECONDS = 6.296433586627245
INTERLEAVE_TARGET_SECONDS = 2.3506999872624874
INTERLEAVE_TARGET_SPEEDUP = 2.6785355939699347

IMG_CONTEXT_ID = 151669
IMG_START_ID = 151670
IMG_END_ID = 151671
IMAGE_SPAN_TOKENS = 66
MIN_PSNR_DB = 46.0
MIN_SSIM = 0.999


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_state() -> dict[str, Any]:
    return {
        "head": _command_output(["git", "rev-parse", "HEAD"]),
        "branch": _command_output(["git", "branch", "--show-current"]),
        "status": _command_output(["git", "status", "--short"]),
    }


def _gpu_state() -> str | None:
    return _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate",
            "--format=csv,noheader",
        ]
    )


def _png_metrics(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = np.asarray(
        Image.open(reference_path).convert("RGB"),
        dtype=np.float64,
    )
    candidate = np.asarray(
        Image.open(candidate_path).convert("RGB"),
        dtype=np.float64,
    )
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"image shape mismatch: {reference.shape} != {candidate.shape}"
        )
    diff = candidate - reference
    mse = float(np.mean(diff**2))
    psnr = float("inf") if mse == 0 else 10 * math.log10((255**2) / mse)
    x = reference.reshape(-1)
    y = candidate.reshape(-1)
    ux = float(x.mean())
    uy = float(y.mean())
    vx = float(((x - ux) ** 2).mean())
    vy = float(((y - uy) ** 2).mean())
    covariance = float(((x - ux) * (y - uy)).mean())
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * ux * uy + c1) * (2 * covariance + c2)) / (
        (ux**2 + uy**2 + c1) * (vx + vy + c2)
    )
    return {
        "width": int(reference.shape[1]),
        "height": int(reference.shape[0]),
        "mse_uint8": mse,
        "psnr_db": psnr,
        "ssim_global_rgb": float(ssim),
        "pixel_max_abs_diff_uint8": int(np.abs(diff).max()),
        "pixel_mean_abs_diff_uint8": float(np.abs(diff).mean()),
        "reference_png_sha256": _sha256(reference_path.read_bytes()),
        "candidate_png_sha256": _sha256(candidate_path.read_bytes()),
    }


def _non_null_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value if item is not None]
    return [] if value is None else [value]


class AcceptanceRunner:
    def __init__(
        self,
        *,
        base_url: str,
        out_dir: Path,
        measurement_repeats: int,
        timeout_s: float,
        hf_t2i_image: Path,
        hf_interleave_image: Path,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.out_dir = out_dir
        self.measurement_repeats = measurement_repeats
        self.timeout_s = timeout_s
        self.hf_t2i_image = hf_t2i_image
        self.hf_interleave_image = hf_interleave_image
        self.session = requests.Session()
        self.session.trust_env = False

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[Any, float]:
        started = time.perf_counter()
        response = self.session.post(
            self.base_url + path,
            json=payload,
            timeout=self.timeout_s,
        )
        wall_s = time.perf_counter() - started
        response.raise_for_status()
        return response.json(), wall_s

    def health(self) -> dict[str, Any]:
        response = self.session.get(
            self.base_url + "/health",
            timeout=min(self.timeout_s, 30),
        )
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "body": response.text,
        }

    def text_request(
        self,
        batch_size: int,
        label: str,
        *,
        exact_bs1: bool,
        decode_steps: int = 32,
    ) -> dict[str, Any]:
        sampling_params: dict[str, Any] = {
            "temperature": 0,
            "max_new_tokens": decode_steps,
            "ignore_eos": True,
        }
        if exact_bs1:
            if batch_size != 1:
                raise ValueError("the exact text path is only valid for batch size 1")
            sampling_params["custom_params"] = {
                "sensenova_u1_exact_text": {
                    "decode_steps": decode_steps,
                    "img_start_token_id": IMG_START_ID,
                    "eos_token_ids": [],
                    "compiled_add_rms": True,
                    "lm_head_linear": True,
                }
            }
        payload = {
            "rid": f"warmed-{label}-{time.time_ns()}",
            "input_ids": (
                TEXT_INPUT_IDS if batch_size == 1 else [TEXT_INPUT_IDS] * batch_size
            ),
            "sampling_params": sampling_params,
        }
        response, wall_s = self._post("/generate", payload)
        rows = response if isinstance(response, list) else [response]
        expected = TEXT_EXPECTED_IDS[:decode_steps]
        output_ids = [[int(token) for token in row["output_ids"]] for row in rows]
        server_s = max(float(row["meta_info"]["e2e_latency"]) for row in rows)
        exact_stats = []
        for row in rows:
            exact_stats.extend(
                _non_null_values(row["meta_info"].get("sensenova_u1_exact_text_stats"))
            )
        return {
            "wall_s": wall_s,
            "server_s": server_s,
            "request_count": len(rows),
            "all_exact": all(tokens == expected for tokens in output_ids),
            "output_ids": output_ids,
            "exact_stats": exact_stats,
        }

    def t2i_request(self, label: str) -> dict[str, Any]:
        payload = {
            "prompt": T2I_PROMPT,
            "width": 256,
            "height": 256,
            "num_inference_steps": 2,
            "flow_shift": 1.0,
            "guidance_scale": 1.0,
            "n": 1,
            "seed": 20260813,
            "response_format": "b64_json",
            "output_format": "png",
        }
        response, wall_s = self._post("/v1/images/generations", payload)
        image_bytes = base64.b64decode(response["data"][0]["b64_json"])
        return {
            "label": label,
            "wall_s": wall_s,
            "model_s": float(response["inference_time_s"]),
            "png_sha256": _sha256(image_bytes),
            "_image_bytes": image_bytes,
        }

    @staticmethod
    def _interleave_text() -> str:
        return (
            f"<|im_start|>system\n{DEFAULT_SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{INTERLEAVE_PROMPT}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def interleave_request(self, label: str) -> dict[str, Any]:
        payload = {
            "rid": f"warmed-{label}-{time.time_ns()}",
            "text": self._interleave_text(),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 127,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "custom_params": {
                    "sensenova_u1_interleave": {
                        "width": 256,
                        "height": 256,
                        "num_steps": 2,
                        "max_images": 1,
                        "seed": 20260813,
                        "timestep_shift": 1.0,
                        "enable_timestep_shift": True,
                        "return_images": True,
                    }
                },
            },
        }
        response, wall_s = self._post("/generate", payload)
        output_ids = [int(token) for token in response["output_ids"]]
        try:
            image_start = output_ids.index(IMG_START_ID)
            image_end = output_ids.index(IMG_END_ID, image_start + 1)
        except ValueError as error:
            raise RuntimeError(
                "interleave output has no complete image span"
            ) from error
        image_context = output_ids[image_start + 1 : image_end]
        if len(image_context) != IMAGE_SPAN_TOKENS - 2 or any(
            token != IMG_CONTEXT_ID for token in image_context
        ):
            raise RuntimeError("interleave output has an invalid image token span")

        meta = response["meta_info"]
        image_values = _non_null_values(meta.get("sensenova_u1_interleave_image_b64"))
        shape_values = _non_null_values(meta.get("sensenova_u1_interleave_image_shape"))
        if len(image_values) != 1 or len(shape_values) != 1:
            raise RuntimeError("interleave output did not return exactly one image")
        image_bytes = base64.b64decode(image_values[0])
        image_shape = [int(value) for value in shape_values[0]]
        pre_ids = output_ids[:image_start]
        post_ids = output_ids[image_end + 1 :]
        contract = {
            "pre_image_token_ids": pre_ids,
            "post_image_token_ids": post_ids,
            "generated_text_tokens": len(pre_ids) + len(post_ids),
            "image_shape": image_shape,
            "image_sha256": _sha256(image_bytes),
            "text": response.get("text", ""),
        }
        return {
            "label": label,
            "wall_s": wall_s,
            "server_s": float(meta["e2e_latency"]),
            "flow_s": meta.get("sensenova_u1_flow_compute_seconds"),
            "cached_tokens": meta.get("cached_tokens"),
            "image_start_index": image_start,
            "image_end_index": image_end,
            "output_ids": output_ids,
            "contract": {
                **contract,
                "contract_sha256": _canonical_digest(contract),
            },
            "_image_bytes": image_bytes,
            "_image_shape": image_shape,
        }

    def _save_interleave_preview(
        self,
        image_bytes: bytes,
        image_shape: list[int],
    ) -> Path:
        image = (
            np.frombuffer(image_bytes, dtype=np.float16)
            .reshape(image_shape)
            .astype(np.float32)
        )
        rgb = np.clip(image[0] * 0.5 + 0.5, 0.0, 1.0)
        uint8 = np.rint(rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
        path = self.out_dir / "interleave.png"
        Image.fromarray(uint8, mode="RGB").save(path)
        return path

    @staticmethod
    def _public_row(row: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    def run(self) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        gpu_before = _gpu_state()
        health = self.health()

        warm_interleave = self.interleave_request("warmup-interleave")
        warm_text_bs1 = self.text_request(1, "warmup-text-bs1", exact_bs1=True)
        coexistence = {
            str(batch_size): self.text_request(
                batch_size,
                f"coexistence-bs{batch_size}",
                exact_bs1=False,
                decode_steps=8,
            )
            for batch_size in (1, 8, 16)
        }
        warm_t2i = self.t2i_request("warmup-t2i")

        text_results = {}
        for batch_size in (1, 8, 16):
            rows = [
                self.text_request(
                    batch_size,
                    f"measure-text-bs{batch_size}-{repeat}",
                    exact_bs1=batch_size == 1,
                )
                for repeat in range(self.measurement_repeats)
            ]
            server_values = [float(row["server_s"]) for row in rows]
            wall_values = [float(row["wall_s"]) for row in rows]
            server_tps = batch_size * 32 / statistics.median(server_values)
            wall_tps = batch_size * 32 / statistics.median(wall_values)
            speedup = server_tps / TEXT_HF_TOKENS_PER_S[batch_size]
            passed = bool(
                all(row["all_exact"] for row in rows)
                and server_tps >= TEXT_TARGET_TOKENS_PER_S[batch_size]
                and speedup >= TEXT_TARGET_SPEEDUP[batch_size]
            )
            text_results[str(batch_size)] = {
                "batch_size": batch_size,
                "hf_tokens_per_s": TEXT_HF_TOKENS_PER_S[batch_size],
                "target_tokens_per_s": TEXT_TARGET_TOKENS_PER_S[batch_size],
                "target_speedup": TEXT_TARGET_SPEEDUP[batch_size],
                "server_tokens_per_s": server_tps,
                "client_wall_tokens_per_s": wall_tps,
                "speedup": speedup,
                "server_latency_s": _stats(server_values),
                "client_wall_s": _stats(wall_values),
                "all_exact": all(row["all_exact"] for row in rows),
                "passed": passed,
                "rows": rows,
            }

        t2i_rows = [
            self.t2i_request(f"measure-t2i-{repeat}")
            for repeat in range(self.measurement_repeats)
        ]
        t2i_path = self.out_dir / "t2i.png"
        t2i_path.write_bytes(t2i_rows[-1]["_image_bytes"])
        t2i_model_s = _stats([float(row["model_s"]) for row in t2i_rows])
        t2i_wall_s = _stats([float(row["wall_s"]) for row in t2i_rows])
        t2i_speedup = T2I_HF_SECONDS / t2i_model_s["median"]
        t2i_metrics = _png_metrics(self.hf_t2i_image, t2i_path)
        t2i_deterministic = (
            len({str(row["png_sha256"]) for row in [warm_t2i, *t2i_rows]}) == 1
        )
        t2i_passed = bool(
            t2i_model_s["median"] <= T2I_TARGET_SECONDS
            and t2i_speedup >= T2I_TARGET_SPEEDUP
            and t2i_deterministic
            and t2i_metrics["psnr_db"] >= MIN_PSNR_DB
            and t2i_metrics["ssim_global_rgb"] >= MIN_SSIM
        )

        interleave_rows = [
            self.interleave_request(f"measure-interleave-{repeat}")
            for repeat in range(self.measurement_repeats)
        ]
        interleave_path = self._save_interleave_preview(
            interleave_rows[-1]["_image_bytes"],
            interleave_rows[-1]["_image_shape"],
        )
        interleave_server_s = _stats(
            [float(row["server_s"]) for row in interleave_rows]
        )
        interleave_wall_s = _stats([float(row["wall_s"]) for row in interleave_rows])
        interleave_speedup = INTERLEAVE_HF_SECONDS / interleave_server_s["median"]
        interleave_metrics = _png_metrics(
            self.hf_interleave_image,
            interleave_path,
        )
        interleave_all_rows = [warm_interleave, *interleave_rows]
        interleave_deterministic = (
            len(
                {str(row["contract"]["contract_sha256"]) for row in interleave_all_rows}
            )
            == 1
        )
        interleave_contract_ok = all(
            row["image_start_index"] == 125
            and len(row["contract"]["pre_image_token_ids"]) == 125
            and len(row["contract"]["post_image_token_ids"]) == 1
            and row["contract"]["generated_text_tokens"] == 126
            for row in interleave_all_rows
        )
        interleave_passed = bool(
            interleave_server_s["median"] <= INTERLEAVE_TARGET_SECONDS
            and interleave_wall_s["median"] <= INTERLEAVE_TARGET_SECONDS
            and interleave_speedup >= INTERLEAVE_TARGET_SPEEDUP
            and interleave_deterministic
            and interleave_contract_ok
            and interleave_metrics["psnr_db"] >= MIN_PSNR_DB
            and interleave_metrics["ssim_global_rgb"] >= MIN_SSIM
        )

        all_passed = bool(
            all(result["passed"] for result in text_results.values())
            and t2i_passed
            and interleave_passed
            and all(row["all_exact"] for row in coexistence.values())
        )
        return {
            "ok": all_passed,
            "decision": (
                "warmed_acceptance_passed" if all_passed else "warmed_acceptance_failed"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "repository": "sgl-project/sglang",
            "base_url": self.base_url,
            "git": _git_state(),
            "health": health,
            "gpu_before": gpu_before,
            "settings": {
                "measurement_repeats": self.measurement_repeats,
                "text_prompt_tokens": len(TEXT_INPUT_IDS),
                "text_decode_steps": 32,
                "image_geometry": [256, 256],
                "image_steps": 2,
                "seed": 20260813,
                "context_length": 4096,
                "workload_weakened": False,
            },
            "warmups": {
                "interleave": self._public_row(warm_interleave),
                "text_bs1": warm_text_bs1,
                "t2i": self._public_row(warm_t2i),
            },
            "graph_and_radix_coexistence": coexistence,
            "text": text_results,
            "t2i": {
                "hf_seconds": T2I_HF_SECONDS,
                "target_seconds": T2I_TARGET_SECONDS,
                "target_speedup": T2I_TARGET_SPEEDUP,
                "model_s": t2i_model_s,
                "client_wall_s": t2i_wall_s,
                "speedup": t2i_speedup,
                "deterministic": t2i_deterministic,
                "image_metrics": t2i_metrics,
                "passed": t2i_passed,
                "rows": [self._public_row(row) for row in t2i_rows],
            },
            "interleave": {
                "hf_seconds": INTERLEAVE_HF_SECONDS,
                "target_seconds": INTERLEAVE_TARGET_SECONDS,
                "target_speedup": INTERLEAVE_TARGET_SPEEDUP,
                "server_s": interleave_server_s,
                "client_wall_s": interleave_wall_s,
                "speedup": interleave_speedup,
                "deterministic": interleave_deterministic,
                "contract_ok": interleave_contract_ok,
                "image_metrics": interleave_metrics,
                "passed": interleave_passed,
                "rows": [self._public_row(row) for row in interleave_rows],
            },
            "artifacts": {
                "t2i_png": str(t2i_path),
                "interleave_png": str(interleave_path),
                "hf_t2i_png": str(self.hf_t2i_image),
                "hf_interleave_png": str(self.hf_interleave_image),
            },
            "gpu_after": _gpu_state(),
        }


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    text_rows = []
    for batch_size in ("1", "8", "16"):
        row = result["text"][batch_size]
        text_rows.append(
            "| {bs} | {tps:.4f} | {speedup:.4f}x | {exact} | {passed} |".format(
                bs=batch_size,
                tps=row["server_tokens_per_s"],
                speedup=row["speedup"],
                exact="PASS" if row["all_exact"] else "FAIL",
                passed="PASS" if row["passed"] else "FAIL",
            )
        )
    lines = [
        "# SenseNova U1 warmed acceptance",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Timestamp UTC: `{result['timestamp_utc']}`",
        (
            "- Workload, correctness, parity, determinism, context, and safety "
            "settings were unchanged."
        ),
        "",
        "## Text-only",
        "",
        "| BS | Native server tok/s | Speedup | Token exact | Gate |",
        "|---:|---:|---:|---:|---:|",
        *text_rows,
        "",
        "## Image and interleave",
        "",
        "| Gate | Native median | Speedup | Deterministic | Parity | Result |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| T2I 256x256, 2 steps | "
            f"{result['t2i']['model_s']['median']:.6f}s | "
            f"{result['t2i']['speedup']:.4f}x | "
            f"{'PASS' if result['t2i']['deterministic'] else 'FAIL'} | "
            f"{result['t2i']['image_metrics']['psnr_db']:.4f} dB / "
            f"{result['t2i']['image_metrics']['ssim_global_rgb']:.6f} | "
            f"{'PASS' if result['t2i']['passed'] else 'FAIL'} |"
        ),
        (
            "| Fixed-work first image | "
            f"{result['interleave']['server_s']['median']:.6f}s server / "
            f"{result['interleave']['client_wall_s']['median']:.6f}s client | "
            f"{result['interleave']['speedup']:.4f}x | "
            f"{'PASS' if result['interleave']['deterministic'] else 'FAIL'} | "
            f"{result['interleave']['image_metrics']['psnr_db']:.4f} dB / "
            f"{result['interleave']['image_metrics']['ssim_global_rgb']:.6f} | "
            f"{'PASS' if result['interleave']['passed'] else 'FAIL'} |"
        ),
        "",
        "Raw evidence: `results.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--measurement-repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument(
        "--hf-t2i-image",
        type=Path,
        default=Path(DEFAULT_HF_T2I_IMAGE),
    )
    parser.add_argument(
        "--hf-interleave-image",
        type=Path,
        default=Path(DEFAULT_HF_INTERLEAVE_IMAGE),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = AcceptanceRunner(
        base_url=args.base_url,
        out_dir=args.out_dir,
        measurement_repeats=args.measurement_repeats,
        timeout_s=args.timeout_s,
        hf_t2i_image=args.hf_t2i_image,
        hf_interleave_image=args.hf_interleave_image,
    )
    try:
        result = runner.run()
    except Exception as error:
        failure = {
            "ok": False,
            "decision": "warmed_acceptance_crashed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "git": _git_state(),
            "gpu": _gpu_state(),
        }
        _write_json(args.out_dir / "results.json", failure)
        raise
    _write_json(args.out_dir / "results.json", result)
    _write_summary(args.out_dir / "summary.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
