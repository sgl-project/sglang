"""Throwaway probe (not for merge): is FLUX.3 T2I / edit deterministic on ROCm?

Same model, request parameters and GPU layout as
test_disagg_server.TestDisaggFlux3Image. Prints a pixel diff for every pair
instead of asserting, so one run separates "disagg differs from monolithic"
from "ROCm gives different pixels run to run".
"""

import base64
import hashlib
import json
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from sglang.multimodal_gen.test.single_test_file import test_disagg_server as tds
from sglang.test.test_utils import terminate_and_kill_process_tree

T = tds.TestDisaggFlux3Image
KW = T.warmup_kwargs


def _pixels(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(png))).astype(np.int16)


def _report(label: str, x: bytes, y: bytes) -> None:
    a, b = _pixels(x), _pixels(y)
    row = {"pair": label, "shape": list(a.shape)}
    if a.shape != b.shape:
        row["shape_b"] = list(b.shape)
    else:
        d = np.abs(a - b)
        row.update(
            identical=bool((d == 0).all()),
            mismatched=int((d > 0).sum()),
            of=int(d.size),
            max_abs=int(d.max()),
            mean_abs=round(float(d.mean()), 4),
        )
    row["sha1"] = [hashlib.sha1(x).hexdigest()[:10], hashlib.sha1(y).hexdigest()[:10]]
    print("[flux3-probe] " + json.dumps(row), flush=True)


def _edit(port: int, reference: bytes) -> bytes:
    response = requests.post(
        f"http://{tds.HOST}:{port}/v1/images/edits",
        files={"image": ("reference.png", reference, "image/png")},
        data=dict(
            model=T.model,
            prompt="Turn the scene into a watercolor painting",
            seed=42,
            num_inference_steps=2,
            quality="exact",
            response_format="b64_json",
        ),
        timeout=600,
    )
    response.raise_for_status()
    return base64.b64decode(response.json()["data"][0]["b64_json"])


class ProbeFlux3Determinism(tds._DisaggTestBase):
    model = T.model
    cluster_name = "flux3_probe"
    gpu_layout = T.gpu_layout
    warmup_kwargs = T.warmup_kwargs
    extra_role_args = T.extra_role_args

    def _monolithic(self, gpu: int, tag: str):
        port = tds.find_free_port(tds.HOST)
        log = tds._LOG_DIR / f"flux3_probe_mono_{tag}.log"
        self.cluster._logs[f"mono_{tag}"] = log
        proc = self.cluster._start_proc(
            [
                "sglang",
                "serve",
                "--model-path",
                self.model,
                "--port",
                str(port),
                "--num-gpus",
                "1",
                "--base-gpu-id",
                str(gpu),
                "--warmup-mode",
                "off",
            ],
            log,
        )
        tds.wait_for_server_health(
            f"http://{tds.HOST}:{port}",
            path="/v1/models",
            timeout=self.cluster.startup_timeout,
        )
        return port, proc

    def test_probe(self):
        c = self.cluster
        out = {}
        # 1. Disaggregated: T2I and edit, each twice in a row.
        out["d_t2i_a"] = tds._generate_image(c.api_port, self.model, **KW)
        out["d_t2i_b"] = tds._generate_image(c.api_port, self.model, **KW)
        ref = BytesIO()
        Image.open(BytesIO(out["d_t2i_a"])).crop((0, 0, 247, 239)).save(
            ref, format="PNG"
        )
        ref = ref.getvalue()
        out["d_edit_a"] = _edit(c.api_port, ref)
        out["d_edit_b"] = _edit(c.api_port, ref)
        c.stop()

        # 2. Monolithic on GPU 1 (as the test does): each twice in a row.
        port, proc = self._monolithic(1, "gpu1_first")
        out["m1_t2i_a"] = tds._generate_image(port, self.model, **KW)
        out["m1_t2i_b"] = tds._generate_image(port, self.model, **KW)
        out["m1_edit_a"] = _edit(port, ref)
        out["m1_edit_b"] = _edit(port, ref)
        terminate_and_kill_process_tree(proc)

        # 3. Monolithic restarted on GPU 1, and on GPU 0.
        port, proc = self._monolithic(1, "gpu1_restart")
        out["m2_t2i"] = tds._generate_image(port, self.model, **KW)
        out["m2_edit"] = _edit(port, ref)
        terminate_and_kill_process_tree(proc)
        port, proc = self._monolithic(0, "gpu0")
        out["m0_t2i"] = tds._generate_image(port, self.model, **KW)
        out["m0_edit"] = _edit(port, ref)
        terminate_and_kill_process_tree(proc)

        for label, x, y in [
            ("disagg t2i: run a vs b", "d_t2i_a", "d_t2i_b"),
            ("disagg edit: run a vs b", "d_edit_a", "d_edit_b"),
            ("mono t2i: run a vs b (same server)", "m1_t2i_a", "m1_t2i_b"),
            ("mono edit: run a vs b (same server)", "m1_edit_a", "m1_edit_b"),
            ("mono t2i: server restart", "m1_t2i_a", "m2_t2i"),
            ("mono edit: server restart", "m1_edit_a", "m2_edit"),
            ("mono t2i: gpu1 vs gpu0", "m1_t2i_a", "m0_t2i"),
            ("mono edit: gpu1 vs gpu0", "m1_edit_a", "m0_edit"),
            ("t2i: disagg vs mono", "d_t2i_a", "m1_t2i_a"),
            ("edit: disagg vs mono", "d_edit_a", "m1_edit_a"),
        ]:
            _report(label, out[x], out[y])
