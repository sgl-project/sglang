"""Throwaway probe (not for merge): where does FLUX.3 edit go nondeterministic on ROCm?

Round 1 showed monolithic T2I is bit-exact on ROCm while the same edit request
gives different pixels each time. This round runs one monolithic server (GPU 1,
as test_disagg_server.TestDisaggFlux3Image does) with SGLANG_FLUX3_HASH_PROBE=1,
which makes the FLUX.3 stages log a content hash at every stage boundary,
every denoise step and every VAE-encoder layer. Three identical edits are then
compared tag by tag, and the first tag whose hash differs is the stage that
introduces the nondeterminism.
"""

import base64
import json
import os
import re
from io import BytesIO

import requests
from PIL import Image

from sglang.multimodal_gen.test.single_test_file import test_disagg_server as tds
from sglang.test.test_utils import CustomTestCase

MODEL = tds.TestDisaggFlux3Image.model
KW = dict(tds.TestDisaggFlux3Image.warmup_kwargs)
EDITS = 3
_LINE = re.compile(
    r"\[flux3-hash\] (\S+) shape=(\S+) dtype=(\S+) sha1=(\w+) absmean=(\S+)"
)


def _edit(port: int, reference: bytes) -> bytes:
    response = requests.post(
        f"http://{tds.HOST}:{port}/v1/images/edits",
        files={"image": ("reference.png", reference, "image/png")},
        data=dict(
            model=MODEL,
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


def _hash_lines(log_path) -> list[tuple]:
    with open(log_path, errors="replace") as fh:
        return [m.groups() for m in map(_LINE.search, fh) if m]


class ProbeFlux3EditStages(CustomTestCase):
    def test_edit_stage_hashes(self):
        os.environ["SGLANG_FLUX3_HASH_PROBE"] = "1"
        cluster = tds.DisaggCluster(
            model=MODEL,
            name="flux3_hash_probe",
            gpu_layout={},
            extra_role_args={},
            warmup_kwargs={},
        )
        port = tds.find_free_port(tds.HOST)
        log = tds._LOG_DIR / "flux3_hash_probe_mono.log"
        cluster._start_proc(
            [
                "sglang",
                "serve",
                "--model-path",
                MODEL,
                "--port",
                str(port),
                "--num-gpus",
                "1",
                "--base-gpu-id",
                "1",
                "--warmup-mode",
                "off",
            ],
            log,
        )
        try:
            tds.wait_for_server_health(
                f"http://{tds.HOST}:{port}",
                path="/v1/models",
                timeout=cluster.startup_timeout,
            )
            marks = [len(_hash_lines(log))]
            t2i = tds._generate_image(port, MODEL, **KW)
            marks.append(len(_hash_lines(log)))
            ref = BytesIO()
            Image.open(BytesIO(t2i)).crop((0, 0, 247, 239)).save(ref, format="PNG")
            for _ in range(EDITS):
                _edit(port, ref.getvalue())
                marks.append(len(_hash_lines(log)))
        finally:
            cluster.stop()

        lines = _hash_lines(log)
        runs = [lines[marks[i] : marks[i + 1]] for i in range(1, len(marks) - 1)]
        print(
            f"[flux3-probe] edit runs: {len(runs)}, hashes per run: "
            f"{[len(r) for r in runs]}",
            flush=True,
        )
        base = runs[0]
        for k, run in enumerate(runs[1:], start=2):
            first = None
            for (tag, shape, dt, h0, m0), (tag1, _, _, h1, m1) in zip(base, run):
                assert tag == tag1, (tag, tag1)
                if h0 != h1 and first is None:
                    first = dict(tag=tag, shape=shape, dtype=dt, absmean=[m0, m1])
            print(
                "[flux3-probe] "
                + json.dumps(
                    {"compare": f"edit 1 vs edit {k}", "first_differing": first}
                ),
                flush=True,
            )
        # The whole table for edit 1 and 2, so the divergence point can be read in order.
        for (tag, shape, dt, h0, m0), (_, _, _, h1, m1) in zip(base, runs[1]):
            print(
                f"[flux3-probe] {'SAME' if h0 == h1 else 'DIFF'} {tag} {shape} {dt} "
                f"{h0} {h1} absmean {m0} {m1}",
                flush=True,
            )
