"""
End-to-end test for GPU memory sleep/wake and in-place weight update workflow
in a running SGLang multimodal server.

Author:

Kun Lin, https://github.com/klhhhhh
Chenyang Zhao, https://github.com/zhaochenyang20
Menyang Liu, https://github.com/dreamyang-liu
shuwen, https://github.com/alphabetc1
Mook, https://github.com/Godmook
Yuzhen Zhou, https://github.com/zyzshishui
Ratish P, https://github.com/Ratish1

This test validates both functional correctness of /release_memory_occupation
and /resume_memory_occupation:

1. Launch a SGLang server process without offloading DiT and text encoder.
   This roughly takes 56GB on H200.

2. Trigger GPU memory release via the `/release_memory_occupation` endpoint
   and verify that GPU memory usage decreases.

   TODO (chenyang): still found some memory usage that can not be released:

   https://github.com/sgl-project/sglang/issues/19441

3. Verify idempotency of release: repeated calls to /release_memory_occupation
   should return 200 and keep the sleeping state.

4. While sleeping, verify that /v1/images/generations rejects requests with
   HTTP 400 and the response mentions the sleep state.

5. Trigger GPU memory resume via the `/resume_memory_occupation` endpoint
   and verify that GPU memory usage increases accordingly.

6. Verify idempotency of resume: repeated calls to /resume_memory_occupation
   should return 200 and keep the awake state.

7. Perform an in-place model weight update using the
   `/update_weights_from_disk` endpoint without restarting the server.

    This is inherent with the usage in RL. SGLang Diffusion Server is slept
    while training and resumed to do the next rollout with new weights.

    However, in our test, we only refit the model weights with the same model,
    just to check the correctness of the refit API and save time.

Related issues:

https://github.com/sgl-project/sglang/issues/19442
https://github.com/sgl-project/sglang/issues/19352
"""

import os

import pytest

from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.utils import (
    launch_server_cmd,
    post_request,
    query_gpu_mem_used_mib,
    terminate_process,
    wait_for_http_ready,
)

logger = init_logger(__name__)

_MODEL_ID = "Qwen/Qwen-Image"


def _assert_mem_changed(
    label: str,
    before: int,
    after: int,
    min_delta_mib: int,
    expect_decrease: bool,
) -> None:
    delta = before - after if expect_decrease else after - before
    direction = "decrease" if expect_decrease else "increase"
    lhs = "before-after" if expect_decrease else "after-before"
    logger.info(
        f"[MEM] {label}: before={before} MiB after={after} MiB delta={delta} MiB (expect {direction})"
    )
    assert delta >= min_delta_mib, (
        f"GPU memory did not {direction} enough for '{label}': "
        f"{lhs}={delta} MiB < {min_delta_mib} MiB "
        f"(before={before} MiB, after={after} MiB)"
    )


def _assert_memory_occupation_ok(
    base_url: str,
    endpoint: str,
    timeout_s: float,
    expect_sleeping: bool,
    label: str,
) -> dict:
    response = post_request(
        base_url,
        endpoint,
        payload={},
        timeout_s=timeout_s,
        logger=logger,
    )
    assert (
        response.status_code == 200
    ), f"{label} should return 200: {response.status_code} {response.text}"
    out = response.json()
    assert out.get("success", False) is True, f"{label} response: {out}"
    assert out.get("sleeping") is expect_sleeping, f"{label} response: {out}"
    return out


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_sleep_wake_refit_generate_e2e():
    payload = {
        "prompt": "a cute panda",
        "width": 256,
        "height": 256,
        "num_inference_steps": 2,
        "response_format": "b64_json",
    }
    process = None
    try:
        cmd = (
            "sglang serve "
            f"--model-path {_MODEL_ID} "
            "--num-gpus 1 "
            "--dit-cpu-offload false "
            "--text-encoder-cpu-offload false"
        )
        process, port = launch_server_cmd(cmd, host="127.0.0.1")
        base_url = f"http://127.0.0.1:{port}"
        visible = (
            os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            or os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
        )
        first_visible = visible.split(",")[0].strip() if visible else ""
        gpu_index = int(first_visible) if first_visible.isdigit() else 0
        logger.info(f"Test start: model={_MODEL_ID} port={port} base_url={base_url}")
        logger.info(
            f"[GPU] memory sampling target: gpu_index={gpu_index} (from CUDA_VISIBLE_DEVICES)"
        )

        wait_for_http_ready(f"{base_url}/health", timeout=900, process=process)
        mem_before_sleep = query_gpu_mem_used_mib(gpu_index)
        first_sleep_out = _assert_memory_occupation_ok(
            base_url=base_url,
            endpoint="/release_memory_occupation",
            timeout_s=180.0,
            expect_sleeping=True,
            label="sleep",
        )

        mem_after_sleep = query_gpu_mem_used_mib(gpu_index)
        min_sleep_delta = int(os.environ.get("SGLD_MMGEN_SLEEP_MEM_DELTA_MIB", "1024"))
        _assert_mem_changed(
            "sleep (baseline -> after sleep)",
            mem_before_sleep,
            mem_after_sleep,
            min_sleep_delta,
            expect_decrease=True,
        )

        # Idempotency: repeated release should still return 200 and keep state.
        second_sleep_out = _assert_memory_occupation_ok(
            base_url=base_url,
            endpoint="/release_memory_occupation",
            timeout_s=180.0,
            expect_sleeping=True,
            label="second sleep",
        )
        assert (
            second_sleep_out["sleeping"] == first_sleep_out["sleeping"]
        ), f"sleep should be idempotent: first={first_sleep_out}, second={second_sleep_out}"

        # While sleeping, generation should be rejected with HTTP 400.
        response = post_request(
            base_url,
            "/v1/images/generations",
            payload,
            timeout_s=180.0,
            logger=logger,
        )
        assert response.status_code == 400, (
            "sleeping generate should fail with 400: "
            f"{response.status_code} {response.text}"
        )
        assert "sleep" in response.text.lower(), (
            "sleeping generate response should mention sleep state: "
            f"{response.status_code} {response.text}"
        )

        first_wake_out = _assert_memory_occupation_ok(
            base_url=base_url,
            endpoint="/resume_memory_occupation",
            timeout_s=300.0,
            expect_sleeping=False,
            label="wake",
        )

        mem_after_wake = query_gpu_mem_used_mib(gpu_index)
        min_wake_delta = int(os.environ.get("SGLD_MMGEN_WAKE_MEM_DELTA_MIB", "1024"))
        _assert_mem_changed(
            "wake (after sleep -> after wake)",
            mem_after_sleep,
            mem_after_wake,
            min_wake_delta,
            expect_decrease=False,
        )

        # Idempotency: repeated resume should still return 200 and keep state.
        second_wake_out = _assert_memory_occupation_ok(
            base_url=base_url,
            endpoint="/resume_memory_occupation",
            timeout_s=300.0,
            expect_sleeping=False,
            label="second wake",
        )
        assert (
            second_wake_out["sleeping"] == first_wake_out["sleeping"]
        ), f"wake should be idempotent: first={first_wake_out}, second={second_wake_out}"

        model_snapshot_path = maybe_download_model(_MODEL_ID)
        r = post_request(
            base_url,
            "/update_weights_from_disk",
            payload={"model_path": model_snapshot_path, "flush_cache": True},
            timeout_s=900.0,
            logger=logger,
        )
        assert (
            r.status_code == 200
        ), f"update_weights_from_disk failed: {r.status_code} {r.text}"
        out = r.json()
        assert out.get("success") is True, f"update_weights_from_disk response: {out}"

        response = post_request(
            base_url, "/v1/images/generations", payload, timeout_s=900.0, logger=logger
        )
        assert (
            response.status_code == 200
        ), f"generate failed: {response.status_code} {response.text}"
    finally:
        if process is not None:
            terminate_process(process)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
