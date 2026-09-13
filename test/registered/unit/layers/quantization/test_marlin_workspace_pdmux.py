"""Regression tests for Marlin per-stream workspaces under PD-Multiplexing.

The Marlin kernels use `workspace` as a cross-CTA lock array and assume every
launch touching one workspace is serialized on a single CUDA stream. Under
PD-Multiplexing the decode forward and the split-prefill slice of the same
layer run concurrently on two green-context streams; a shared per-layer
workspace then lets both launches consume the same barrier slots and a CTA
spins forever (`barrier_acquire` waits for exact counter equality).

Two properties are pinned here:

1. Host-side: workspace selection is keyed by the executing stream, falls
   back to the per-layer workspace when the layer is not opted in, opt-in
   happens only under PDMux, and the per-call `max_blocks_per_sm` sizing is
   preserved.
2. Device-side: the same layer driven concurrently on two streams with deep
   queues (24 launches per stream before any synchronization — shallow
   one-out/one-in submission never overlaps two launches on the device and
   cannot reproduce the race) completes and matches the solo reference.

The device-side case runs in a child process with a hard timeout so a
regression manifests as a clean failure instead of a hung CI runner.
"""

import hashlib
import json
import multiprocessing as mp
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    get_marlin_workspace_for_forward,
    marlin_init_stream_workspaces,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

DEEP_QUEUE_DEPTH = 24
CHILD_TIMEOUT_S = 180


def _make_fp8_marlin_layer(k: int, n: int, dev: torch.device):
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        prepare_fp8_layer_for_marlin,
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        (torch.randn(k, n, device=dev, dtype=torch.float32) / 10).to(
            torch.float8_e4m3fn
        ),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.rand(1, n, device=dev, dtype=torch.float32) * 0.05 + 0.01,
        requires_grad=False,
    )
    layer.orig_dtype = torch.bfloat16
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.weight_block_size = None
    prepare_fp8_layer_for_marlin(layer, size_k_first=True)
    layer.workspace = marlin_make_workspace(dev)
    return layer


def _run_gemm(layer, x):
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
    )

    return apply_fp8_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=get_marlin_workspace_for_forward(layer),
        size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition,
        bias=None,
    )


def _hash_out(t: torch.Tensor) -> str:
    return hashlib.sha256(t.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _child_deep_queue(result_path: str) -> None:
    """Interleave two streams on one layer with deep queues, then verify."""
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.manual_seed(0)

    layer = _make_fp8_marlin_layer(512, 128, dev)
    x = torch.randn(8, 512, device=dev, dtype=torch.bfloat16)

    ref = _run_gemm(layer, x)
    torch.cuda.synchronize()
    ref_sha = _hash_out(ref)

    # Opt in exactly the state prepare_fp8_layer_for_marlin creates on a
    # PDMux server (marlin_init_stream_workspaces under enabled stream
    # groups). The test machine may not run PDMux, so set the state
    # directly: this is the contract get_marlin_workspace_for_forward
    # must honor.
    layer.marlin_stream_workspaces = {}

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    for _ in range(4):
        outs = []
        for _ in range(DEEP_QUEUE_DEPTH):
            with torch.cuda.stream(stream_a):
                outs.append(_run_gemm(layer, x))
        for _ in range(DEEP_QUEUE_DEPTH):
            with torch.cuda.stream(stream_b):
                outs.append(_run_gemm(layer, x))
        stream_a.synchronize()
        stream_b.synchronize()
        for idx, out in enumerate(outs):
            if _hash_out(out) != ref_sha:
                raise AssertionError(
                    f"wrong result at launch {idx}: {_hash_out(out)} != {ref_sha}"
                )

    # Both streams must have received private workspaces.
    assert len(layer.marlin_stream_workspaces) == 2, (
        f"expected per-stream workspaces, got {len(layer.marlin_stream_workspaces)}"
    )
    with open(result_path, "w") as f:
        json.dump({"ok": True}, f)


class TestMarlinWorkspaceSelection(CustomTestCase):
    def test_falls_back_to_layer_workspace_without_opt_in(self):
        layer = torch.nn.Module()
        layer.workspace = torch.zeros(1, dtype=torch.int32)
        self.assertIs(get_marlin_workspace_for_forward(layer), layer.workspace)

    @patch(
        "sglang.srt.layers.quantization.marlin_utils.marlin_stream_workspaces_enabled",
        return_value=False,
    )
    def test_init_is_noop_without_pdmux(self, _mock_enabled):
        layer = torch.nn.Module()
        marlin_init_stream_workspaces(layer, max_blocks_per_sm=4)
        self.assertFalse(hasattr(layer, "marlin_stream_workspaces"))

    @patch(
        "sglang.srt.layers.quantization.marlin_utils.marlin_stream_workspaces_enabled",
        return_value=True,
    )
    def test_init_opts_in_and_records_make_kwargs(self, _mock_enabled):
        layer = torch.nn.Module()
        marlin_init_stream_workspaces(layer, max_blocks_per_sm=4)
        self.assertEqual(layer.marlin_stream_workspaces, {})
        self.assertEqual(layer.marlin_workspace_make_kwargs, {"max_blocks_per_sm": 4})

    def test_stream_keyed_selection_and_sizing(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count

        layer = _make_fp8_marlin_layer(512, 128, dev)
        layer.marlin_stream_workspaces = {}
        layer.marlin_workspace_make_kwargs = {"max_blocks_per_sm": 4}

        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        with torch.cuda.stream(stream_a):
            ws_a = get_marlin_workspace_for_forward(layer)
        with torch.cuda.stream(stream_b):
            ws_b = get_marlin_workspace_for_forward(layer)
        self.assertIsNot(ws_a, ws_b)
        self.assertEqual(ws_a.numel(), sms * 4)
        self.assertEqual(ws_b.numel(), sms * 4)
        with torch.cuda.stream(stream_a):
            self.assertIs(get_marlin_workspace_for_forward(layer), ws_a)


class TestMarlinDeepQueueConcurrency(CustomTestCase):
    def test_two_streams_deep_queue_complete_and_match(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        ctx = mp.get_context("spawn")
        result_path = "/tmp/marlin_ws_deep_queue_result.json"
        child = ctx.Process(target=_child_deep_queue, args=(result_path,))
        child.start()
        child.join(timeout=CHILD_TIMEOUT_S)
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
            self.fail(
                "deep-queue marlin concurrency did not complete within "
                f"{CHILD_TIMEOUT_S}s: workspace serialization regression?"
            )
        self.assertEqual(child.exitcode, 0)
        with open(result_path) as f:
            self.assertTrue(json.load(f)["ok"])


if __name__ == "__main__":
    unittest.main()
