import json
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.managers.mm_utils import _scatter_mm_embedding, embed_mm_inputs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

NUM_TOKENS = 64


def _make_mask(pattern: str) -> torch.Tensor:
    mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
    if pattern == "interleaved":
        mask[::3] = True
    elif pattern == "blocks":
        mask[5:20] = True
        mask[40:41] = True
    elif pattern == "all_true":
        mask[:] = True
    return mask.unsqueeze(-1)


@pytest.mark.parametrize("width", [8, 24])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mask_pattern", ["interleaved", "blocks", "all_true", "all_false"]
)
def test_scatter_matches_masked_scatter_bitwise(width, src_dtype, mask_pattern):
    """The row-index mm embedding merge must stay bitwise identical to
    masked_scatter_ semantics, whose internal transients it avoids."""
    torch.manual_seed(0)
    mask = _make_mask(mask_pattern)
    dest = torch.randn(NUM_TOKENS, width).to(torch.bfloat16)
    src = torch.randn(int(mask.sum()), width, dtype=src_dtype)

    expected = dest.clone()
    expected.masked_scatter_(mask.expand_as(expected), src.to(expected.dtype))

    actual = dest.clone()
    _scatter_mm_embedding(dest=actual, mask=mask, src=src)
    assert torch.equal(actual, expected)


def test_scatter_row_count_mismatch_fails_loud():
    """A mask/src row-count mismatch must raise, not silently corrupt rows."""
    dest = torch.zeros(8, 4)
    src_short_mask = _make_mask("all_false")[:8]
    src_short_mask[1] = True
    with pytest.raises((RuntimeError, IndexError)):
        _scatter_mm_embedding(dest=dest, mask=src_short_mask, src=torch.ones(3, 4))
    mask_heavy = src_short_mask.clone()
    mask_heavy[2:6] = True
    with pytest.raises((RuntimeError, IndexError)):
        _scatter_mm_embedding(dest=dest, mask=mask_heavy, src=torch.ones(1, 4))


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_placeholder_copy_device_routing(device_type):
    device = torch.device("cuda:1") if device_type == "cuda" else Mock(type="npu")
    # torch_npu's transfer_to_npu also makes NPU tensors report is_cuda=True.
    input_ids = Mock(device=device, is_cuda=True)
    pad_value = (1 << 40) + 7
    item = MultimodalDataItem(
        modality=Modality.IMAGE, pad_value=pad_value, offsets=[(0, 0)]
    )
    placeholder_cpu = Mock()
    with (
        patch(
            "sglang.srt.managers.mm_utils.torch.tensor",
            return_value=placeholder_cpu,
        ) as make_tensor,
        patch("sglang.srt.managers.mm_utils.torch.as_tensor") as as_tensor,
        patch(
            "sglang.srt.managers.mm_utils.get_embedding_and_mask",
            return_value=(None, None, input_ids),
        ) as get_embedding,
    ):
        embed_mm_inputs(
            mm_inputs_list=[MultimodalInputs(mm_items=[item])],
            extend_prefix_lens=[0],
            extend_seq_lens=[1],
            input_ids=input_ids,
            input_embedding=Mock(num_embeddings=8),
            data_embedding_func_mapping={Modality.IMAGE: Mock()},
        )

    if device_type == "cuda":
        make_tensor.assert_called_once_with(
            [pad_value], dtype=torch.int64, device="cpu", pin_memory=True
        )
        placeholder_cpu.to.assert_called_once_with(device, non_blocking=True)
        as_tensor.assert_not_called()
        expected = placeholder_cpu.to.return_value
    else:
        make_tensor.assert_not_called()
        placeholder_cpu.to.assert_not_called()
        as_tensor.assert_called_once_with([pad_value], device=device)
        expected = as_tensor.return_value
    assert get_embedding.call_args.kwargs["placeholder_tensor"] is expected


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_embed_mm_inputs_preserves_large_placeholder_ids(device, tmp_path):
    stream = torch.cuda.Stream() if device == "cuda" else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        pad_values = [(1 << 40) + 7, (1 << 40) + 9]
        input_ids = torch.tensor([0, pad_values[0], 1, pad_values[1], 2], device=device)
        input_embedding = torch.nn.Embedding.from_pretrained(
            torch.arange(24, dtype=torch.float32, device=device).reshape(6, 4)
        )
        features = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
        items = [
            MultimodalDataItem(
                modality=modality,
                pad_value=pad_values[index],
                offsets=[(2 * index + 1, 2 * index + 1)],
                precomputed_embeddings=features[index : index + 1],
            )
            for index, modality in enumerate((Modality.IMAGE, Modality.AUDIO))
        ]
        expected = input_embedding(torch.tensor([0, 0, 1, 0, 2], device=device))
        expected[1] = features[0]
        expected[3] = features[1]

        profiler = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=False,
            )
            if stream is not None
            else nullcontext()
        )
        with profiler as capture:
            with torch.profiler.record_function("mm.placeholder.embedding"):
                result, other_info = embed_mm_inputs(
                    mm_inputs_list=[MultimodalInputs(mm_items=items)],
                    extend_prefix_lens=[0],
                    extend_seq_lens=[5],
                    input_ids=input_ids,
                    input_embedding=input_embedding,
                    data_embedding_func_mapping={
                        Modality.IMAGE: Mock(),
                        Modality.AUDIO: Mock(),
                    },
                )
    if stream is not None:
        stream.synchronize()

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert other_info == {}
    if capture is not None:
        trace_path = tmp_path / "mm-embedding-no-stack.trace.json"
        capture.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text())["traceEvents"]
        scope = next(e for e in events if e.get("name") == "mm.placeholder.embedding")
        # Exclude profiler teardown, which synchronizes after the measured scope.
        runtime = [
            e
            for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver")
            and e.get("tid") == scope["tid"]
            and scope["ts"] <= e["ts"] < scope["ts"] + scope["dur"]
        ]
        assert runtime, "The CUDA profiler did not capture runtime events"
        syncs = [e["name"] for e in runtime if "Synchronize" in e["name"]]
        assert not syncs, f"Embedding preparation blocked the host: {syncs}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
