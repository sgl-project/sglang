import torch

import sglang.srt.layers.attention.deepseek_v4_backend_radix as dsv4_radix
import sglang.srt.layers.deep_gemm_wrapper.paged_mqa_logits as paged_mqa
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def test_dsv4_flashmla_prefill_chunking_slices_row_tensors(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_PREFILL_METADATA_CHUNK_SIZE", "4")
    created_metadata = []

    def fake_metadata():
        metadata = object()
        created_metadata.append(metadata)
        return metadata

    monkeypatch.setattr(dsv4_radix, "_create_flashmla_metadata", fake_metadata)

    calls = []

    def fake_runner(**kwargs):
        calls.append(
            {
                "q": kwargs["q"].clone(),
                "indices": kwargs["indices"].clone(),
                "topk_length": kwargs["topk_length"].clone(),
                "k_cache_shape": kwargs["k_cache"].shape,
                "backend": kwargs["backend"],
                "tile_scheduler_metadata": kwargs["tile_scheduler_metadata"],
            }
        )
        return (kwargs["q"] + 1,)

    input_dict = {
        "q": torch.arange(10, dtype=torch.float32).view(10, 1, 1, 1),
        "k_cache": torch.zeros(3, 128, 1, 1),
        "indices": torch.arange(10 * 64, dtype=torch.int32).view(10, 1, 64),
        "topk_length": torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=torch.int32),
        "extra_indices_in_kvcache": None,
        "extra_topk_length": None,
    }

    out = dsv4_radix._flash_mla_with_optional_prefill_chunking(
        input_dict=input_dict,
        backend="kernel",
        forward_mode=ForwardMode.EXTEND,
        runner=fake_runner,
    )

    assert out.shape == input_dict["q"].shape
    assert torch.equal(out, input_dict["q"] + 1)
    assert [call["q"].shape[0] for call in calls] == [4, 4, 2]
    assert [call["indices"].shape[0] for call in calls] == [4, 4, 2]
    assert [call["topk_length"].shape[0] for call in calls] == [4, 4, 2]
    assert all(call["k_cache_shape"] == input_dict["k_cache"].shape for call in calls)
    assert all(call["backend"] == "kernel" for call in calls)
    assert [call["tile_scheduler_metadata"] for call in calls] == created_metadata
    assert len(set(map(id, created_metadata))) == 3


def test_dsv4_flashmla_chunking_is_prefill_only(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_PREFILL_METADATA_CHUNK_SIZE", "4")

    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs["q"].shape[0])
        return (kwargs["q"],)

    input_dict = {
        "q": torch.zeros(10, 1, 1, 1),
        "k_cache": torch.zeros(3, 128, 1, 1),
        "indices": torch.zeros(10, 1, 64, dtype=torch.int32),
        "topk_length": torch.ones(10, dtype=torch.int32),
        "extra_indices_in_kvcache": None,
        "extra_topk_length": None,
    }

    dsv4_radix._flash_mla_with_optional_prefill_chunking(
        input_dict=input_dict,
        backend="kernel",
        forward_mode=ForwardMode.DECODE,
        runner=fake_runner,
    )

    assert calls == [10]


def test_dsv4_paged_mqa_metadata_and_logits_chunking(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_PREFILL_METADATA_CHUNK_SIZE", "4")
    metadata_calls = []
    logits_calls = []

    class FakeDeepGemm:
        @staticmethod
        def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms):
            metadata_calls.append((context_lens.clone(), block_kv, num_sms))
            return {"rows": context_lens.shape[0]}

        @staticmethod
        def fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            clean_logits,
        ):
            logits_calls.append(
                {
                    "q": q.clone(),
                    "weights": weights.clone(),
                    "context_lens": context_lens.clone(),
                    "block_table": block_table.clone(),
                    "schedule_meta": schedule_meta,
                    "max_context_len": max_context_len,
                    "clean_logits": clean_logits,
                }
            )
            return q[:, 0, 0, :1].to(torch.float32)

    monkeypatch.setattr(paged_mqa, "deep_gemm", FakeDeepGemm)

    context_lens = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=torch.int32)
    schedule_meta = paged_mqa.get_paged_mqa_logits_metadata_chunked(
        context_lens=context_lens,
        block_kv=64,
        num_sms=132,
    )

    assert [call[0].flatten().tolist() for call in metadata_calls] == [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2],
    ]
    assert [chunk.schedule_meta["rows"] for chunk in schedule_meta.chunks] == [4, 4, 2]

    q = torch.arange(10 * 1 * 1 * 2, dtype=torch.float32).view(10, 1, 1, 2)
    kv_cache = torch.zeros(3, 64, 1, 132)
    weights = torch.arange(10, dtype=torch.float32).view(10, 1)
    block_table = torch.arange(10 * 3, dtype=torch.int32).view(10, 3)

    logits = paged_mqa.fp8_paged_mqa_logits_chunked(
        q=q,
        kv_cache=kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_meta=schedule_meta,
        max_context_len=192,
        clean_logits=False,
    )

    assert logits.shape == (10, 1)
    assert torch.equal(logits.flatten(), q[:, 0, 0, 0])
    assert [call["q"].shape[0] for call in logits_calls] == [4, 4, 2]
    assert [call["weights"].flatten().tolist() for call in logits_calls] == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0],
    ]
    assert [call["block_table"][:, 0].tolist() for call in logits_calls] == [
        [0, 3, 6, 9],
        [12, 15, 18, 21],
        [24, 27],
    ]
