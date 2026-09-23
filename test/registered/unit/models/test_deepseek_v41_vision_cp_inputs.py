"""Vision inputs under prefill CP merge on the full extend layout before the shard."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.utils import prepare_cp_forward
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

HIDDEN = 8
VOCAB = 64
IMAGE_TOKEN_ID = 7
CP_SIZE = 4
# (prefix_len, extend_len) per request. Request 1 carries one image whose span
# [2, 8] starts inside its prefix, so only span rows 1..6 land in this chunk.
CHUNKS = [(0, 7), (3, 9), (1, 5)]
IMAGE_OFFSET = (2, 8)
IMAGE_HASH = 12345
NUM_TOKENS = sum(extend_len for _, extend_len in CHUNKS)
# 21 tokens over 4 ranks give logical [6, 5, 5, 5], padded to the CP alignment.
PHYSICAL_ROWS = 8
IMAGE_ROWS = torch.arange(7, 13)
POSITIONS = torch.cat([torch.arange(p, p + n) for p, n in CHUNKS])


def _image_span(item: MultimodalDataItem) -> torch.Tensor:
    start, end = item.offsets[0]
    rows = end - start + 1
    return torch.arange(rows * HIDDEN, dtype=torch.float32).view(rows, HIDDEN) + 100.0


def _pad(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros(PHYSICAL_ROWS - x.shape[0], *x.shape[1:])])


class _RecordingBody:
    def __init__(self, embed: nn.Embedding):
        self.embed = embed
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def __call__(self, input_ids, positions, forward_batch, input_embeds=None):
        self.calls.append(
            SimpleNamespace(
                input_ids=input_ids,
                input_embeds=input_embeds,
                input_ids_global=forward_batch.input_ids_global,
            )
        )
        return input_embeds, input_embeds


class _VisionStub(DeepseekV4ForCausalLM):
    def __init__(self, embed: nn.Embedding):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.vision = object()
        self.tp_size = 1
        self.mm_owner_group = None
        self.model = _RecordingBody(embed)
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.lm_head = object()
        self.capture_aux_hidden_states = False
        self.logits_calls = []

    def get_image_feature(self, items):
        return [_image_span(item) for item in items]

    def logits_processor(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states=None,
        hidden_states_before_norm=None,
    ):
        self.logits_calls.append(
            SimpleNamespace(input_ids=input_ids, hidden_states=hidden_states)
        )
        return object()


def _build_batch():
    item = MultimodalDataItem(
        modality=Modality.IMAGE, feature=torch.zeros(1), offsets=[IMAGE_OFFSET]
    )
    item.set_hash(IMAGE_HASH)
    ids = list(range(10, 17))
    ids += [item.pad_value] * len(IMAGE_ROWS) + [20, 21, 22]
    ids += list(range(30, 35))
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mm_inputs=[
            MultimodalInputs(mm_items=[]),
            MultimodalInputs(mm_items=[item], im_token_id=IMAGE_TOKEN_ID),
            None,
        ],
        extend_prefix_lens_cpu=[prefix for prefix, _ in CHUNKS],
        extend_seq_lens_cpu=[extend_len for _, extend_len in CHUNKS],
        seq_lens_cpu=[prefix + extend_len for prefix, extend_len in CHUNKS],
        input_ids=torch.tensor(ids, dtype=torch.long),
        positions=POSITIONS.clone(),
        mm_input_embeds=None,
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=None,
        input_ids_global=torch.zeros(1, dtype=torch.long),
    )
    return forward_batch, item


def _expected_embeds(embed, scheduler_ids, item):
    with torch.no_grad():
        full = embed(scheduler_ids.clamp(max=VOCAB - 1))
    full[IMAGE_ROWS] = _image_span(item)[1:7]
    return full


def _canonical(scheduler_ids):
    canonical = scheduler_ids.clone()
    canonical[IMAGE_ROWS] = IMAGE_TOKEN_ID
    return canonical


class TestDeepseekV41VisionPrefillCPInputs(CustomTestCase):
    def setUp(self):
        mm_schedule.init_mm_embedding_cache(1 << 20)
        init_cp_strategy(
            enable_prefill_cp=True, cp_size=CP_SIZE, cp_strategy="interleave"
        )
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.model = _VisionStub(self.embed)

    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="interleave")

    @contextmanager
    def _cp_collectives(self, full: torch.Tensor, rank: int):
        def all_gather(output, input_tensor):
            # Peers contribute their expected shards; this rank's rows come from
            # what the runner actually handed to the collective.
            output.zero_()
            for peer in range(CP_SIZE):
                rows = full[peer::CP_SIZE]
                output[peer * PHYSICAL_ROWS : peer * PHYSICAL_ROWS + rows.shape[0]] = (
                    rows
                )
            output[rank * PHYSICAL_ROWS : (rank + 1) * PHYSICAL_ROWS] = input_tensor

        with (
            patch("torch.cuda.current_stream", return_value=None),
            patch(
                "sglang.srt.layers.cp.interleave.attn_cp_all_gather_into_tensor",
                side_effect=all_gather,
            ),
            patch(
                "sglang.srt.layers.cp.interleave.is_allocation_symmetric",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.cp.interleave.use_symmetric_memory",
                return_value=torch.no_grad(),
            ),
        ):
            yield

    def test_cp_runner_merges_before_shard(self):
        runner = EagerRunner.__new__(EagerRunner)
        runner.model_runner = SimpleNamespace(model=self.model)
        padded = torch.zeros(CP_SIZE * PHYSICAL_ROWS, dtype=torch.long)

        for rank in range(CP_SIZE):
            forward_batch, item = _build_batch()
            scheduler_ids = forward_batch.input_ids.clone()
            canonical = _canonical(scheduler_ids)
            full = _expected_embeds(self.embed, scheduler_ids, item)
            padded[:NUM_TOKENS] = canonical
            rank_major_ids = padded.view(-1, CP_SIZE).T.flatten()
            self.model.model.calls.clear()
            self.model.logits_calls.clear()

            with (
                get_parallel().override(
                    attn_cp_rank=rank, attn_cp_size=CP_SIZE, attn_cp_group=object()
                ),
                self._cp_collectives(full, rank),
                torch.no_grad(),
            ):
                prepare_cp_forward(forward_batch)
                runner._execute_extend_cp(forward_batch, {})

            with self.subTest(rank=rank):
                metadata = forward_batch.attn_cp_metadata
                self.assertEqual(metadata.per_rank_actual_token, [PHYSICAL_ROWS] * 4)
                (body,) = self.model.model.calls
                self.assertTrue(
                    torch.equal(body.input_ids, _pad(canonical[rank::CP_SIZE]))
                )
                self.assertTrue(
                    torch.equal(body.input_embeds, _pad(full[rank::CP_SIZE]))
                )
                self.assertTrue(torch.equal(body.input_ids_global, rank_major_ids))

                (logits,) = self.model.logits_calls
                self.assertTrue(torch.equal(logits.input_ids, canonical))
                self.assertTrue(torch.equal(logits.hidden_states, full))

                self.assertTrue(torch.equal(forward_batch.mm_input_embeds, full))
                self.assertTrue(torch.equal(forward_batch.input_ids, scheduler_ids))

    def test_external_embeddings_with_images_are_rejected(self):
        forward_batch, _ = _build_batch()
        with self.assertRaisesRegex(ValueError, "Cannot combine"):
            self.model.prepare_model_inputs(
                input_ids=forward_batch.input_ids,
                forward_batch=forward_batch,
                input_embeds=torch.zeros(NUM_TOKENS, HIDDEN),
            )


if __name__ == "__main__":
    unittest.main()
