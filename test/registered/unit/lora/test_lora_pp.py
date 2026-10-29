# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.mem_pool import EMPTY_SLOT, LoRAMemoryPool
from sglang.srt.lora.utils import LoRAType
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.runtime_context import get_context, get_parallel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _req(lora_id, finished=False):
    return SimpleNamespace(lora_id=lora_id, finished=lambda: finished)


def _batch(*reqs):
    return SimpleNamespace(reqs=list(reqs))


def _lora_manager(slots):
    pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
    pool.max_loras_per_batch = slots
    pool.uid_to_buffer_id = {}
    pool.buffer_id_to_uid = [EMPTY_SLOT] * slots
    pool.eviction_policy = get_eviction_policy("lru")
    pool.load_lora_weight_to_buffer = lambda *_args: None

    manager = LoRAManager.__new__(LoRAManager)
    manager.max_loras_per_batch = slots
    manager.memory_pool = pool
    manager.num_pinned_loras = 0
    manager.loras = {}
    manager.lora_refs = {}
    manager.lora_modules = []
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    return manager


class TestAllPPLoras(CustomTestCase):
    def test_covers_every_micro_batch_except_finished_requests(self):
        """A launched prefill not yet merged into its decode batch still counts;
        requests that finished before their micro-batch's next turn do not."""
        scheduler = Scheduler.__new__(Scheduler)
        decoding, base, launched = _req("decoding"), _req(None), _req("launched")
        scheduler.running_mbs = [
            _batch(decoding, _req("decoding", finished=True)),
            _batch(base),
        ]
        scheduler.mbs = [_batch(launched), None]
        self.assertCountEqual(
            map(id, scheduler._get_all_pp_reqs()), map(id, [decoding, base, launched])
        )

        scheduler.enable_lora = True
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(lora_manager=_lora_manager(slots=4))
        )
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.scheduler = scheduler
        reporter.stats = SimpleNamespace()
        with get_parallel().override(pp_size=2):
            reporter._update_lora_metrics()
        self.assertEqual(
            (
                reporter.stats.lora_pool_slots_used,
                reporter.stats.lora_pool_slots_total,
                reporter.stats.lora_pool_utilization,
            ),
            (2, 4, 0.5),
        )

    @mock.patch("time.monotonic", return_value=100.0)
    def test_draining_uses_all_micro_batches(self, _clock):
        def request(lora_id, max_new_tokens):
            req = _req(lora_id)
            req.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
            req.output_ids = []
            req.is_retracted = False
            req.time_stats = SimpleNamespace(wait_queue_entry_time=0.0)
            return req

        short, long = request("short", 5), request("long", 100)
        waiting = request("waiting", 100)
        first, second = _batch(short), _batch(long)
        for batch in (first, second):
            batch.batch_is_full = False
            batch.is_empty = lambda: False

        # Stop after LoRA admission setup, before allocating or running a prefill.
        adder = SimpleNamespace(
            can_run_list=[], preempt_list=[], budget_state=lambda: AddReqResult.OTHER
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.__dict__.update(
            running_mbs=[first, second],
            mbs=[first, second],
            waiting_queue=[waiting],
            lora_drainer=LoRADrainer(max_loras_per_batch=2, max_wait_time_secs=3),
            enable_lora=True,
            enable_lora_overlap_loading=False,
            enable_priority_preemption=False,
            is_hybrid_swa=False,
            chunked_req=None,
            min_free_slots_delayer=None,
            is_mixed_chunk=False,
            dllm_config=None,
            grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
            policy=mock.Mock(),
            processed_tokens_counter=0,
            disagg_prefill_bootstrap_queue=None,
            get_num_allocatable_reqs=lambda *args, **kwargs: 1,
            get_prefill_adder_class=lambda: lambda *args, **kwargs: adder,
            page_size=1,
            tree_cache=SimpleNamespace(buffer_pipeline=None),
            token_to_kv_pool_allocator=SimpleNamespace(state_slots=None),
            req_to_token_pool=SimpleNamespace(),
            new_token_ratio_tracker=SimpleNamespace(current=1),
            max_prefill_tokens=1,
            chunked_prefill_size=1,
            priority_scheduling_preemption_threshold=0,
            max_prefill_bs=1,
            max_running_requests=2,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(), lora_manager=_lora_manager(slots=2)
                )
            ),
        )

        with get_context().override_server_args(pp_size=2):
            # Neither microbatch alone fills the pool. Alternating them must
            # keep the shorter adapter draining for the waiting third adapter.
            for running_batch in (first, second, first):
                scheduler._get_new_batch_prefill_raw(None, running_batch)
                self.assertFalse(
                    scheduler.lora_drainer.can_schedule(request("short", 100))
                )
                self.assertFalse(
                    scheduler.can_schedule_lora_req(
                        waiting, {req.lora_id for req in scheduler._get_all_pp_reqs()}
                    )
                )

            short.finished = lambda: True
            scheduler._get_new_batch_prefill_raw(None, second)
            self.assertTrue(
                scheduler.can_schedule_lora_req(
                    waiting, {req.lora_id for req in scheduler._get_all_pp_reqs()}
                )
            )


class TestFetchKeepsLiveAdapters(CustomTestCase):
    def test_evicts_an_idle_adapter_instead_of_a_live_one(self):
        """LRU alone would evict "live": it was used least recently."""
        manager = _lora_manager(slots=2)
        manager.fetch_new_loras({"live"})
        manager.fetch_new_loras({"idle"})

        manager.fetch_new_loras({"new"}, pp_loras={"live"})

        self.assertEqual(set(manager.memory_pool.uid_to_buffer_id), {"live", "new"})

    def test_still_loads_the_batch_when_live_adapters_do_not_fit(self):
        manager = _lora_manager(slots=2)
        manager.fetch_new_loras({"a", "b"})

        manager.fetch_new_loras({"c", "d"}, pp_loras={"a", "b"})

        self.assertEqual(set(manager.memory_pool.uid_to_buffer_id), {"c", "d"})


# A 4-layer model on two PP stages: the first holds embed_tokens and layers 0-1,
# the last holds layers 2-3 and lm_head.
NUM_LAYERS, HIDDEN, VOCAB, RANK = 4, 8, 32, 2


def _stage_pool(layers, embed_tokens=False, lm_head=False, added_tokens_size=0):
    config = SimpleNamespace(
        num_hidden_layers=NUM_LAYERS, hidden_size=HIDDEN, vocab_size=VOCAB
    )
    dims = {"embed_tokens": (VOCAB, HIDDEN), "lm_head": (HIDDEN, VOCAB)}
    base_model = torch.nn.Linear(HIDDEN, HIDDEN)
    base_model.config = config
    base_model.get_hidden_dim = lambda name, _: dims.get(name, (HIDDEN, HIDDEN))
    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1):
        return LoRAMemoryPool(
            base_hf_config=config,
            max_loras_per_batch=1,
            dtype=torch.float32,
            max_lora_rank=RANK,
            target_modules={"o_proj", "embed_tokens", "lm_head"},
            base_model=base_model,
            eviction_policy="lru",
            lora_added_tokens_size=added_tokens_size,
            lora_layer_ids=layers,
            has_embed_tokens_lora=embed_tokens,
            has_lm_head_lora=lm_head,
        )


def _allocated(pool):
    return {
        "layers": [i for i, b in enumerate(pool.A_buffer["o_proj"]) if b is not None],
        "embed_tokens": bool(pool.embedding_A_buffer),
        "lm_head": bool(pool.lm_head_A_buffer),
    }


def _o_proj_weight(layer):
    return torch.full((RANK, HIDDEN), float(layer))


def _full_adapter():
    layers = []
    for i in range(NUM_LAYERS):
        o_proj = f"model.layers.{i}.self_attn.o_proj"
        weights = {
            f"{o_proj}.lora_A.weight": _o_proj_weight(i),
            f"{o_proj}.lora_B.weight": torch.ones(HIDDEN, RANK),
        }
        layers.append(SimpleNamespace(weights=weights, pinned_weights={}))
    return SimpleNamespace(
        config=SimpleNamespace(r=RANK, lora_added_tokens_size=0),
        layers=layers,
        embedding_layers={
            "model.embed_tokens.lora_embedding_A": torch.ones(RANK, VOCAB),
            "model.embed_tokens.lora_embedding_B": torch.ones(HIDDEN, RANK),
        },
        added_tokens_embeddings={},
        pinned_embedding_layers={},
        pinned_added_tokens_embeddings={},
    )


def _load_into_mem_pool(pool, uid, adapter):
    no_slicing = SimpleNamespace(
        slice_lora_a_weights=lambda w: w, slice_lora_b_weights=lambda w: w
    )
    lora_modules = [{} for _ in range(NUM_LAYERS)]
    for i in pool.lora_layer_ids:
        lora_modules[i][f"model.layers.{i}.self_attn.o_proj"] = no_slicing
    pool.load_lora_weight_to_buffer(uid, 0, adapter, lora_modules, None, None)


class TestStagePool(CustomTestCase):
    def test_each_stage_allocates_and_loads_only_its_part(self):
        first = _stage_pool(layers=[0, 1], embed_tokens=True)
        last = _stage_pool(layers=[2, 3], lm_head=True, added_tokens_size=1)
        self.assertEqual(
            _allocated(first),
            {"layers": [0, 1], "embed_tokens": True, "lm_head": False},
        )
        self.assertEqual(
            _allocated(last), {"layers": [2, 3], "embed_tokens": False, "lm_head": True}
        )
        self.assertIsNone(last.get_embedding_tensor("added_tokens", LoRAType.LORA_A))

        # The last stage must skip the adapter's layers 0-1 and embed_tokens weights.
        _load_into_mem_pool(last, "adapter", _full_adapter())
        for i in (2, 3):
            self.assertTrue(
                torch.equal(last.A_buffer["o_proj"][i][0], _o_proj_weight(i))
            )

        # Handing the slot to the base model zeroes it.
        _load_into_mem_pool(last, None, None)
        for i in (2, 3):
            self.assertFalse(last.A_buffer["o_proj"][i][0].any())


if __name__ == "__main__":
    unittest.main()
