"""Synthetic snapshot fixtures shared by unit and Store integration tests."""

import ctypes
import time
from types import SimpleNamespace

import requests
import torch

from sglang.srt.training_capture.protocol import (
    DTYPES,
    KVSpec,
    LayerGeometry,
    Provenance,
    SequenceInfo,
    TeacherIdentity,
    canonical_bytes,
    decode_manifest,
    digest_bytes,
    tensor_bytes,
    validate_tensors,
)
from sglang.srt.training_capture.snapshot import SnapshotMetadata, build_snapshot
from sglang.srt.training_capture.teacher import capture_teacher


class CaptureTestRequest(SimpleNamespace):
    def __init__(self, rid, max_new_tokens=3):
        from sglang.srt.sampling.sampling_params import SamplingParams

        super().__init__(
            rid=rid,
            is_retracted=False,
            output_ids=[],
            finished_reason=None,
            finished_len=None,
            eos_token_ids=set(),
            lora_id=None,
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            session=None,
            custom_logit_processor=None,
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
            origin_input_ids=[3, 4],
            training_capture_attempted=False,
            training_capture_context=None,
            training_capture_pd=None,
            training_capture_finalize=None,
            training_capture_latency=None,
            time_stats=SimpleNamespace(scheduler_recv_time=0.0),
        )

    def finished(self):
        return self.finished_reason is not None

    @property
    def output_ids_through_stop(self):
        return self.output_ids[: self.finished_len]


def read_snapshot(store, publication):
    data = store.get_tensor(
        publication["manifest_key"],
        [publication["manifest_nbytes"]],
        torch.uint8,
        publication["manifest_sha256"],
    )
    manifest = decode_manifest(bytes(tensor_bytes(data)))
    tensors = {
        obj.key: store.get_tensor(obj.key, obj.shape, DTYPES[obj.dtype], obj.sha256)
        for obj in manifest.objects
    }
    validate_tensors(manifest, tensors)
    packed = {
        obj.name: tensors[obj.key] for obj in manifest.objects if obj.kind == "aux"
    }
    nv = max(obj.token_range[1] for obj in manifest.objects if obj.kind == "kv")
    for layer in manifest.kv.layers:
        for component, dim in (("k", layer.key_head_dim), ("v", layer.value_head_dim)):
            packed[f"target_{component}.{layer.layer_id}"] = torch.empty(
                nv, layer.num_kv_heads, dim, dtype=DTYPES[manifest.kv.dtype]
            )
    for obj in manifest.objects:
        if obj.kind == "kv":
            t0, t1 = obj.token_range
            h0, h1 = obj.head_range
            packed[obj.name][t0:t1, h0:h1].copy_(tensors[obj.key])
    return manifest, packed


def exercise_capture_abort(
    test, *, url, rid, prompt, max_new_tokens, distributed=False
):
    """Cancel a live stream and require a fenced failure without publication."""
    state = requests.get(url + "/server_info", timeout=10).json()["internal_states"][0][
        "training_capture"
    ]
    previous_admitted = state["counters"].get("admitted", 0)
    previous_cancelled = state.get("request_router", {}).get("cancelled", 0)
    reason = "cohort_failed" if distributed else "request_aborted_or_retracted"
    with test.catalog.condition:
        previous_publications = len(test.catalog.publications)
        previous_failures = {
            capture_id
            for capture_id, record in test.catalog.captures.items()
            if record["state"] == "FAILED"
        }
    with requests.post(
        url + "/generate",
        json={
            "rid": rid,
            "input_ids": prompt,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
                "logit_bias": {"100": 100.0},
            },
        },
        stream=True,
        timeout=120,
    ) as response:
        test.assertEqual(
            response.status_code,
            200,
            response.text if response.status_code != 200 else "",
        )
        for line in response.iter_lines(chunk_size=1):
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                status = requests.post(
                    url + "/abort_request", json={"rid": rid}, timeout=20
                )
                test.assertEqual(status.status_code, 200, status.text)
                break
        else:
            test.fail("stream ended without any output before abort")
    with test.catalog.condition:
        test.assertTrue(
            test.catalog.condition.wait_for(
                lambda: any(
                    capture_id not in previous_failures
                    and record["state"] == "FAILED"
                    and record.get("reason") == reason
                    for capture_id, record in test.catalog.captures.items()
                ),
                timeout=20,
            ),
            "aborted capture did not finish its fenced Catalog failure",
        )
        test.assertEqual(len(test.catalog.publications), previous_publications)
    deadline = time.monotonic() + 20
    while True:
        state = requests.get(url + "/server_info", timeout=10).json()[
            "internal_states"
        ][0]["training_capture"]
        if state["states"].get("available", 0) == state["reservations"]:
            break
        test.assertLess(time.monotonic(), deadline, state)
        time.sleep(0.05)
    test.assertEqual(state["counters"].get("admitted", 0), previous_admitted + 1)
    if distributed:
        test.assertGreater(
            state["request_router"].get("cancelled", 0), previous_cancelled
        )
    else:
        test.assertGreater(
            state["counters"].get("failed_request_aborted_or_retracted", 0), 0
        )
    test.assertEqual(state["host_pool"]["quarantined"], 0)
    return state


class VerifyCaptureFixture:
    def __init__(self, coordinator, request, *, prefill_pending=False):
        from sglang.srt.training_capture.coordinator import CaptureBatch, CaptureStep

        self.coordinator, self.request = coordinator, request
        coordinator.capture_mode = "speculative_accepted_target_path"
        coordinator.enable_overlap = prefill_pending
        coordinator.before_forward([request])
        self.record = request.training_capture_context
        for index, buffer in enumerate(coordinator.exporter.buffers.values()):
            buffer.copy_(torch.arange(buffer.numel()).view_as(buffer) + index * 256)
        self.sources = {k: v.clone() for k, v in coordinator.exporter.buffers.items()}
        context = self.record.context
        context.export_kv(coordinator.exporter, torch.tensor([7, 3]), end=2)
        self.prefill_logits = torch.arange(256).float()[None]
        context.record_teacher(
            capture_teacher(self.prefill_logits, 256), row=0, position=2
        )
        self.prefill_result = CaptureBatch((CaptureStep(request, self.record, 2),))
        if not prefill_pending:
            context.commit_token(position=2, token_id=10)
            request.output_ids.append(10)

    def forward(
        self,
        *,
        inputs=(10, 20, 30, 40),
        prefix=2,
        selected_row=0,
        verify_lens=None,
        padding=0,
    ):
        num_requests = len(verify_lens) if verify_lens is not None else selected_row + 1
        requests = [CaptureTestRequest("not-selected") for _ in range(num_requests)]
        requests[selected_row] = self.request
        width = len(inputs)
        self.forward_batch = SimpleNamespace(
            input_ids=torch.tensor(list(inputs) * len(requests)),
            positions=torch.arange(prefix, prefix + width).repeat(len(requests)),
            out_cache_loc=torch.tensor([6, 1, 9, 5]).repeat(len(requests)),
        )
        generator = torch.Generator().manual_seed(prefix)
        self.logits = torch.randn(len(requests) * width, 256, generator=generator)
        if verify_lens is not None:
            indices = torch.tensor(
                [
                    row * width + column
                    for row, count in enumerate(verify_lens)
                    for column in range(count)
                ],
                dtype=torch.long,
            )
            for name in ("input_ids", "positions", "out_cache_loc"):
                selected = getattr(self.forward_batch, name).index_select(0, indices)
                setattr(
                    self.forward_batch,
                    name,
                    torch.nn.functional.pad(selected, (0, padding)),
                )
            self.logits = torch.nn.functional.pad(
                self.logits[indices], (0, 0, 0, padding), value=999
            )
        return self.coordinator.after_verify_forward(
            SimpleNamespace(reqs=requests, seq_lens_cpu=[prefix] * len(requests)),
            self.forward_batch,
            SimpleNamespace(next_token_logits=self.logits),
            width=width,
            can_run_cuda_graph=False,
            verify_lens=torch.tensor(verify_lens) if verify_lens is not None else None,
        )

    def accept(self, ticket, outputs, count):
        return self.coordinator.after_verify_accept(
            ticket, commit_lens=torch.tensor(count), out_tokens=torch.tensor(outputs)
        )

    def finish(self, outputs, length, reason=None):
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        self.request.output_ids = outputs
        self.request.finished_len = length
        self.request.finished_reason = reason or FINISH_LENGTH(length)
        self.coordinator.on_release(self.request)


class OverlapCaptureFixture:
    def __init__(self, coordinator, request):
        self.coordinator, self.request = coordinator, request
        coordinator.enable_overlap = True
        request.req_pool_idx = 0
        self.slots = torch.tensor([7, 3, 6, 1, 9, 5, 8, 2])
        coordinator.req_to_token = SimpleNamespace(req_to_token=self.slots[None])
        coordinator.before_forward([request])
        self.record = request.training_capture_context
        for index, buffer in enumerate(coordinator.exporter.buffers.values()):
            buffer.copy_(torch.arange(buffer.numel()).view_as(buffer) + index * 256)
        self.sources = {k: v.clone() for k, v in coordinator.exporter.buffers.items()}

    def forward(self, end):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        extend = end == 2
        return self.coordinator.after_forward(
            SimpleNamespace(
                reqs=[self.request],
                seq_lens_cpu=[end],
                forward_mode=ForwardMode.EXTEND if extend else ForwardMode.DECODE,
            ),
            SimpleNamespace(
                extend_seq_lens_cpu=[2],
                positions=torch.arange(0 if extend else end - 1, end),
            ),
            SimpleNamespace(next_token_logits=torch.arange(256).float()[None] + end),
            can_run_cuda_graph=False,
        )


def make_kv_spec():
    rope = {
        "type": "default",
        "theta": 10000.0,
        "rotary_dim": 4,
        "interleaved": False,
        "scaling": None,
    }
    return KVSpec(
        codec="dense_bf16_post_rope_v1",
        dtype="bfloat16",
        selected_layer_ids=[3, 1],
        layers=[
            LayerGeometry(layer_id=i, num_kv_heads=2, key_head_dim=4, value_head_dim=4)
            for i in (3, 1)
        ],
        source_k_stage="post_rope",
        source_k_norm="none",
        rope_config=rope,
        rope_config_sha256=digest_bytes(canonical_bytes(rope)),
        storage_chunk_tokens=2,
        source_page_size=2,
    )


def make_snapshot(response_length=3):
    p, r = 2, response_length
    n = p + r
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(r, 256, generator=generator)
    teacher = capture_teacher(logits, 256)
    buffers = {
        "token_ids": torch.arange(n, dtype=torch.int32) + 3,
        "position_ids": torch.arange(n, dtype=torch.int64),
        "loss_mask": torch.tensor([0] * p + [1] * r, dtype=torch.uint8),
        "kv_valid": torch.tensor([1] * (n - 1) + [0], dtype=torch.uint8),
        "logits_positions": torch.arange(p, n, dtype=torch.int32),
        "teacher_topk_ids": teacher.token_ids,
        "teacher_topk_logits": teacher.logits,
        "teacher_logsumexp": teacher.logsumexp,
    }
    kv = make_kv_spec()
    for layer in kv.layers:
        for component in ("k", "v"):
            buffers[f"target_{component}.{layer.layer_id}"] = torch.randn(
                n - 1, 2, 4, generator=generator
            ).bfloat16()
    metadata = SnapshotMetadata(
        dataset_id="test-dataset",
        sample_id="test-sample",
        generation_id="g1",
        teacher=TeacherIdentity(
            model_id="synthetic-dense",
            weights_revision="weights-1",
            adapter_revision=None,
            tokenizer_revision="tokenizer-1",
            fingerprint_sha256="1" * 64,
            vocab_size=256,
            output_transform="identity",
        ),
        sequence=SequenceInfo(
            prompt_length=p, response_length=r, total_length=n, stop_reason="length"
        ),
        kv=kv,
        provenance=Provenance(
            capture_mode="autoregressive",
            producer_revision="unit-test",
            capture_config_sha256="2" * 64,
            sampling_config={"temperature": 0.8},
            trace_id="trace-1",
        ),
    )
    return build_snapshot(metadata, buffers, valid_kv_tokens=n - 1)


class FakeReplicateConfig:
    with_hard_pin = False
    replica_num = 1


class BufferStore:
    """Byte-addressed fake with the SDK's distinct status/byte-count returns."""

    def __init__(self):
        self.data = {}
        self.registered = {}
        self.put_status = 0
        self.get_count = None
        self.unregister_status = 0
        self.put_keys = []
        self.put_batches = []
        self.exists_batches = []
        self.closed = False

    def register_buffer(self, pointer, size):
        self.registered[pointer] = size
        return 0

    def unregister_buffer(self, pointer):
        if self.unregister_status == 0:
            del self.registered[pointer]
        return self.unregister_status

    def put_from(self, key, pointer, size, config):
        assert config.with_hard_pin
        self.put_keys.append(key)
        if self.put_status == 0:
            self.data.setdefault(key, ctypes.string_at(pointer, size))
        return self.put_status

    def get_into(self, key, pointer, size):
        if key not in self.data:
            return -1
        data = self.data[key]
        count = min(size, len(data)) if self.get_count is None else self.get_count
        if count >= 0:
            ctypes.memmove(pointer, data, min(count, size, len(data)))
        return count

    def is_exist(self, key):
        return int(key in self.data)

    def batch_is_exist(self, keys):
        self.exists_batches.append(list(keys))
        return [self.is_exist(key) for key in keys]

    def batch_put_from(self, keys, pointers, sizes, config):
        assert len(keys) == len(pointers) == len(sizes)
        self.put_batches.append(list(keys))
        return [
            self.put_from(key, pointer, size, config)
            for key, pointer, size in zip(keys, pointers, sizes)
        ]

    def remove(self, key, force=False):
        assert not force
        self.data.pop(key, None)
        return 0

    def close(self):
        self.closed = True
        self.registered.clear()


class Registrar:
    def __init__(self):
        self.buffers = {}
        self.registrations = 0

    def register(self, tensor):
        self.buffers[tensor.data_ptr()] = tensor
        self.registrations += 1

    def unregister(self, tensor):
        del self.buffers[tensor.data_ptr()]
