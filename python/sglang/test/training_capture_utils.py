"""Synthetic snapshot fixtures shared by unit and Store integration tests."""

import ctypes

import torch

from sglang.srt.training_capture.protocol import (
    KVSpec,
    LayerGeometry,
    Provenance,
    SequenceInfo,
    TeacherIdentity,
    canonical_bytes,
    digest_bytes,
)
from sglang.srt.training_capture.snapshot import SnapshotMetadata, build_snapshot
from sglang.srt.training_capture.teacher import capture_teacher


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
