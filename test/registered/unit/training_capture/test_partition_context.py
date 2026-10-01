"""Owner-local token ledgers must produce one coherent distributed snapshot."""

import unittest
from contextlib import ExitStack

import torch
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.protocol import ContractError, validate_tensors
from sglang.srt.training_capture.snapshot import SnapshotMetadata, assemble_snapshot
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import Registrar, make_snapshot

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPartitionContext(CustomTestCase):
    def run_trace(self, *, verify, divergent=False):
        base, _ = make_snapshot()
        kv = base.kv
        layout = plan_capture_layout(
            kv,
            tp_size=4,
            pp_layer_ranges=[(0, 4), (4, 6)],
            aux_tp_rank=1,
        )
        indices = torch.tensor([7, 2, 12, 4, 9, 1], dtype=torch.int32)
        full_sources = {
            f"target_{component}.{layer.layer_id}": (
                torch.arange(16 * 8).reshape(16, 2, 4).bfloat16() + layer.layer_id * 100
            )
            for layer in kv.layers
            for component in ("k", "v")
        }
        prepared, tensors, contexts = [], {}, []
        with ExitStack() as stack:
            for partition in layout.partitions:
                if not partition.active:
                    continue
                pool = HostBufferPool(
                    kv=kv,
                    max_tokens=8,
                    slots=1,
                    max_bytes=2 << 20,
                    registrar=Registrar(),
                    pin_memory=False,
                    partition=partition,
                    device=torch.device("cpu"),
                    kv_d2h_batch_tokens=3,
                    max_device_bytes=4096,
                )
                stack.callback(pool.close)
                slot = pool.acquire()
                stack.callback(pool.release, slot, transfer_complete=True)
                context = RequestCaptureContext(
                    slot=slot,
                    prompt_ids=(3, 4),
                    max_tokens=8,
                    vocab_size=256,
                    partition=partition,
                )
                contexts.append(context)
                sources = {
                    f"target_{component}.{heads.layer_id}": full_sources[
                        f"target_{component}.{heads.layer_id}"
                    ][:, heads.start : heads.end].clone()
                    for heads in partition.heads
                    for component in ("k", "v")
                }
                exporter = (
                    SelectedLayerKVExporter(kv, sources, partition=partition)
                    if sources
                    else None
                )
                ends = (1, 2, 5) if verify else (1, 2, 3, 4)
                for end in ends:
                    start = context.kv_end
                    if exporter is not None:
                        context.export_kv(exporter, indices[start:end], end=end)
                        for source in sources.values():
                            source[indices[start:end]] = -99
                    else:
                        context.record_kv_progress(end=end)
                    if end < 2 or (verify and end == 2):
                        continue
                    count = 4 if verify else 1
                    position = 2 if verify else end
                    if partition.include_aux:
                        rows = capture_teacher(
                            torch.arange(256).float().repeat(count, 1), 256
                        )
                        context.record_teacher_range(
                            rows, row=0, position=position, count=count
                        )
                        rows.logits.zero_()
                    accepted = [10, 11] if verify else [end + 8]
                    if (
                        divergent
                        and partition.owner_id == layout.topology.owners[0]
                        and end == ends[-1]
                    ):
                        accepted[-1] += 1
                    context.observe_tokens(
                        position=position,
                        tokens=accepted + ([90, 91] if verify else []),
                    )
                    for offset, token in enumerate(accepted):
                        context.commit_token(position=position + offset, token_id=token)
                if verify:
                    context.trim_terminal_prefix()
                context.seal("length")
                metadata = SnapshotMetadata(
                    dataset_id=base.dataset_id,
                    sample_id=base.sample_id,
                    generation_id=base.generation_id,
                    teacher=base.teacher,
                    sequence=context.sequence,
                    kv=kv,
                    provenance=base.provenance,
                    topology=layout.topology,
                )
                part, payloads = context.prepare_partition(
                    **{
                        name: getattr(metadata, name)
                        for name in SnapshotMetadata.__struct_fields__
                        if name != "sequence"
                    }
                )
                prepared.append(part)
                tensors.update(payloads)
                if not partition.include_aux:
                    self.assertEqual(context.teacher_rows, 0)
                    self.assertTrue(
                        all(name.startswith("target_") for name in slot.tensors)
                    )
                    with self.assertRaises(ContractError):
                        context.snapshot()
            if divergent:
                with self.assertRaisesRegex(ContractError, "token sequence"):
                    assemble_snapshot(metadata, prepared, layout=layout)
                return
            manifest = assemble_snapshot(metadata, prepared, layout=layout)
            validate_tensors(manifest, tensors)
            self.assertEqual(len({part.token_ids_sha256 for part in prepared}), 1)
            tokens = next(
                tensors[obj.key] for obj in manifest.objects if obj.name == "token_ids"
            )
            self.assertEqual(
                tokens.tolist(), [3, 4, 10, 11] if verify else [3, 4, 10, 11, 12]
            )
            for obj in manifest.objects:
                if obj.kind != "kv":
                    continue
                start, end = obj.token_range
                first, last = obj.head_range
                expected = full_sources[obj.name][indices[start:end], first:last]
                torch.testing.assert_close(tensors[obj.key], expected, rtol=0, atol=0)
                self.assertLessEqual(end, 4)
            aux = next(context for context in contexts if context.owns_aux)
            aux.slot.tensors["token_ids"][0] = 99
            with self.assertRaisesRegex(ContractError, "aux token payload"):
                aux.prepare_partition(
                    **{
                        name: getattr(metadata, name)
                        for name in SnapshotMetadata.__struct_fields__
                        if name != "sequence"
                    }
                )
            # Sealing one owner does not make its data publishable after a peer abort.
            contexts[0].abort("peer_failed")
            with self.assertRaises(ContractError):
                contexts[0].prepare_partition()

    def test_chunked_prefill_and_decode_keep_one_accepted_sequence_across_owners(self):
        self.run_trace(verify=False)

    def test_verify_lookahead_and_staging_tail_exclude_unaccepted_rows(self):
        self.run_trace(verify=True)

    def test_same_lengths_from_different_local_generations_cannot_assemble(self):
        self.run_trace(verify=False, divergent=True)

    def test_owners_cannot_forge_payload_progress_or_use_another_partition_slot(self):
        base, _ = make_snapshot()
        layout = plan_capture_layout(
            base.kv, tp_size=4, pp_layer_ranges=[(0, 4)], aux_tp_rank=1
        )
        with ExitStack() as stack:
            for owner in ("dp0-pp0-tp0", "dp0-pp0-tp1"):
                partition = layout.partition(owner)
                pool = HostBufferPool(
                    kv=base.kv,
                    max_tokens=8,
                    slots=1,
                    max_bytes=2 << 20,
                    registrar=Registrar(),
                    pin_memory=False,
                    partition=partition,
                )
                stack.callback(pool.close)
                slot = pool.acquire()
                stack.callback(pool.release, slot, transfer_complete=True)
                context = RequestCaptureContext(
                    slot=slot,
                    prompt_ids=(3, 4),
                    max_tokens=8,
                    vocab_size=256,
                    partition=partition,
                )
                with self.assertRaises(ContractError):
                    context.commit_token(position=2, token_id=10)
                if partition.include_aux:
                    with self.assertRaises(ContractError):
                        context.export_kv(None, torch.tensor([0]), end=1)
                    context.record_kv_progress(end=2)
                    with self.assertRaises(ContractError):
                        context.commit_token(position=2, token_id=10)
                else:
                    with self.assertRaises(ContractError):
                        context.record_kv_progress(end=2)
                    with self.assertRaises(ContractError):
                        context.record_teacher_range(None, row=0, position=2, count=1)
                    with self.assertRaises(ContractError):
                        context.record_positions(torch.arange(2), start=0)
                with self.assertRaises(ContractError):
                    RequestCaptureContext(
                        slot=slot,
                        prompt_ids=(3, 4),
                        max_tokens=8,
                        vocab_size=256,
                        partition=layout.partition(
                            "dp0-pp0-tp1" if context.owns_kv else "dp0-pp0-tp0"
                        ),
                    )
                context.abort("cancelled")
                with self.assertRaises(ContractError):
                    context.seal("length")


if __name__ == "__main__":
    unittest.main()
