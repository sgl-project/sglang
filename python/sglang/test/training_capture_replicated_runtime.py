"""Independent expected ownership for a real two-KV-head target at TP4."""

import json


class ReplicatedKVAssertions:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    def check_snapshot(self, manifest, tensors, references, *, capture_mode):
        super().check_snapshot(manifest, tensors, references, capture_mode=capture_mode)
        self.assertEqual((manifest.topology.tp_size, manifest.topology.pp_size), (4, 1))
        owners = ["dp0-pp0-tp0", "dp0-pp0-tp2"]
        self.assertEqual(manifest.topology.owners, owners)
        self.assertEqual(manifest.topology.aux_owner, owners[0])
        self.assertEqual(manifest.kv.selected_layer_ids, [0, 14, 27])
        self.assertEqual(manifest.kv.source_k_norm, "none")
        self.assertEqual(manifest.kv.rope_config["theta"], 1000000.0)
        for layer in manifest.kv.layers:
            self.assertEqual(
                (layer.num_kv_heads, layer.key_head_dim, layer.value_head_dim),
                (2, 128, 128),
            )
        for obj in manifest.objects:
            if obj.kind == "aux":
                self.assertEqual(obj.owner_id, owners[0])
            else:
                self.assertIn(obj.owner_id, owners)
                head = owners.index(obj.owner_id)
                self.assertEqual(obj.head_range, (head, head + 1))
                self.assertEqual(obj.shape[1:], [1, 128])
        frames = [
            item
            for item in references
            if item["trace_id"] == manifest.provenance.trace_id
            and item.get("capture_partition") is not None
        ]
        self.assertEqual({item["tp_rank"] for item in frames}, {0, 1, 2, 3})
        for item in frames:
            rank, part = item["tp_rank"], item["capture_partition"]
            self.assertEqual(part["owner_id"], f"dp0-pp0-tp{rank}")
            self.assertEqual(part["active"], rank in (0, 2))
            self.assertEqual(part["include_aux"], rank == 0)
            if rank in (1, 3):
                self.assertFalse(part["head_ranges"])
                self.assertEqual(part["host_allocated_bytes"], 0)
                self.assertEqual(part["device_allocated_bytes"], 0)
                self.assertFalse(item["kv"])
            else:
                self.assertEqual(
                    part["head_ranges"],
                    [(layer, rank // 2, rank // 2 + 1) for layer in (0, 14, 27)],
                )
                self.assertGreater(part["host_allocated_bytes"], 0)
                self.assertTrue(item["kv"])
                self.assertTrue(
                    all(v.shape[1:] == (1, 128) for v in item["kv"].values())
                )
        print(
            json.dumps(
                {
                    "replicated_snapshot": manifest.provenance.trace_id,
                    "model_id": manifest.teacher.model_id,
                    "teacher_fingerprint": manifest.teacher.fingerprint_sha256,
                    "weights_revision": manifest.teacher.weights_revision,
                    "tokenizer_revision": manifest.teacher.tokenizer_revision,
                    "owners": owners,
                    "nonowner_ranks_without_payload_buffers": [1, 3],
                    "capture_mode": capture_mode,
                    "tensor_objects": len(manifest.objects),
                    "tensor_bytes": manifest.total_tensor_bytes,
                }
            ),
            flush=True,
        )
