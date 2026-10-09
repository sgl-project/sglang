"""Validate position coverage and reject structurally plausible corrupt samples."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema
import msgspec
import numpy as np
import torch
from sglang.srt.training_capture import protocol
from sglang.srt.training_capture.protocol import (
    ContractError,
    _all_finite,
    canonical_bytes,
    decode_manifest,
    digest_bytes,
    tensor_bytes,
    validate_manifest,
    validate_tensors,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import make_snapshot

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSnapshotProtocol(CustomTestCase):
    def test_schema_and_single_token_page_boundary_tail_coverage(self):
        schema_path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/training_capture/schemas/manifest.schema.json"
        )
        schema = json.loads(schema_path.read_text())
        for response_length in (1, 3, 4):
            with self.subTest(response_length=response_length):
                manifest, tensors = make_snapshot(response_length)
                encoded = canonical_bytes(manifest)
                jsonschema.Draft202012Validator(
                    schema, format_checker=jsonschema.FormatChecker()
                ).validate(json.loads(encoded))
                decoded = decode_manifest(encoded)
                validate_tensors(decoded, tensors)
                n = 2 + response_length
                self.assertEqual(validate_manifest(decoded), n - 1)
                aux = {
                    o.name: tensors[o.key] for o in decoded.objects if o.kind == "aux"
                }
                self.assertEqual(aux["logits_positions"].tolist(), list(range(2, n)))
                self.assertEqual(aux["kv_valid"].tolist(), [1] * (n - 1) + [0])
                self.assertEqual([g.layer_id for g in decoded.kv.layers], [3, 1])

    def test_descriptor_corruption_and_allocation_limits(self):
        manifest, _ = make_snapshot()
        original = json.loads(canonical_bytes(manifest))

        def missing_constant(value):
            del value["logits"]["normalization"]

        mutations = {
            "missing_aux": lambda x: x["objects"].pop(0),
            "missing_kv": lambda x: x["objects"].pop(),
            "duplicate_object": lambda x: x["objects"].append(x["objects"][0]),
            "wrong_response_length": lambda x: x["sequence"].update(response_length=4),
            "foreign_key": lambda x: x["objects"][0].update(
                key="draft-data/other/sample/g1/aux/token_ids"
            ),
            "bad_rope": lambda x: x["kv"]["rope_config"].update(theta=999.0),
            "overlap_kv": lambda x: x["objects"][-1].update(token_range=[0, 2]),
            "missing_required_constant": missing_constant,
            "wrong_temperature": lambda x: x["logits"].update(lse_temperature=0.8),
            "example_only": lambda x: x["extensions"].update(example_only=True),
            "huge_allocation": lambda x: x["objects"][0].update(
                shape=[2**63], nbytes=4 * 2**63
            ),
            "missing_owner": lambda x: x["topology"]["owners"].append("pp1-tp0"),
        }
        for name, mutate in mutations.items():
            with self.subTest(corruption=name):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(ContractError):
                    decode_manifest(json.dumps(value).encode())
        with self.assertRaises(ContractError):
            decode_manifest(canonical_bytes(manifest), max_tensor_bytes=1)

    def test_content_corruption_with_valid_checksums(self):
        for name in (
            "logits_positions",
            "kv_valid",
            "loss_mask",
            "teacher_topk_ids",
            "teacher_logsumexp",
        ):
            with self.subTest(field=name):
                manifest, tensors = make_snapshot()
                obj = next(o for o in manifest.objects if o.name == name)
                value = tensors[obj.key].clone()
                if name == "teacher_topk_ids":
                    value[0, 1] = value[0, 0]
                elif name == "teacher_logsumexp":
                    value -= 100
                elif name == "loss_mask":
                    value[0] = 1
                else:
                    value[0] += 1
                tensors[obj.key] = value
                replacement = msgspec.structs.replace(
                    obj, sha256=digest_bytes(tensor_bytes(value))
                )
                manifest = msgspec.structs.replace(
                    manifest,
                    objects=[
                        replacement if o.key == obj.key else o for o in manifest.objects
                    ],
                )
                with self.assertRaises(ContractError):
                    validate_tensors(manifest, tensors)

    def test_byte_corruption_cannot_pass_descriptor_validation_alone(self):
        manifest, tensors = make_snapshot()
        tensors[manifest.objects[0].key][0] += 1
        validate_manifest(manifest)
        with self.assertRaisesRegex(ContractError, "checksum"):
            validate_tensors(manifest, tensors)

    def test_all_bfloat16_and_float16_encodings_match_torch(self):
        words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        for dtype in (torch.bfloat16, torch.float16):
            values = words.view(dtype)
            expected = torch.isfinite(values)
            data = tensor_bytes(values)
            name = str(dtype).removeprefix("torch.")
            for index, finite in enumerate(expected.tolist()):
                actual = _all_finite(data[index * 2 : (index + 1) * 2], name)
                self.assertEqual(actual, finite, (name, hex(index)))
            self.assertTrue(_all_finite(tensor_bytes(values[expected]), name))
            self.assertFalse(_all_finite(data, name))

    def test_float32_encodings_and_nonfinite_chunk_boundaries(self):
        generator = torch.Generator().manual_seed(42)
        words = torch.cat(
            (
                torch.tensor(
                    [
                        0,
                        1,
                        0x007FFFFF,
                        0x00800000,
                        0x7F7FFFFF,
                        0x7F800000,
                        0x7F800001,
                        0x7FC00000,
                        0x7FFFFFFF,
                        0x80000000,
                        0x80000001,
                        0xFF7FFFFF,
                        0xFF800000,
                        0xFF800001,
                        0xFFC00000,
                        0xFFFFFFFF,
                    ],
                    dtype=torch.int64,
                ),
                torch.randint(0, 1 << 32, (8192,), generator=generator),
            )
        ).to(torch.int32)
        values = words.view(torch.float32)
        data = tensor_bytes(values)
        for index, finite in enumerate(torch.isfinite(values).tolist()):
            self.assertEqual(
                _all_finite(data[index * 4 : (index + 1) * 4], "float32"),
                finite,
                hex(int(words[index]) & 0xFFFFFFFF),
            )
        size = protocol._FINITE_CHUNK_ELEMENTS
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            values = torch.ones(size * 2 + 3, dtype=dtype)
            original = values.clone()
            name = str(dtype).removeprefix("torch.")
            with patch.object(np, "bitwise_and", wraps=np.bitwise_and) as scan:
                self.assertTrue(_all_finite(tensor_bytes(values), name))
            self.assertTrue(torch.equal(values, original))
            self.assertEqual(
                sum(call.args[0].size for call in scan.call_args_list), values.numel()
            )
            self.assertLessEqual(
                max(call.args[0].size for call in scan.call_args_list), size
            )
            for index in (0, size - 1, size, size * 2, values.numel() - 1):
                for nonfinite in (float("inf"), float("-inf"), float("nan")):
                    values[index] = nonfinite
                    self.assertFalse(_all_finite(tensor_bytes(values), name))
                values[index] = 1

    def test_nonfinite_kv_and_teacher_rejected_with_valid_checksums(self):
        for kv_dtype in ("bfloat16", "float16"):
            for name in (
                "target_k",
                "target_v",
                "teacher_topk_logits",
                "teacher_logsumexp",
            ):
                for nonfinite in (float("inf"), float("-inf"), float("nan")):
                    with self.subTest(dtype=kv_dtype, field=name, value=nonfinite):
                        manifest, tensors = make_snapshot()
                        if kv_dtype == "float16":
                            objects = []
                            for obj in manifest.objects:
                                if obj.kind == "kv":
                                    tensors[obj.key] = tensors[obj.key].half()
                                    obj = msgspec.structs.replace(
                                        obj,
                                        dtype=kv_dtype,
                                        sha256=digest_bytes(
                                            tensor_bytes(tensors[obj.key])
                                        ),
                                    )
                                objects.append(obj)
                            manifest = msgspec.structs.replace(
                                manifest,
                                objects=objects,
                                kv=msgspec.structs.replace(
                                    manifest.kv,
                                    dtype=kv_dtype,
                                    codec="dense_fp16_post_rope_v1",
                                ),
                            )
                        validate_tensors(manifest, tensors)
                        obj = next(
                            o
                            for o in manifest.objects
                            if o.name == name or o.name.startswith(name + ".")
                        )
                        tensors[obj.key].view(-1)[-1] = nonfinite
                        replacement = msgspec.structs.replace(
                            obj, sha256=digest_bytes(tensor_bytes(tensors[obj.key]))
                        )
                        manifest = msgspec.structs.replace(
                            manifest,
                            objects=[
                                replacement if o.key == obj.key else o
                                for o in manifest.objects
                            ],
                        )
                        with self.assertRaisesRegex(ContractError, "nonfinite"):
                            validate_tensors(manifest, tensors)


if __name__ == "__main__":
    unittest.main()
