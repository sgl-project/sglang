"""Validate position coverage and reject structurally plausible corrupt samples."""

import copy
import json
import unittest
from pathlib import Path

import jsonschema
import msgspec
from sglang.srt.training_capture.protocol import (
    ContractError,
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
            / "mooncake-study/training-data-contract/manifest.schema.json"
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


if __name__ == "__main__":
    unittest.main()
