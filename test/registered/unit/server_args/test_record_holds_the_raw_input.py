"""After resolution the record still holds exactly what the caller passed.

Two directions: a field must not be *rebound* (the snapshot holds the value the
caller passed, and `getattr` must still answer with it), and an
operator-supplied mutable must not be *edited in place* -- the record points at
the caller's own dict or list, so a handler that reaches into one changes a
value the caller still holds and the snapshot cannot see it.
"""

import copy
import dataclasses
import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 128,
    "intermediate_size": 256,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
    "vocab_size": 1000,
}

# One shape per family of handlers that decides something.
_SHAPES = {
    "plain": {},
    "data_parallel": {"dp_size": 2},
    "tensor_parallel": {"tp_size": 2},
    "speculative": {"speculative_algorithm": "EAGLE"},
    "speculative_mtp": {"speculative_algorithm": "NEXTN"},
    "cuda_graph_knobs": {"cuda_graph_max_bs_decode": 16, "page_size": 32},
    "explicit_graph_json": {"cuda_graph_config": {"decode": {"max_bs": 12}}},
    "attention_backend": {"attention_backend": "triton"},
    "lora": {"lora_paths": ["adapter=/tmp/does-not-need-to-exist"]},
    "quantization": {"quantization": "fp8"},
    "disaggregation": {"disaggregation_mode": "prefill"},
    "deterministic": {"enable_deterministic_inference": True},
    "hierarchical_cache": {"enable_hierarchical_cache": True},
    "kv_events": {"kv_events_config": '{"publisher":"zmq"}'},
    "chunked_prefill": {"chunked_prefill_size": 1024},
}


class TestRecordHoldsTheRawInput(CustomTestCase):
    def setUp(self):
        # Resolution writes environment variables, which outlive the record.
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)

    def _model_path(self):
        path = tempfile.mkdtemp(prefix="raw_input_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        return path

    def _resolve(self, **supplied):
        """A fully-resolved record: a real config.json, so the pipeline runs
        past its dummy-model early return."""
        server_args = ServerArgs(
            model_path=self._model_path(),
            device="cuda",
            random_seed=42,
            **supplied,
        )
        server_args.resolve_once()
        server_args.check_server_args()
        return server_args

    def test_no_field_moves_from_what_the_caller_passed(self):
        for name, supplied in _SHAPES.items():
            with self.subTest(shape=name):
                server_args = self._resolve(**supplied)
                raw = server_args._raw_input

                def _moved(current, original):
                    if current is original:
                        return False
                    if isinstance(original, (list, dict, set, bytearray)) or isinstance(
                        current, (list, dict, set, bytearray)
                    ):
                        # A mutable is only unmoved when it is the *same*
                        # object: an equal copy no longer shares with the caller.
                        return True
                    # Equal ints and strings are not always the same object.
                    return current != original

                moved = {
                    field.name: (raw[field.name], getattr(server_args, field.name))
                    for field in dataclasses.fields(server_args)
                    if _moved(getattr(server_args, field.name), raw[field.name])
                }
                self.assertEqual(
                    {},
                    moved,
                    f"resolution moved these fields on the {name} shape, so the "
                    "record no longer answers with the operator's input and a "
                    "reader that takes a decision off it disagrees with the bags: "
                    f"{moved}",
                )

    def test_the_snapshot_is_the_value_the_caller_passed(self):
        paths = ["adapter=/tmp/does-not-need-to-exist"]
        supplied = {"lora_paths": paths, "cuda_graph_max_bs_decode": 16}
        expected = copy.deepcopy(supplied)
        server_args = self._resolve(**supplied)
        for field, value in expected.items():
            self.assertEqual(
                value,
                server_args._raw_input[field],
                f"the snapshot of {field} is not what the caller passed, so "
                "every comparison against it is vacuous",
            )

    def test_an_operator_supplied_mutable_is_not_edited_in_place(self):
        supplied = {
            "cuda_graph_config": {"decode": {"max_bs": 7}},
            "lora_paths": ["adapter=/tmp/does-not-need-to-exist"],
        }
        before = copy.deepcopy(supplied)
        server_args = self._resolve(**supplied)

        for field, value in before.items():
            self.assertEqual(
                value,
                supplied[field],
                f"resolution edited the {field} object the caller still holds; "
                "the raw-input snapshot stores the reference, so a field-by-field "
                "comparison cannot see this",
            )
        self.assertIs(
            supplied["cuda_graph_config"],
            server_args.cuda_graph_config,
            "the record stopped pointing at the caller's object, so the reads "
            "above are no longer testing what the caller can observe",
        )


if __name__ == "__main__":
    unittest.main()
