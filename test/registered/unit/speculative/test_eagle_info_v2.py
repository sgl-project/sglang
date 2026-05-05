"""Unit tests for srt/speculative/eagle_info_v2.py — no server, no model loading."""

import ast
import unittest
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[4]
TARGET_FILE = REPO_ROOT / "python/sglang/srt/speculative/eagle_info_v2.py"
EXPECTED_LOGITS = torch.tensor([[10.0, 11.0, 12.0]])


class _ForwardModeStub:
    def __init__(self, idle=False):
        self._idle = idle

    def is_idle(self):
        return self._idle


class _DummyPenalizer:
    is_required = False

    def apply(self, logits):
        return None

    def filter(self, keep_indices_device):
        return None


class _DummySamplingInfo:
    def __init__(self):
        self.is_all_greedy = True
        self.has_custom_logit_processor = True
        self.penalizer_orchestrator = _DummyPenalizer()
        self.acc_additive_penalties = None
        self.acc_scaling_penalties = None
        self.logit_bias = None
        self.temperatures = torch.ones((1, 1), dtype=torch.float32)
        self.top_ks = torch.ones((1,), dtype=torch.int32)
        self.top_ps = torch.ones((1,), dtype=torch.float32)

    def __len__(self):
        return 1

    def apply_logits_bias(self, logits):
        return None

    def filter_batch(self, keep_indices, keep_indices_device):
        return None


def _extract_method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return dedent(ast.get_source_segment(source, item))
    raise ValueError(f"Could not find {class_name}.{method_name} in {path}")


def _build_method(path: Path, class_name: str, method_name: str, namespace: dict):
    method_src = _extract_method_source(path, class_name, method_name)
    exec(method_src, namespace)
    return namespace[method_name]


def _stub_verify_tree_greedy_func(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    target_predict,
    topk,
    retrive_index=None,
    retrive_next_token=None,
    retrive_next_sibling=None,
    retrieve_index=None,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
):
    retrive_index = retrive_index if retrive_index is not None else retrieve_index
    retrive_next_token = (
        retrive_next_token if retrive_next_token is not None else retrieve_next_token
    )
    retrive_next_sibling = (
        retrive_next_sibling
        if retrive_next_sibling is not None
        else retrieve_next_sibling
    )
    assert retrive_index is not None
    assert retrive_next_token is not None
    assert retrive_next_sibling is not None
    predicts.copy_(target_predict.flatten())
    accept_index.fill_(-1)
    accept_index[:, 0] = 0
    accept_token_num.fill_(1)
    return predicts, accept_index, accept_token_num


def _make_logits_output():
    return SimpleNamespace(
        next_token_logits=torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    )


def _make_batch():
    return SimpleNamespace(
        forward_mode=_ForwardModeStub(idle=False),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        input_ids=torch.tensor([7], dtype=torch.int64),
        sampling_info=_DummySamplingInfo(),
        device="cpu",
    )


def _make_verify_input():
    return SimpleNamespace(
        draft_token=torch.tensor([1], dtype=torch.int32),
        draft_token_num=1,
        retrive_index=torch.tensor([[0]], dtype=torch.int32),
        retrive_next_token=torch.tensor([[0]], dtype=torch.int32),
        retrive_next_sibling=torch.tensor([[-1]], dtype=torch.int32),
        retrieve_index=torch.tensor([[0]], dtype=torch.int32),
        retrieve_next_token=torch.tensor([[0]], dtype=torch.int32),
        retrieve_next_sibling=torch.tensor([[-1]], dtype=torch.int32),
        spec_steps=0,
        topk=1,
        grammar=None,
    )


class TestEagleVerifyV2(CustomTestCase):
    def test_spec_v2_verify_applies_custom_logit_processor(self):
        call_log = []

        def spy_apply_custom_logit_processor(
            logits, sampling_info, num_tokens_in_batch=1
        ):
            call_log.append(num_tokens_in_batch)
            logits.add_(10.0)

        namespace = {
            "torch": torch,
            "F": torch.nn.functional,
            "EagleVerifyInput": object,
            "ModelWorkerBatch": object,
            "LogitsProcessorOutput": object,
            "apply_custom_logit_processor": spy_apply_custom_logit_processor,
            "_is_npu": False,
            "_is_hip": False,
            "verify_tree_greedy_func": _stub_verify_tree_greedy_func,
            "SIMULATE_ACC_LEN": 0,
            "generate_simulated_accept_index": lambda **kwargs: kwargs["accept_index"],
            "get_global_server_args": lambda: SimpleNamespace(
                speculative_accept_threshold_single=0.0,
                speculative_accept_threshold_acc=0.0,
            ),
        }

        v2_sample = _build_method(
            TARGET_FILE,
            "EagleVerifyInputV2Mixin",
            "sample",
            namespace,
        )

        v2_input = _make_verify_input()
        v2_batch = _make_batch()
        v2_logits = _make_logits_output()
        v2_sample(v2_input, v2_batch, v2_logits)

        self.assertEqual(call_log, [1])
        self.assertTrue(torch.equal(v2_logits.next_token_logits, EXPECTED_LOGITS))


if __name__ == "__main__":
    unittest.main()
