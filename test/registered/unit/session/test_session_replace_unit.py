"""Regression tests for replacing every request in a non-streaming session."""

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionReqNode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSessionReplace(CustomTestCase):
    def test_replace_without_rid_clears_each_branch(self):
        session = Session(capacity_of_str_len=1000, session_id="s")

        def old_node(rid, parent=None):
            req = SimpleNamespace(rid=rid, finished_reason=None, to_finish=None)
            node = SessionReqNode(req, parent=parent)
            session.req_nodes[rid] = node
            return node

        root = old_node("root")
        child = old_node("child", parent=root)
        second_root = old_node("second_root")
        fields = dict.fromkeys(
            (
                "lora_id",
                "custom_logit_processor",
                "token_ids_logprob",
                "bootstrap_host",
                "bootstrap_port",
                "bootstrap_room",
                "routed_dp_rank",
                "disagg_prefill_dp_rank",
                "priority",
                "routing_key",
                "extra_key",
                "cache_salt",
                "http_worker_ipc",
                "time_stats",
            )
        )
        recv = SimpleNamespace(
            **fields,
            rid="replacement",
            input_ids=array("q", [1, 2]),
            session_params=SimpleNamespace(replace=True, rid=None),
            sampling_params=SamplingParams(max_new_tokens=1),
            stream=False,
            return_logprob=False,
            top_logprobs_num=0,
            return_sampling_mask=False,
            sampling_logprobs_mode="selected",
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
        )

        new_req = session.create_req(recv, tokenizer=None, vocab_size=1024)

        self.assertEqual(new_req.rid, "replacement")
        self.assertEqual(list(session.req_nodes), ["replacement"])
        for node in (root, child, second_root):
            self.assertIsNotNone(node.req.to_finish)


if __name__ == "__main__":
    unittest.main()
