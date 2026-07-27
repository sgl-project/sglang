"""Positional-ABI tripwire for the embedded Rust server's scheduler wire.

`rust/sglang-server`'s `GenerateRequest::to_header_msgpack` hand-writes
`TokenizedGenerateReqInput` as a positional tagged msgpack array (msgspec
`array_like=True` layout: `[tag, *fields]`), ending at `disagg_prefill_dp_rank`
so the PD bootstrap block rides the wire. msgspec resolves fields by position,
so a reorder of / insertion into the first 30 fields of the Python struct would
silently corrupt every rust-ingress request rather than fail loudly — exactly
how `return_hidden_states` once landed in `return_sampling_mask`.

Pins, in black-box terms:
1. the field-order prefix the Rust encoder targets (bookkeeping: someone
   editing `io_struct.py` must consciously touch the Rust encoder too), and
2. the two decode properties the encoder relies on, through the same
   `msgpack_decode_explained` path `RustServer.drain()` uses: a 31-element
   array lands the PD fields on the right slots, and a short array (ending at
   `stream`, the last non-defaulted field) decodes with defaulted trailing
   fields.
"""

import unittest

import msgspec

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.utils import msgpack_decode_explained
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-c-test-cpu")

# The exact prefix `to_header_msgpack` (rust/sglang-server/src/message.rs)
# emits, in wire order. Array index = field index + 1 (idx 0 is the tag).
RUST_HEADER_FIELDS = (
    "rid",
    "http_worker_ipc",
    "input_text",
    "input_ids",
    "input_embeds",
    "mm_inputs",
    "token_type_ids",
    "sampling_params",
    "return_logprob",
    "logprob_start_len",
    "top_logprobs_num",
    "token_ids_logprob",
    "stream",
    "return_sampling_mask",
    "return_hidden_states",
    "return_routed_experts",
    "routed_experts_start_len",
    "return_indexer_topk",
    "session_id",
    "session_params",
    "lora_id",
    "custom_logit_processor",
    "positional_embed_overrides",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "bootstrap_pair_key",
    "decode_tp_size",
    "routed_dp_rank",
    "disagg_prefill_dp_rank",
)


def _rust_header(elements: list) -> bytes:
    """Encode a hand-built positional array — byte-shape-identical to what the
    Rust encoder produces (both are plain msgpack arrays)."""
    return msgspec.msgpack.encode(elements)


class TestRustHeaderAbi(CustomTestCase):
    def test_struct_prefix_matches_rust_encoder(self):
        self.assertEqual(
            TokenizedGenerateReqInput.__struct_fields__[: len(RUST_HEADER_FIELDS)],
            RUST_HEADER_FIELDS,
            "TokenizedGenerateReqInput's leading fields moved: the Rust "
            "encoder writes these slots positionally — update "
            "rust/sglang-server/src/message.rs::to_header_msgpack in the same "
            "change or every rust-ingress request is silently corrupted.",
        )

    def test_full_rust_array_lands_pd_fields(self):
        # Mirrors a PD /generate request as the Rust server emits it: tag,
        # scalars resolved to wire defaults, PD block at idx 24-30.
        header = _rust_header(
            [
                "TokenizedGenerateReqInput",  # 0  tag
                "rid-1",  # 1
                None,  # 2  http_worker_ipc
                "hello",  # 3  input_text
                None,  # 4  input_ids (rides columnar)
                None,  # 5  input_embeds
                None,  # 6  mm_inputs
                None,  # 7  token_type_ids
                {"temperature": 0.0, "max_new_tokens": 8},  # 8
                False,  # 9  return_logprob
                -1,  # 10 logprob_start_len
                0,  # 11 top_logprobs_num
                None,  # 12 token_ids_logprob
                True,  # 13 stream
                False,  # 14 return_sampling_mask
                True,  # 15 return_hidden_states
                False,  # 16 return_routed_experts
                0,  # 17 routed_experts_start_len
                False,  # 18 return_indexer_topk
                None,  # 19 session_id
                None,  # 20 session_params
                None,  # 21 lora_id
                None,  # 22 custom_logit_processor
                None,  # 23 positional_embed_overrides
                "10.0.0.1",  # 24 bootstrap_host
                8998,  # 25 bootstrap_port
                2**63 - 1,  # 26 bootstrap_room (mini_lb's max)
                "pk",  # 27 bootstrap_pair_key
                2,  # 28 decode_tp_size
                1,  # 29 routed_dp_rank
                0,  # 30 disagg_prefill_dp_rank
            ]
        )
        req = msgpack_decode_explained(header)
        self.assertIsInstance(req, TokenizedGenerateReqInput)
        self.assertEqual(req.rid, "rid-1")
        self.assertTrue(req.stream)
        self.assertFalse(req.return_sampling_mask)
        self.assertTrue(req.return_hidden_states)
        self.assertEqual(req.bootstrap_host, "10.0.0.1")
        self.assertEqual(req.bootstrap_port, 8998)
        self.assertEqual(req.bootstrap_room, 2**63 - 1)
        self.assertEqual(req.bootstrap_pair_key, "pk")
        self.assertEqual(req.decode_tp_size, 2)
        self.assertEqual(req.routed_dp_rank, 1)
        self.assertEqual(req.disagg_prefill_dp_rank, 0)

    def test_short_array_decodes_with_defaulted_tail(self):
        # The Rust wire may end before the trailing defaulted fields (it did,
        # at `stream`, before the PD block existed). msgspec must fill the rest
        # from defaults — the property that lets the encoder stop at
        # `disagg_prefill_dp_rank` and ignore later Python-side additions.
        header = _rust_header(
            [
                "TokenizedGenerateReqInput",
                "rid-2",
                None,
                None,
                None,
                None,
                None,
                None,
                {},
                False,
                -1,
                0,
                None,
                False,  # 13 stream — last non-defaulted field
            ]
        )
        req = msgpack_decode_explained(header)
        self.assertIsInstance(req, TokenizedGenerateReqInput)
        self.assertFalse(req.return_sampling_mask)
        self.assertIsNone(req.bootstrap_host)
        self.assertIsNone(req.bootstrap_room)
        self.assertIsNone(req.disagg_prefill_dp_rank)


if __name__ == "__main__":
    unittest.main()
