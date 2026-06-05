import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestPDBase


class TestChunkedFeaturePDPP(ChunkedTestPDBase):
    # PD disaggregation with pipeline parallelism on the prefill side. The shared
    # base already forces chunked prefill (chunked_prefill_size=256), so each
    # prompt's per-chunk KV is sent across the real prefill -> decode boundary while
    # pipelined across the prefill's PP micro-batches. Decode TP matches prefill TP
    # so the transferred KV layout lines up; decode sits on base-gpu-id 4 since the
    # prefill tp2 x pp2 occupies GPUs 0-3.
    gsm8k_threshold = 0.50
    feature_args = [
        "--tp-size",
        "2",
        "--pp-size",
        "2",
        "--disable-overlap-schedule",
    ]
    decode_feature_args = [
        "--tp-size",
        "2",
        "--base-gpu-id",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
