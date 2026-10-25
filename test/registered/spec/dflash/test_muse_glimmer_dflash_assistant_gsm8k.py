import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="nightly", runner_config="1-gpu-large")

TARGET_MODEL = "meta-models/Muse-Glimmer-30B"
DRAFT_MODEL = "meta-models/Muse-Glimmer-30B-assistant"


class TestMuseGlimmerDflashAssistantGSM8K(CustomTestCase, GSM8KMixin):
    """GSM8K + DFlash accept-length regression test for the native
    MuseGlimmerAssistantModel draft (``meta-models/Muse-Glimmer-30B-assistant``).

    This checkpoint loads through sglang's own native ``models/dflash.py`` /
    ``configs/muse_glimmer.py::MuseGlimmerAssistantConfig`` with no extra wheel
    dependency. Integrating it surfaced two bugs that fail *silently*, never raise, and do not move
    GSM8K accuracy at temperature 0 (speculative decoding always falls back
    to the target's own correct token on a draft miss, so a broken draft
    still produces exactly the target's answers -- just slower):

    1. The vendor's weight names (``encoder.fc.weight`` /
       ``encoder.output_norm_enc.weight``) didn't match what
       ``DFlashDraftModel.load_weights`` expected (``fc.weight`` /
       ``hidden_norm.weight``), so those two tensors silently stayed at
       random init.
    2. The vendor's ``target_layer_ids`` are in the HF "output of layer k"
       convention; ``models/muse_glimmer.py::set_dflash_layers_to_capture``
       uses ids as-is (Muse Glimmer's own draft configs carry llama.cpp's
       layer-*input* convention), so every captured layer was off by one.

    Both together collapsed real (non-simulated) accept_length to ~1.00 at
    ``--speculative-dflash-block-size 5`` -- effectively no speculation, only
    the mandatory bonus token -- with GSM8K accuracy unaffected throughout.
    ``gsm8k_accept_length_thres`` is therefore the actual regression guard
    here; the accuracy threshold alone would not have caught this.

    Measured after both fixes, same block size, same target: median
    accept_length 3.12 (this draft) vs 3.09 (our own GGUF-converted draft,
    the previous ground truth) across batch sizes 1/4/8 at 512in/256out --
    statistically indistinguishable. 2.5 leaves headroom below that for
    workload variance while sitting well above the ~1.0 broken-state value.
    """

    model = TARGET_MODEL
    gsm8k_backend = (
        "sgl_eval"  # chat completions API, not /generate or raw /completions
    )
    gsm8k_score_threshold = 0.85
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 2.5

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--reasoning-parser",
                "muse",
                "--tool-call-parser",
                "muse",
                "--language-model-only",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--speculative-draft-load-format",
                "auto",
                "--speculative-dflash-block-size",
                "5",
                "--mem-fraction-static",
                "0.85",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
