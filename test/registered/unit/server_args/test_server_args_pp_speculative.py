"""Unit tests for the pipeline-parallelism + speculative-decoding gate.

The gate is two layers and this file pins both:

* ``arg_groups/validation_hook.check_server_args`` -- model-agnostic policy:
  ``pp_size > 1`` requires ``disable_overlap_schedule``, and if a speculative
  algorithm is set it additionally requires the PD-disaggregated *prefill* role.
* the per-algorithm hooks in ``arg_groups/speculative_hook`` -- each narrows the
  policy to what its own draft path can actually do. Only DSpark takes the
  prefill exception; the eagle family and ngram still require ``pp_size == 1``.

The two layers need different drivers: ``model_path="dummy"`` short-circuits
``run_resolution_pipeline`` (pipeline.py, right after the hardware validation),
so a dummy-model record reaches ``check_server_args`` but never reaches the
per-algorithm hooks. The second class therefore calls
``handle_speculative_decoding`` directly, the same way
``unit/spec/test_spec_cpu_overlap_constraint.py`` does.

Why a unit test and not only an e2e one: the policy is a matrix (pp x algorithm
x disaggregation role x overlap schedule) that costs nothing to check on CPU,
while the e2e test can only afford one point of it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPipelineParallelSpeculativeGate(CustomTestCase):
    """``check_server_args`` accepts PP + spec only on the PD prefill role."""

    @classmethod
    def setUpClass(cls):
        # CPU-only runners have no CUDA device; the gate under test does not
        # care, but resolution upstream of it does.
        cls._device_patch = patch(
            "sglang.srt.arg_groups.serving_hook.get_device", return_value="cuda"
        )
        cls._device_patch.start()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_device_patch"):
            cls._device_patch.stop()

    @staticmethod
    def _validate(**overrides):
        """Run the full arg validation on an otherwise minimal ServerArgs.

        ``served_model_name``, ``chunked_prefill_size`` and ``page_size`` are
        normally derived from the model config and the device; with a dummy
        model path they stay ``None`` and trip unrelated checks *after* the gate
        (a colon test on the served name, a divisibility test on the chunk
        size). Pinning them keeps a failure here attributable to the gate.
        """
        args = ServerArgs(
            model_path="dummy",
            served_model_name="dummy",
            chunked_prefill_size=8192,
            page_size=1,
            **overrides,
        )
        args.resolve_once()
        args.check_server_args()

    # ---- accepted ----------------------------------------------------------

    def test_prefill_role_accepts_pp_with_speculative_decoding(self):
        self._validate(
            pp_size=2,
            speculative_algorithm="NGRAM",
            disaggregation_mode="prefill",
            disable_overlap_schedule=True,
        )

    def test_pp_size_one_is_unaffected(self):
        # The default, non-pipelined path must not have become stricter.
        self._validate(pp_size=1, speculative_algorithm="NGRAM")

    def test_pp_without_speculative_decoding_is_unaffected(self):
        self._validate(pp_size=2, disable_overlap_schedule=True)

    # ---- rejected ----------------------------------------------------------

    def test_decode_role_still_rejects_pp_with_speculative_decoding(self):
        with self.assertRaisesRegex(AssertionError, "prefill role"):
            self._validate(
                pp_size=2,
                speculative_algorithm="NGRAM",
                disaggregation_mode="decode",
                disable_overlap_schedule=True,
            )

    def test_non_disaggregated_still_rejects_pp_with_speculative_decoding(self):
        with self.assertRaisesRegex(AssertionError, "prefill role"):
            self._validate(
                pp_size=2,
                speculative_algorithm="NGRAM",
                disaggregation_mode="null",
                disable_overlap_schedule=True,
            )

    def test_overlap_schedule_requirement_survives_the_split(self):
        # The overlap-schedule and speculative-decoding halves used to be one
        # assertion. Splitting them must not have dropped the overlap half.
        with self.assertRaisesRegex(AssertionError, "overlap schedule"):
            self._validate(
                pp_size=2,
                speculative_algorithm="NGRAM",
                disaggregation_mode="prefill",
                disable_overlap_schedule=False,
            )

    def test_overlap_schedule_requirement_holds_without_speculative_decoding(self):
        with self.assertRaisesRegex(AssertionError, "overlap schedule"):
            self._validate(pp_size=2, disable_overlap_schedule=False)

    # ---- algorithm-agnostic ------------------------------------------------

    def test_policy_layer_is_not_specific_to_one_algorithm(self):
        # This layer keys off `speculative_algorithm is not None`, nothing more;
        # which algorithms can actually use the prefill exception is decided one
        # layer down, in TestPipelineParallelSpeculativePerAlgorithmGate.
        for algorithm in ("NGRAM", "EAGLE", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                self._validate(
                    pp_size=2,
                    speculative_algorithm=algorithm,
                    disaggregation_mode="prefill",
                    disable_overlap_schedule=True,
                )
                with self.assertRaisesRegex(AssertionError, "prefill role"):
                    self._validate(
                        pp_size=2,
                        speculative_algorithm=algorithm,
                        disaggregation_mode="decode",
                        disable_overlap_schedule=True,
                    )


class TestPipelineParallelSpeculativePerAlgorithmGate(CustomTestCase):
    """Only DSpark takes the PD-prefill exception to ``pp_size == 1``."""

    @staticmethod
    def _hook_error(algorithm, **overrides):
        """Return the message the per-algorithm hook raises, or None.

        The stand-in ``_model_config`` is enough for the ``pp_size`` guards,
        which all run before any real model introspection; a hook that gets past
        its guard may still trip on an unrelated check further down, so callers
        assert on *which* message came back rather than on raising at all.
        """
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = algorithm
        args.device = "cuda"
        args.speculative_num_steps = 3
        args.speculative_eagle_topk = 1
        args.speculative_num_draft_tokens = 4
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"],
                get_text_config=lambda: SimpleNamespace(),
            )
        )
        for key, value in overrides.items():
            setattr(args, key, value)

        try:
            handle_speculative_decoding(args)
        except ValueError as error:
            return str(error)
        return None

    def _assert_rejects_pp(self, algorithm, **overrides):
        message = self._hook_error(algorithm, pp_size=2, **overrides)
        self.assertIsNotNone(
            message, f"{algorithm} accepted pp_size=2 at the hook layer"
        )
        self.assertIn("pp_size == 1", message)

    def _assert_accepts_pp(self, algorithm, **overrides):
        # Equivalence rather than "did not raise": with a stand-in model config
        # the hook trips on a later, unrelated check either way, so the claim
        # under test is that pp_size=2 gets the *same* outcome as pp_size=1.
        self.assertEqual(
            self._hook_error(algorithm, pp_size=2, **overrides),
            self._hook_error(algorithm, pp_size=1, **overrides),
        )

    def test_dspark_accepts_pp_on_the_prefill_role(self):
        self._assert_accepts_pp("DSPARK", disaggregation_mode="prefill")

    def test_dspark_rejects_pp_on_the_decode_role(self):
        self._assert_rejects_pp("DSPARK", disaggregation_mode="decode")

    def test_dspark_rejects_pp_without_disaggregation(self):
        self._assert_rejects_pp("DSPARK", disaggregation_mode="null")

    def test_eagle_rejects_pp_even_on_the_prefill_role(self):
        # The draft head shares the target's embed_tokens / lm_head, which a
        # pipeline-split target does not hold on any single stage.
        for mode in ("prefill", "decode", "null"):
            with self.subTest(disaggregation_mode=mode):
                self._assert_rejects_pp("EAGLE", disaggregation_mode=mode)

    def test_ngram_rejects_pp_even_on_the_prefill_role(self):
        # The pipeline relay of the draft tensors expects an EagleDraftInput.
        for mode in ("prefill", "decode", "null"):
            with self.subTest(disaggregation_mode=mode):
                self._assert_rejects_pp("NGRAM", disaggregation_mode=mode)

    def test_dflash_pp_rejection_is_untouched(self):
        # Pre-existing guard; here so relaxing the policy layer cannot silently
        # widen an algorithm that was never validated for pipeline parallelism.
        self._assert_rejects_pp("DFLASH", disaggregation_mode="prefill")


if __name__ == "__main__":
    unittest.main()
