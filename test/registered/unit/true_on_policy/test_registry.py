import unittest

from sglang.srt.true_on_policy.registry import (
    EVERY,
    EVERY_SPEC,
    Cell,
    Family,
    ServingEnvelope,
    ServingLayout,
    Spec,
    admit,
    at_least,
    find_family,
    only,
    refused,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layout(**kwargs):
    fields = dict(ep=1, dp_attention=1, pp=1, spec=None)
    fields.update(kwargs)
    return ServingLayout(**fields)


def _envelope(**kwargs):
    fields = dict(ep=EVERY, dp_attention=EVERY, pp=EVERY, spec=EVERY_SPEC)
    fields.update(kwargs)
    return ServingEnvelope(**fields)


def _admit(layout, envelopes):
    family = Family(name="Toy", model_types=("toy",), ops=tuple(envelopes))
    return admit(
        "toy",
        layout,
        fp8=False,
        side="serving",
        envelopes_of=lambda name: envelopes[name],
        families=(family,),
    )


class TestAdmission(unittest.TestCase):
    def test_any_refusing_op_refuses_the_family_layout(self):
        envelopes = {
            "attention": (_envelope(),),
            "recurrent": (_envelope(pp=refused("state is not handed across stages")),),
        }
        _admit(_layout(), envelopes)
        with self.assertRaises(NotImplementedError) as refusal:
            _admit(_layout(pp=2), envelopes)
        message = str(refusal.exception)
        self.assertIn("op recurrent: pp=2 refused: state is not handed", message)
        self.assertNotIn("op attention", message)

    def test_envelopes_state_cross_axis_rules(self):
        """Two envelopes admit (EP1, any PP) or (EP>=2, PP1), never a per-axis mix of both."""
        envelopes = {
            "experts": (
                _envelope(ep=only(1, why="EP needs PP1")),
                _envelope(
                    ep=at_least(2, why="EP runs behind the slot transport"),
                    pp=refused("EP needs PP1"),
                ),
            )
        }
        _admit(_layout(pp=2), envelopes)
        _admit(_layout(ep=4), envelopes)
        with self.assertRaises(NotImplementedError):
            _admit(_layout(ep=4, pp=2), envelopes)

    def test_spec_patterns(self):
        chain = Spec(recipes=((None, 1, None),), refused="chain verification only")
        envelopes = {"recurrent": (_envelope(spec=chain),)}
        _admit(_layout(), envelopes)
        _admit(_layout(spec=(3, 1, 4)), envelopes)
        with self.assertRaisesRegex(NotImplementedError, "chain verification only"):
            _admit(_layout(spec=(3, 2, 4)), envelopes)

    def test_a_restricting_cell_needs_a_reason(self):
        for restricting in (dict(sizes=(1,)), dict(at_least=2)):
            with self.subTest(**restricting), self.assertRaises(ValueError):
                Cell(**restricting)
        with self.assertRaises(ValueError):
            Spec(recipes=((None, 1, None),))

    def test_precision_selects_the_ops_not_their_cells(self):
        envelopes = {
            "experts": (_envelope(ep=only(2, why="needs DeepEP")),),
            "local_experts": (_envelope(),),
        }
        family = Family(
            name="Toy",
            model_types=("toy",),
            ops=("experts",),
            fp8_ops=("local_experts",),
        )

        def run(*, fp8):
            return admit(
                "toy",
                _layout(ep=1),
                fp8=fp8,
                side="serving",
                envelopes_of=lambda name: envelopes[name],
                families=(family,),
            )

        run(fp8=True)
        with self.assertRaisesRegex(NotImplementedError, "op experts: ep=1"):
            run(fp8=False)

    def test_a_family_without_fp8_ops_refuses_an_fp8_checkpoint(self):
        family = Family(name="Toy", model_types=("toy",), ops=("experts",))
        self.assertEqual(family.ops_for(fp8=False), ("experts",))
        with self.assertRaisesRegex(
            NotImplementedError, "Toy has no block-FP8 program"
        ):
            admit(
                "toy",
                _layout(),
                fp8=True,
                side="serving",
                envelopes_of=lambda name: (_envelope(),),
                families=(family,),
            )

    def test_deepseek_v4_block_fp8_admits_its_fp8_program(self):
        """DS4 is block-FP8 only: its ops are the FP8 program, declared as fp8_ops too, so an FP8 checkpoint does not
        depend on falling back to the default ops."""
        from sglang.srt.true_on_policy.registry import OPS_BY_NAME

        family = find_family("deepseek_v4")
        self.assertIsNotNone(family.fp8_ops)
        self.assertEqual(family.ops_for(fp8=True), family.ops)
        for spec in (None, (3, 1, 4)):
            admitted = admit(
                "deepseek_v4",
                _layout(ep=4, dp_attention=4, spec=spec),
                fp8=True,
                side="serving",
                envelopes_of=lambda name: OPS_BY_NAME[name].serving,
            )
            self.assertEqual(admitted.name, "DeepSeek-V4")

    def test_glm5_3_flash_speculates_with_its_mtp_layer_as_a_chain(self):
        """Every GLM-5.3-Flash op admits its MTP draft's recipes; KDA and DSA verify chains only, so a tree is refused,
        as is a depth the draft op has not qualified."""
        from sglang.srt.true_on_policy.registry import OPS_BY_NAME

        def run(spec):
            return admit(
                "glm5_next",
                _layout(ep=4, dp_attention=4, spec=spec),
                fp8=False,
                side="serving",
                envelopes_of=lambda name: OPS_BY_NAME[name].serving,
            )

        for spec in ((1, 1, 2), (5, 1, 6)):
            with self.subTest(spec=spec):
                self.assertEqual(run(spec).name, "GLM-5.3-Flash")
        with self.assertRaisesRegex(NotImplementedError, "op kda: spec=.*chain"):
            run((5, 4, 8))
        with self.assertRaisesRegex(NotImplementedError, "op glm5_next_mtp_draft"):
            run((3, 1, 4))

    def test_a_model_outside_every_family_is_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "no TOP family admits"):
            find_family("not_a_top_model")

    def test_images_are_admitted_where_the_family_has_a_vision_program(self):
        """The vision ops join the text ops for a run with images: the Qwen GDN families and DeepSeek-V4.1 serve images
        on every layout they serve text on, and a family without a vision program refuses images outright."""
        from sglang.srt.true_on_policy.registry import OPS_BY_NAME

        def run(model_type, *, vision, **layout):
            return admit(
                model_type,
                _layout(**layout),
                fp8=False,
                side="serving",
                envelopes_of=lambda name: OPS_BY_NAME[name].serving,
                vision=vision,
            )

        dense = find_family("qwen3_5")
        self.assertEqual(
            dense.ops_for(fp8=False, vision=True), dense.ops + ("qwen_vision_tower",)
        )
        for model_type, layout in (
            ("qwen3_5", dict()),
            ("qwen3_5", dict(dp_attention=2)),
            ("qwen3_5_moe", dict(ep=4, dp_attention=4, spec=(3, 1, 4))),
            ("qwen4_exp", dict(ep=4, dp_attention=4, spec=(3, 1, 4))),
            ("deepseek_v41", dict(ep=4, dp_attention=4, spec=(1, 1, 6))),
        ):
            with self.subTest(model_type=model_type, layout=layout):
                self.assertEqual(
                    run(model_type, vision=True, **layout),
                    run(model_type, vision=False, **layout),
                )
        with self.assertRaisesRegex(NotImplementedError, "GLM has no vision program"):
            run("glm_moe_dsa", vision=True, ep=4, dp_attention=4)


if __name__ == "__main__":
    unittest.main()
