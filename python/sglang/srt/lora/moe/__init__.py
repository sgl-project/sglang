"""SGLang's MoE LoRA execution engine.

Plan tables load in ``execution_plan`` (what runs, per layout x phase, with
the one measured rank band); tile tables load in ``launch_config`` (how each
kernel launches, rank- and M-bucketed). ``MoeLoraLayerEngine`` resolves both
once per layer at weight bind; the forward path is a phase lookup.
"""
