import dataclasses

import msgspec
from dsv41_args import DeepseekV41Args


def to_v41_args(ref_args) -> DeepseekV41Args:
    fields = dataclasses.asdict(ref_args)
    names = set(DeepseekV41Args.__struct_fields__)
    kept = {k: v for k, v in fields.items() if k in names}
    kept["vision_enabled"] = ref_args.vision_enabled
    kept["expert_fp4"] = ref_args.expert_dtype == "fp4"
    return msgspec.convert(kept, DeepseekV41Args)
