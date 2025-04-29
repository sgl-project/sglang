import torch

if torch.version.hip is None:

    def all_to_all(
        fa,
        output,
        input,
        plan_meta,
        reg_buffer: int,
    ):
        torch.ops.sgl_kernel.all_to_all.default(
            fa,
            output,
            input,
            plan_meta,
            reg_buffer,
        )

    def all_to_all_plan(
        fa,
        output,
        input,
        output_split_sizes,
        input_split_sizes,
        chunk_size,
        output_split_offsets,
        input_split_offsets,
        plan_meta,
    ):
        if output_split_offsets is None:
            output_split_offsets = torch.zeros(
                (0,), dtype=torch.int64, device=input.device
            )
        if input_split_offsets is None:
            input_split_offsets = torch.zeros(
                (0,), dtype=torch.int64, device=input.device
            )
        return torch.ops.sgl_kernel.all_to_all_plan.default(
            fa,
            output,
            input,
            output_split_sizes,
            input_split_sizes,
            chunk_size,
            output_split_offsets,
            input_split_offsets,
            plan_meta,
        )
