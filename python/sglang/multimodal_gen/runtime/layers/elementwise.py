from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp


class MulAdd(CustomOp):
    """
    Fuse elementwise mul and add
    Input: a, b, c, OptionalInt[k]
    Output: a * (k + b) + c
    """

    def __init__(self, prefix: str = ""):
        self.function_name = "mul_add"
        super().__init__()
