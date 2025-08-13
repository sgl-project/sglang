可以用来运行 w4afp8-block 量化格式的sglang
内含w4afp8-block相关的加速算子，cu128镜像下使用

若要启用activation_scheme 的  dynamic 模式
需要修改 sglang-w4a8-tp/python/sglang/srt/layers/quantization/w4afp8.py代码
moe_activation_scheme = "dynamic"
moe_activation_scheme = "static"


sglang-w4afp8-tp/python/sglang/srt/layers/quantization/w4afp8.py文件中的
"from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe"
有时会出现循环导入问题。
将其放入相应函数中可避开。
    def apply(
        self,
				...
    ) -> torch.Tensor:
        # avoid circular import
        from sglang.srt.layers.moe.topk import select_experts
        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe  