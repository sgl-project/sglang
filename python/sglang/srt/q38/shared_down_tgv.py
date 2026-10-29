"""Independent target shared-down BF16 GEMM experiment; no quantization change."""
def install():
    import torch,logging
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm
    log=logging.getLogger(__name__);original=Qwen2MoeSparseMoeBlock.__init__;seen=set()
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        # The draft's existing math and all routed experts retain their paths.
        if kwargs.get('is_nextn',False) or self.shared_expert is None:return
        down=self.shared_expert.down_proj
        if type(down.quant_method).__name__!='UnquantizedLinearMethod':return
        native=down.quant_method.apply
        def apply(layer,x,bias=None):
            if (bias is None and isinstance(x,torch.Tensor) and x.is_cuda
                and x.dtype==torch.bfloat16 and layer.weight.dtype==torch.bfloat16
                and x.ndim==2 and x.shape[0] in (4,8,16)
                and tuple(layer.weight.shape)==(2560,640)):
                m=x.shape[0]
                if m not in seen:
                    seen.add(m);log.info('Q38_SHARED_DOWN_TGV_ENGAGED rows=%s N2560 K640 BF16',m)
                return cutedsl_bf16_gemm(x,layer.weight)
            return native(layer,x,bias)
        down.quant_method.apply=apply
    Qwen2MoeSparseMoeBlock.__init__=init
