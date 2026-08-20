# Vendored from flashinfer-ai/flashinfer main at 76704c4 (SM100 GDN CP
# prefill closure, incl. #4436 pooled state / checkpointing / dtype parity);
# pending a FlashInfer release that ships it.
import cutlass
import cutlass.cute as cute

# ─── AlphaProcessor ──────────────────────────────────────────────────────────
# Translates FlatMainloopTmaWarpSpecializedDeltaRule::AlphaProcessor.


class AlphaProcessor:
    CUMSUM_LOG = 0
    CUMPROD = 1
    CUMPROD_SCALE = 2
    CUMPROD_END_RCP = 2
    CUMPROD_NEG_END_RCP = 2
    NUM_CHANNELS = 3

    @cute.jit
    def run(
        self,
        vecs: cute.Tensor,
        scale: cutlass.Float32,
        channel2_neg_end_rcp: cutlass.Constexpr = False,
        channel2_end_rcp: cutlass.Constexpr = False,
    ):
        WARP_SIZE = 32
        blk_q = cute.size(vecs.shape[0])
        num_iters = blk_q // WARP_SIZE

        lane_id = cute.arch.lane_idx()

        # vecs shape (blk_q, NUM_CHANNELS) col-major.
        vecs_32 = cute.flat_divide(vecs, (WARP_SIZE,))

        frag = cute.make_rmem_tensor(num_iters, cutlass.Float32)
        for i in cutlass.range_constexpr(num_iters):
            raw = cutlass.Float32(vecs_32[lane_id, i, AlphaProcessor.CUMSUM_LOG])
            frag[i] = cute.math.log2(raw + cutlass.Float32(1e-10), fastmath=True)

        for log_off in cutlass.range_constexpr(5):  # offsets 1, 2, 4, 8, 16
            off = 1 << log_off
            for i in cutlass.range_constexpr(num_iters):
                v = cute.arch.shuffle_sync_up(frag[i], off, mask_and_clamp=0)
                if lane_id >= off:
                    frag[i] = frag[i] + v

        for i in cutlass.range_constexpr(1, num_iters):
            carry = cute.arch.shuffle_sync(frag[i - 1], 31)
            frag[i] = frag[i] + carry

        if cutlass.const_expr(channel2_neg_end_rcp or channel2_end_rcp):
            end_log = cute.arch.shuffle_sync(frag[num_iters - 1], 31)
        for i in cutlass.range_constexpr(num_iters):
            vecs_32[lane_id, i, AlphaProcessor.CUMSUM_LOG] = frag[i]
            cumprod = cute.math.exp2(frag[i], fastmath=True)
            vecs_32[lane_id, i, AlphaProcessor.CUMPROD] = cumprod
            if cutlass.const_expr(channel2_neg_end_rcp):
                vecs_32[lane_id, i, AlphaProcessor.CUMPROD_NEG_END_RCP] = (
                    -cute.math.exp2(end_log - frag[i], fastmath=True)
                )
            elif cutlass.const_expr(channel2_end_rcp):
                vecs_32[lane_id, i, AlphaProcessor.CUMPROD_END_RCP] = cute.math.exp2(
                    end_log - frag[i], fastmath=True
                )
            else:
                vecs_32[lane_id, i, AlphaProcessor.CUMPROD_SCALE] = cumprod * scale
