import sgl_kernel
import torch

from sglang.jit_kernel.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def torch_top_k_renorm_probs(probs, top_k):
    """Vectorized PyTorch implementation of top-k renormalization."""
    batch_size, vocab_size = probs.shape

    # Handle scalar or tensor k
    if isinstance(top_k, int):
        k_val = min(max(top_k, 1), vocab_size)
        # Get top-k indices for all batches at once
        _, topk_indices = torch.topk(probs, k_val, dim=1, largest=True)

        # Create mask: batch_size x vocab_size
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_indices, 1.0)

        # Vectorized renormalization
        masked_probs = probs * mask
        renorm_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-10)
        return renorm_probs
    else:
        # Variable k per batch - need to handle separately
        renorm_probs = torch.zeros_like(probs)
        for i in range(batch_size):
            k_val = min(max(top_k[i].item(), 1), vocab_size)
            _, topk_indices = torch.topk(probs[i], k_val, largest=True)
            mask = torch.zeros_like(probs[i])
            mask[topk_indices] = 1.0
            masked_probs = probs[i] * mask
            renorm_probs[i] = masked_probs / (masked_probs.sum() + 1e-10)
        return renorm_probs


def torch_top_p_renorm_probs(probs, top_p, eps=1e-5):
    """Vectorized PyTorch implementation of top-p renormalization."""
    batch_size, vocab_size = probs.shape

    # Handle scalar or tensor p
    if isinstance(top_p, float):
        p_val = top_p
        # Vectorized implementation for uniform top_p
        # Sort probs in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)

        # Find cutoff: where cumsum exceeds top_p
        cutoff_mask = cumsum_probs <= p_val
        # Keep at least one token (the highest prob)
        cutoff_mask[:, 0] = True

        # Create mask in original order
        mask = torch.zeros_like(probs)
        mask.scatter_(1, sorted_indices, cutoff_mask.float())

        # Vectorized renormalization
        masked_probs = probs * mask
        renorm_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + eps)
        return renorm_probs
    else:
        # Variable p per batch - need to handle separately
        renorm_probs = torch.zeros_like(probs)
        for i in range(batch_size):
            p_val = top_p[i].item()
            sorted_prob, indices = torch.sort(probs[i], descending=False)
            cdf = torch.cumsum(sorted_prob, dim=-1)
            mask = torch.zeros(vocab_size, dtype=torch.float32, device=probs.device)
            mask.scatter_(0, indices, (cdf >= (1 - p_val) - eps).float())
            masked_probs = probs[i] * mask
            renorm_probs[i] = masked_probs / (masked_probs.sum() + eps)
        return renorm_probs


def _top_k_fn(provider, probs, top_k_tensor):
    if provider == "torch":
        return torch_top_k_renorm_probs(probs, top_k_tensor)
    return sgl_kernel.top_k_renorm_prob(probs, top_k_tensor)


def _top_p_fn(provider, probs, top_p_tensor):
    if provider == "torch":
        return torch_top_p_renorm_probs(probs, top_p_tensor)
    return sgl_kernel.top_p_renorm_prob(probs, top_p_tensor)


@marker.parametrize("k", [10, 100, 500], [10])
@marker.parametrize("vocab_size", [111, 32000, 128256], [111])
@marker.parametrize("batch_size", [16, 64, 128], [16])
@marker.benchmark("provider", ["torch", "sglang"])
def benchmark_top_k_renorm(batch_size, vocab_size, k, provider):
    # Skip invalid configurations
    if k >= vocab_size:
        marker.skip(f"k={k} >= vocab_size={vocab_size}")

    torch.manual_seed(42)
    device = torch.device("cuda")

    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.int32)

    return marker.do_bench(
        _top_k_fn,
        input_args=(provider, probs, top_k_tensor),
        # probs is read; clone it per iter. provider/top_k_tensor handled separately.
        graph_clone_args=(1,),
        use_cuda_graph=False,
        memory_args=(probs, top_k_tensor),
    )


@marker.parametrize("p", [0.1, 0.5, 0.9], [0.5])
@marker.parametrize("vocab_size", [111, 32000, 128256], [111])
@marker.parametrize("batch_size", [16, 64, 128], [16])
@marker.benchmark("provider", ["torch", "sglang"])
def benchmark_top_p_renorm(batch_size, vocab_size, p, provider):
    torch.manual_seed(42)
    device = torch.device("cuda")

    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_p_tensor = torch.full((batch_size,), p, device=device, dtype=torch.float32)

    return marker.do_bench(
        _top_p_fn,
        input_args=(provider, probs, top_p_tensor),
        graph_clone_args=(1,),
        use_cuda_graph=False,
        memory_args=(probs, top_p_tensor),
    )


if __name__ == "__main__":
    benchmark_top_k_renorm.run()
    benchmark_top_p_renorm.run()
