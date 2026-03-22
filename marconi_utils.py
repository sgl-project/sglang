
def _key_match(key0, key1):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _normalize(values):
    # min-max normalization: scales the data linearly
    if len(values) > 1:
        min_val = min(values)
        max_val = max(values)
        if min_val != max_val:
            normalized_values = [(val - min_val) / (max_val - min_val) for val in values]
            return normalized_values
    return [1] * len(values)  # if list has <=1 elements or all elements are the same

def get_attn_flops(l, d):
    """Gets the number of floating-point operations (FLOPs)
    required in an Attention block.
    Formula (forward only): 8 * L * D^2 + 4 * L^2 * D
    
    Breakdown (operation -> FLOPs):
    - Attention KVQ (linear layer in multi-head self-attention that projects input into
        queries, keys, and values): 3 * 2 * L * D^2
    - Attention Mask (dot product between K and Q): 2 * L^2 * D
    - Attention Softmax: 3 * L^2
    - Attention Reduction (att V): 2 * L^2 * D
    - Attention Project (linear layer to project concatenated attention heads output to d_model):
        2 * L * D^2
    
    References:
    - https://www.adamcasson.com/posts/transformer-flops
    - https://arxiv.org/abs/2203.15556 (Chinchilla paper)

    Args:
        l (int): Sequence length.
        d (int): Model dimension.

    Returns:
        int: Total FLOPs.
    """
    return 8 * l * d**2 + 4 * l**2 * d


def get_mlp_flops(l, d):
    """Gets the number of floating-point operations (FLOPs)
    required in a MLP block (two feedforward linear layers).
    Formula (forward only): 2 * 2 * D * (4 * D) * L.
    
    References:
    - https://www.adamcasson.com/posts/transformer-flops
    - https://arxiv.org/abs/2203.15556 (Chinchilla paper)

    Args:
        l (int): Sequence length.
        d (int): Model dimension.

    Returns:
        int: Total FLOPs.
    """
    return 16 * l * d**2


def get_mamba1_flops(l, d, n):
    """Gets the number of floating-point operations (FLOPs)
    required in a Mamba1 layer. The FLOPs for a Mamba2 layer is similar.
    Formula (forward only):
    - Matmul: 12 * L * D^2
    - Vector engine: 16 * L * D * N
    - Scalar engine: 10 * L * D
    
    Breakdown (operation -> FLOPs, assuming the expansion factor is fixed to 2):
    - RMS Norm
    - Input proj: 4 * L * D * (D * 2)
    - 2 Silu: 2 * L * (D * 2)
    - SSM
        - 1D conv: 6 * L * D * 2
        - Proj: 2 * (2 * L * (D * 2) * N)
        - Proj dt: 2 * L * dt * (D * 2) + 2 * L * (D * 2) * dt
        - Softplus: L * (D * 2)
        - da: L * (D * 2) * N
        - exp: L * (D * 2) * N
        - dbu: 2 * L * (D * 2) * N
        - scan: 2 * L * (D * 2) * N
        - Out map: 2 * L * (D * 2) * N
        - y += Du: 2 * L * (D * 2)
    - Product: L * D * 2
    - Output: 2 * L * D * D * 2
    
    Reference: https://github.com/state-spaces/mamba

    Args:
        l (int): Sequence length.
        d (int): Model dimension.
        n (int): State/feature dimension.

    Returns:
        int: Total FLOPs.
    """
    # NOTE(ruipan) to self: this formula missed a term but was used in all experiments for the submission.
    # All numbers in the camera-ready version of the paper are now produced using the correct formula.
    # return 12 * l * d**2 + 16 * l * d * n + 10 * l  # orig submission version
    return 12 * l * d**2 + 16 * l * d * n + 10 * l * d  # correct version


def get_kvs_size(l, d):
    """Returns the size of the KV cache of an Attention layer in bytes.

    Args:
        l (int): Sequence length.
        d (int): Model dimension.

    Returns:
        int: KV cache size in bytes
    """
    # 2 is for k and v; d_model is essentially n_heads * d_head; fp16 is 2 bytes/parameter
    return 2 * l * d * 2


def get_mamba_state_size(d, n, conv_kernel=4, expand=2):
    """Returns the size of an SSM state of an SSM layer in bytes.

    Reference: https://github.com/state-spaces/mamba

    Args:
        d (int): Model dimension.
        n (int): State/feature dimension.
        conv_kernel (int, optional): Local convolution width. Defaults to 4.
        expand (int, optional): Block expansion factor. Defaults to 2.

    Returns:
        int: SSM state size in bytes
    """
    # ssm states: d_model is essentially n_heads * d_head; n is the hidden recurrent state dimensions; fp16, 2 bytes/parameter
    # conv states: 
    # in_channels = config.intermediate_size + 2 * config.state_size (defaults to 128)
    # intermediate_size is int(expand * self.hidden_size) = 2 * 4096 (for 7B)
    # default conv_kernel=4. conv state size is in_channels * conv_kernel * 2
    intermediate_size = expand * d
    in_channels = intermediate_size + 2 * n
    return d * n * 2 + (expand * d + 2 * n) * conv_kernel * 2


def get_model_state_size(l, d, n, num_mamba_layers=24, num_attn_layers=4):
    """Gets the total size of a model's states in bytes.

    Args:
        l (int): Sequence length.
        d (int): Model dimension.
        n (int): State/feature dimension.
        num_mamba_layers (int, optional): Number of Mamba layers in the model. Defaults to 24.
        num_attn_layers (int, optional): Number of Attention layers in the model. Defaults to 4.

    Returns:
        int: Total state size in bytes
    """
    return num_mamba_layers * get_mamba_state_size(d, n) + num_attn_layers * get_kvs_size(l, d)

def get_flops_efficiency(l, d, n, num_mamba_layers, num_attn_layers, num_mlp_layers):
    total_flops_saved = num_mamba_layers * get_mamba1_flops(l, d, n) + \
        num_attn_layers * get_attn_flops(l, d) + \
        num_mlp_layers * get_mlp_flops(l, d)
    total_state_size = get_model_state_size(l, d, n, num_mamba_layers=num_mamba_layers, num_attn_layers=num_attn_layers)
    return total_flops_saved / total_state_size
