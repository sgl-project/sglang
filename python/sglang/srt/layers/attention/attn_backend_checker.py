from sglang.srt.server_args import ServerArgs

# TODO: add more features
@dataclass
class AttentionFeatureList:
    hardware: Union[str, List[str]] = "cuda_sm90"
    page_size_gt_1: Union[bool, List[int]] = False # If a list of int, it means the backend only supports limited page sizes
    cuda_graph: bool = False
    spec: bool = False
    spec_topk_gt_1: bool = False
    mla: bool = False
    sliding_window: bool = False
    dp_attention: bool = False
    chunked_prefix_cache: bool = False

# TODO: add more backends
support_matrix = {
    "fa3": AttentionFeatureList(
        hardware=["cuda_sm90"],
        page_size_gt_1=True,
        cuda_graph=True,
        spec=True,
        spec_topk_gt_1=True,
        mla=True,
        sliding_window=True,
        dp_attention=True,
        chunked_prefix_cache=True,
    ),
}

def check_page_size(feature_list: AttentionFeatureList, page_size: int, attention_backend: str):
    if isinstance(feature_list.page_size_gt_1, list):
        assert page_size in feature_list.page_size_gt_1, f"Page size {page_size} is not supported for {attention_backend}. It should be one of {feature_list.page_size_gt_1}."
    else:
        assert feature_list.page_size_gt_1, f"Page size > 1 is not supported for {attention_backend}."

def check_attention_backend_support(server_args: ServerArgs):
    attention_backend = server_args.attention_backend
    assert attention_backend in support_matrix, f"Attention backend {attention_backend} is not supported."

    feature_list = support_matrix[attention_backend]
    check_page_size(feature_list, server_args.page_size, attention_backend)
    #TODO: add checkers for other features
