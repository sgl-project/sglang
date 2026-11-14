"""Query width for target decode and verification."""


def sparse_query_width(forward_batch):
    """Decode has one query per request; verify uses its chain width."""
    if forward_batch.forward_mode.is_decode():
        return 1
    return forward_batch.spec_info.num_tokens_per_req
