try:
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )
except ImportError:
    raise RuntimeError(
        "LMCache is not installed. Please install it by running `pip install lmcache` in the root directory of LMCache"
    )

import os

import torch

from sglang.srt.configs.model_config import ModelConfig

os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CONFIG_FILE"] = "example_config.yaml"


def test_load_store_metadata():
    model_config = ModelConfig(
        model_path="Qwen/Qwen3-4B",
    )

    # Generate Dummy KV Cache
    head_num = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    layer_num = model_config.num_hidden_layers
    buffer_size = 256
    input_id_len = 16

    k_buffer = [
        torch.randn(buffer_size, head_num, head_dim, dtype=torch.bfloat16).cuda()
        for _ in range(layer_num)
    ]
    v_buffer = [
        torch.randn(buffer_size, head_num, head_dim, dtype=torch.bfloat16).cuda()
        for _ in range(layer_num)
    ]

    connector = LMCacheLayerwiseConnector(model_config, 1, 0, k_buffer, v_buffer)

    fake_token_ids = torch.randint(0, model_config.vocab_size, (input_id_len,)).tolist()
    fake_kv_indices = torch.randint(0, buffer_size, (input_id_len,))
    offset = 0

    store_metadata = StoreMetadata(
        last_node=None,
        token_ids=fake_token_ids,
        kv_indices=fake_kv_indices,
        offset=offset,
    )

    load_metadata = LoadMetadata(
        token_ids=fake_token_ids,
        slot_mapping=fake_kv_indices,
        offset=offset,
    )

    current_stream = torch.cuda.current_stream()

    retrieve_token_num = connector.start_load_kv(load_metadata)
    assert retrieve_token_num == 0

    connector.store_kv(store_metadata)
    current_stream.synchronize()

    # check retrieve
    gt_key_buffer = [
        torch.zeros(input_id_len, head_num, head_dim, dtype=torch.bfloat16).cuda()
        for _ in range(layer_num)
    ]
    gt_value_buffer = [
        torch.zeros(input_id_len, head_num, head_dim, dtype=torch.bfloat16).cuda()
        for _ in range(layer_num)
    ]

    for i in range(layer_num):
        gt_key_buffer[i] = k_buffer[i][fake_kv_indices]
        gt_value_buffer[i] = v_buffer[i][fake_kv_indices]

    # clear the k_buffer and v_buffer
    for _ in range(layer_num):
        k_buffer[i].zero_()
        v_buffer[i].zero_()

    retrieve_token_num = connector.start_load_kv(load_metadata)
    assert retrieve_token_num == input_id_len

    for i in range(layer_num):
        current_stream.synchronize()
        connector.load_kv_layerwise(i)

    current_stream.synchronize()
    test_key_buffer = [
        torch.zeros(input_id_len, head_num, head_dim, dtype=torch.bfloat16).cuda()
        for _ in range(layer_num)
    ]
    test_value_buffer = [
        torch.zeros(input_id_len, head_num, head_dim, dtype=torch.bfloat16).cuda()
        for _ in range(layer_num)
    ]

    for i in range(layer_num):
        test_key_buffer[i] = k_buffer[i][fake_kv_indices]
        test_value_buffer[i] = v_buffer[i][fake_kv_indices]

    for i in range(layer_num):
        assert torch.allclose(test_key_buffer[i], gt_key_buffer[i])
        assert torch.allclose(test_value_buffer[i], gt_value_buffer[i])

    print("================================================")
    print("TEST_LOAD_STORE_METADATA PASSED!")
    print("================================================")
    connector.close()


if __name__ == "__main__":
    test_load_store_metadata()
