import torch
from mooncake_store import MooncakeStore


def test_init_and_warmup():
    store = MooncakeStore()
    assert store.store is not None


def test_register_buffer():
    store = MooncakeStore()
    tensor = torch.zeros(1024, dtype=torch.float32)
    store.register_buffer(tensor)


def test_set_and_get():
    store = MooncakeStore()

    key = ["test_key_" + str(i) for i in range(2)]
    tensor = torch.arange(256, dtype=torch.float32).cuda()
    ptrs = [tensor.data_ptr(), tensor.data_ptr()]
    sizes = [tensor.numel() * tensor.element_size()] * 2

    store.set(key, target_location=ptrs, target_sizes=sizes)
    store.get(key, target_location=ptrs, target_sizes=sizes)


def test_exists():
    store = MooncakeStore()
    keys = ["test_key_0", "non_existent_key"]
    result = store.exists(keys)
    assert isinstance(result, dict)
    assert "test_key_0" in result


if __name__ == "__main__":
    test_init_and_warmup()
    test_register_buffer()
    test_set_and_get()
    test_exists()
