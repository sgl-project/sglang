import pytest

torch = pytest.importorskip("torch")

from test_store import MemoryBackend

from radix_transfer_plugins import OutputStore
from radix_transfer_plugins.schema import RawField
from radix_transfer_plugins.stashes.tensor_stash import TensorStash


@pytest.fixture
def store():
    return OutputStore(backend=MemoryBackend())


def _roundtrip(store, tensors):
    (ref,) = store.put_many([TensorStash(tensors)])
    (stash,) = store.read_many([(ref, list(tensors))], stash_type=TensorStash)
    return ref, stash.tensors


@pytest.mark.parametrize(
    "tensor",
    [
        torch.randn(2, 3, 4, dtype=torch.bfloat16),
        torch.randn(5),
        torch.arange(6, dtype=torch.int64).reshape(2, 3),
        torch.tensor([True, False, True]),
        torch.tensor(2.5),
        torch.empty(0, 4),
        torch.randn(4, 6)[:, ::2],
        torch.randn(3, requires_grad=True),
    ],
)
def test_roundtrip_keeps_dtype_shape_and_values(store, tensor):
    ref, tensors = _roundtrip(store, {"x": tensor})
    (spec,) = ref.fields
    assert spec.dtype == str(tensor.dtype).removeprefix("torch.")
    assert spec.shape == tuple(tensor.shape)
    assert tensors["x"].dtype == tensor.dtype
    assert torch.equal(tensors["x"], tensor.detach())


def test_reads_a_subset_of_fields(store):
    tensors = {"a": torch.ones(2), "b.c.0": torch.zeros(3, dtype=torch.int32)}
    (ref,) = store.put_many([TensorStash(tensors)])
    (stash,) = store.read_many([(ref, ["b.c.0"])], stash_type=TensorStash)
    assert list(stash.tensors) == ["b.c.0"]
    assert torch.equal(stash.tensors["b.c.0"], tensors["b.c.0"])


def test_empty_stash():
    assert TensorStash({}).is_empty()
    assert not TensorStash({"x": torch.ones(1)}).is_empty()


def test_unknown_dtype_is_rejected():
    field = RawField(dtype="complex64", shape=(1,), data=memoryview(bytes(8)))
    with pytest.raises(ValueError, match="complex64"):
        TensorStash.from_fields({"x": field})


def test_cuda_tensor_is_stored_from_host_memory(store):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tensor = torch.randn(4, device="cuda", dtype=torch.bfloat16)
    _, tensors = _roundtrip(store, {"x": tensor})
    assert tensors["x"].device.type == "cpu"
    assert torch.equal(tensors["x"], tensor.cpu())
