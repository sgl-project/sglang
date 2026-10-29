import json

import msgspec
import numpy as np
import pytest

from radix_transfer_plugins import (
    OutputStore,
    RangeRead,
    RawField,
    StorageBackend,
    StoreRef,
)
from radix_transfer_plugins.stashes.token_output import TokenOutputStash


class MemoryBackend(StorageBackend):
    name = "memory"
    config_type = msgspec.Struct

    def __init__(self):
        self.objects: dict[str, dict[str, bytes]] = {}
        self.reads: list[list[str]] = []
        self.range_reads: list[list[RangeRead]] = []
        self.next_id = 0

    def put_many(self, objects):
        keys = []
        for parts in objects:
            key = f"mem/{self.next_id}"
            self.next_id += 1
            self.objects[key] = {name: bytes(data) for name, data in parts.items()}
            keys.append(key)
        return keys

    def get_many(self, keys, *, parts):
        out = []
        for key, named in zip(keys, parts):
            self.reads.append([name for name, _ in named])
            stored = self.objects[key]
            # Like a real backend: the caller owns what a read returns.
            out.append({name: memoryview(bytearray(stored[name])) for name, _ in named})
        return out

    def get_ranges(self, reads, *, into):
        self.range_reads.append(list(reads))
        for read in reads:
            stored = self.objects[read.ref.key][read.field]
            data = stored[read.src_offset : read.src_offset + read.nbytes]
            into[read.dst_offset : read.dst_offset + read.nbytes] = data

    def delete_many(self, keys, *, parts):
        for key, names in zip(keys, parts):
            assert sorted(self.objects[key]) == sorted(names)
            del self.objects[key]

    def close(self):
        pass


class BytesStash(msgspec.Struct, frozen=True):
    """A stash whose fields are raw already, e.g. bfloat16 bytes numpy cannot hold."""

    fields: dict

    def is_empty(self):
        return not self.fields

    def to_fields(self):
        return dict(self.fields)

    @classmethod
    def from_fields(cls, fields):
        return cls(fields=dict(fields))


def token_stash(seed: int, *, routed_dtype=np.int32, num_tokens=6, mask_rows=None):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(
        1, 4, size=num_tokens if mask_rows is None else mask_rows
    ).astype(np.int32)
    return TokenOutputStash(
        routed_experts=rng.integers(0, 128, size=(num_tokens, 48, 8)).astype(
            routed_dtype
        ),
        indexer_topk=rng.integers(0, 4096, size=(num_tokens, 2, 16)).astype(np.int32),
        output_token_sampling_mask_lengths=lengths,
        output_token_sampling_mask_token_ids=rng.integers(
            0, 151936, size=int(lengths.sum())
        ).astype(np.int32),
        output_token_sampling_logprobs=rng.standard_normal(int(lengths.sum())).astype(
            np.float32
        ),
    )


def assert_stash_equal(got: TokenOutputStash, want: TokenOutputStash):
    for name in TokenOutputStash.__struct_fields__:
        g, w = getattr(got, name), getattr(want, name)
        if w is None:
            assert g is None, name
            continue
        assert g.dtype == w.dtype and g.shape == w.shape, name
        np.testing.assert_array_equal(g, w)


def through_json(ref: StoreRef) -> StoreRef:
    return StoreRef.from_builtins(json.loads(json.dumps(ref.to_builtins())))


def read_all(store: OutputStore, refs, *, stash_type):
    """Every field of each ref."""
    return store.read_many(
        [(ref, [s.name for s in ref.fields]) for ref in refs], stash_type=stash_type
    )


def test_round_trip_through_a_json_ref():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    stashes = [token_stash(1), token_stash(2)]
    refs = store.put_many(stashes)

    assert [ref.backend for ref in refs] == ["memory", "memory"]
    spec = {s.name: s for s in refs[0].fields}["routed_experts"]
    assert (spec.dtype, spec.shape, spec.codec, spec.nbytes) == (
        "int32",
        (6, 48, 8),
        "identity",
        6 * 48 * 8 * 4,
    )

    got = read_all(
        store, [through_json(ref) for ref in refs], stash_type=TokenOutputStash
    )
    for g, w in zip(got, stashes):
        assert_stash_equal(g, w)


@pytest.mark.parametrize("dtype", [np.int32, np.uint16, np.uint8])
def test_any_routed_experts_dtype_round_trips(dtype):
    store = OutputStore(backend=MemoryBackend())
    stash = token_stash(3, routed_dtype=dtype)
    (ref,) = store.put_many([stash])
    assert {s.name: s.dtype for s in ref.fields}["routed_experts"] == np.dtype(
        dtype
    ).name
    (got,) = read_all(store, [ref], stash_type=TokenOutputStash)
    assert_stash_equal(got, stash)


def test_reading_a_subset_fetches_only_those_parts():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    stash = token_stash(4)
    (ref,) = store.put_many([stash])

    (got,) = store.read_many([(ref, ["routed_experts"])], stash_type=TokenOutputStash)
    assert backend.reads[-1] == ["routed_experts"]
    np.testing.assert_array_equal(got.routed_experts, stash.routed_experts)
    assert got.indexer_topk is None and got.output_token_sampling_logprobs is None


def test_empty_fields_live_only_in_the_ref():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    stash = token_stash(5, num_tokens=0, mask_rows=0)
    (ref,) = store.put_many([stash])

    assert all(s.nbytes == 0 for s in ref.fields)
    assert backend.objects[ref.key] == {}
    (got,) = read_all(store, [ref], stash_type=TokenOutputStash)
    assert_stash_equal(got, stash)


def test_dtype_is_opaque_to_the_store():
    store = OutputStore(backend=MemoryBackend())
    raw = RawField(dtype="bfloat16", shape=(2, 3), data=memoryview(bytes(range(12))))
    (ref,) = store.put_many([BytesStash(fields={"latents": raw})])
    (got,) = read_all(store, [ref], stash_type=BytesStash)
    field = got.fields["latents"]
    assert (field.dtype, field.shape, bytes(field.data)) == (
        "bfloat16",
        (2, 3),
        bytes(range(12)),
    )


def test_fields_share_memory_with_the_arrays():
    stash = token_stash(6)
    data = stash.to_fields()["routed_experts"].data
    assert np.shares_memory(np.frombuffer(data, dtype=np.uint8), stash.routed_experts)


def test_delete_removes_every_stored_part():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    refs = store.put_many([token_stash(7), token_stash(8)])
    store.delete_many(refs)
    assert backend.objects == {}


def test_refs_of_another_backend_are_rejected():
    store = OutputStore(backend=MemoryBackend())
    (ref,) = store.put_many([token_stash(9)])
    other = msgspec.structs.replace(ref, backend="mooncake")
    with pytest.raises(ValueError, match="mooncake"):
        store.read_many([(other, ["routed_experts"])], stash_type=TokenOutputStash)
    with pytest.raises(ValueError, match="mooncake"):
        store.delete_many([other])


def test_ref_parsing_is_strict():
    store = OutputStore(backend=MemoryBackend())
    (ref,) = store.put_many([token_stash(10)])
    value = ref.to_builtins()
    value["extra"] = 1
    with pytest.raises(msgspec.ValidationError):
        StoreRef.from_builtins(value)


def test_unknown_codecs_fail_on_both_sides():
    with pytest.raises(ValueError, match="unknown codec"):
        OutputStore(backend=MemoryBackend(), codecs={"routed_experts": "zstd"})
    store = OutputStore(backend=MemoryBackend())
    (ref,) = store.put_many([token_stash(11)])
    fields = tuple(msgspec.structs.replace(s, codec="zstd") for s in ref.fields)
    with pytest.raises(ValueError, match="unknown codec"):
        read_all(
            store,
            [msgspec.structs.replace(ref, fields=fields)],
            stash_type=TokenOutputStash,
        )


def test_field_names_cannot_hold_a_slash():
    store = OutputStore(backend=MemoryBackend())
    raw = RawField(dtype="uint8", shape=(1,), data=memoryview(b"x"))
    with pytest.raises(ValueError, match="invalid field name"):
        store.put_many([BytesStash(fields={"a/b": raw})])


def test_unknown_fields_are_rejected_by_the_stash():
    raw = RawField(dtype="int32", shape=(0,), data=memoryview(b""))
    with pytest.raises(ValueError, match="unknown token output fields"):
        TokenOutputStash.from_fields({"routed_experts": raw, "hidden_states": raw})


def test_empty_stash():
    assert TokenOutputStash().is_empty()
    assert TokenOutputStash().to_fields() == {}
    assert not token_stash(12).is_empty()


def test_read_many_reads_named_fields_into_arrays_the_caller_owns():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    stashes = [token_stash(20 + i) for i in range(3)]
    refs = store.put_many(stashes)

    got = store.read_many(
        [(ref, ["routed_experts", "indexer_topk"]) for ref in refs],
        stash_type=TokenOutputStash,
    )

    assert backend.reads == [["routed_experts", "indexer_topk"]] * 3
    for g, w in zip(got, stashes):
        np.testing.assert_array_equal(g.routed_experts, w.routed_experts)
        np.testing.assert_array_equal(g.indexer_topk, w.indexer_topk)
        assert g.output_token_sampling_mask_lengths is None
        assert g.routed_experts.flags.writeable


def test_read_many_fails_before_reading_a_missing_field():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    (ref,) = store.put_many([token_stash(31)])
    with pytest.raises(ValueError, match="no fields"):
        store.read_many(
            [(ref, ["routed_experts", "hidden_states"])], stash_type=TokenOutputStash
        )
    assert backend.reads == []


def test_read_many_returns_empty_fields_without_reading():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    (ref,) = store.put_many([token_stash(32, num_tokens=0, mask_rows=0)])
    (got,) = store.read_many([(ref, ["routed_experts"])], stash_type=TokenOutputStash)
    assert got.routed_experts.shape == (0, 48, 8)
    assert backend.reads == [[]]


def test_read_ranges_match_numpy_slices():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    stashes = [token_stash(40 + i) for i in range(3)]
    refs = store.put_many(stashes)
    rng = np.random.default_rng(0)
    # Random row slices of random fields, packed into `into` in the order drawn.
    reads, wants, end = [], [], 0
    for _ in range(12):
        i = int(rng.integers(len(refs)))
        name = str(rng.choice(TokenOutputStash.__struct_fields__))
        array = getattr(stashes[i], name)
        start = int(rng.integers(len(array)))
        want = array[start : int(rng.integers(start + 1, len(array) + 1))]
        reads.append(
            RangeRead(
                ref=refs[i],
                field=name,
                src_offset=start * array[0].nbytes,
                nbytes=want.nbytes,
                dst_offset=end,
            )
        )
        wants.append(want)
        end += want.nbytes
    into = np.zeros(end, dtype=np.uint8)

    # Passed out of order: each range lands at its own dst_offset.
    store.read_ranges(
        [reads[j] for j in rng.permutation(len(reads))], into=memoryview(into)
    )

    assert into.tobytes() == b"".join(want.tobytes() for want in wants)
    assert len(backend.range_reads) == 1


def test_read_ranges_check_every_range_before_reading():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    (ref,) = store.put_many([token_stash(50)])
    size = {s.name: s.nbytes for s in ref.fields}["indexer_topk"]
    good = RangeRead(
        ref=ref, field="indexer_topk", src_offset=0, nbytes=size, dst_offset=0
    )
    zstd = msgspec.structs.replace(
        ref, fields=tuple(msgspec.structs.replace(s, codec="zstd") for s in ref.fields)
    )
    into = memoryview(bytearray(size))
    for change, match in [
        ({"field": "hidden_states"}, "no fields"),
        ({"ref": zstd}, "zstd coded"),
        ({"src_offset": -1}, "not a range"),
        ({"nbytes": 0}, "not a range"),
        ({"src_offset": 1}, "not a range"),  # runs past the field
        ({"dst_offset": -1}, "outside"),
        ({"dst_offset": 1}, "outside"),  # runs past `into`
    ]:
        bad = msgspec.structs.replace(good, **change)
        with pytest.raises(ValueError, match=match):
            store.read_ranges([good, bad], into=into)
    with pytest.raises(ValueError, match="read-only"):
        store.read_ranges([good], into=memoryview(bytes(size)))
    store.read_ranges([], into=memoryview(bytearray()))
    assert backend.range_reads == []
