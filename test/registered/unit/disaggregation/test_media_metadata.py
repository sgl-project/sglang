from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.srt.disaggregation.media_metadata import (
    MediaMetadata,
    expand_image_prompt,
    export_metadata,
    grid_row,
    image_mrope,
    import_metadata,
    request_image_metadata,
)
from sglang.srt.managers.req_types import MultimodalInputs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_positions_match_explicit_two_image_reference():
    # 2 text, 2x3 image, 1 text, 1x2 image, 2 text. Image positions use
    # max(h,w), rather than number of image tokens, to advance the text base.
    pos, delta = image_mrope(13, [(1, 4, 6), (1, 2, 4)], [(2, 7), (9, 10)], 2)
    expected = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 6, 8, 9],
            [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 6, 8, 9],
            [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9],
        ]
    )
    torch.testing.assert_close(pos, expected)
    assert delta.item() == -3


@pytest.mark.parametrize(
    "grid", [[], [1, 2], [0, 4, 4], [2, 4, 4], [1, 3, 4], [1, True, 4], [1, 2.0, 4]]
)
def test_bad_grids_are_rejected(grid):
    with pytest.raises(ValueError):
        grid_row(grid, merge_size=2)


def test_expansion_has_bounded_allocation_and_exact_item_count():
    assert expand_image_prompt([7, 99, 8], [(1, 4, 6)], 99, 2, 20) == (
        [7] + [99] * 6 + [8],
        [(1, 6)],
    )
    with pytest.raises(ValueError, match="context"):
        expand_image_prompt([99], [(1, 100000, 100000)], 99, 2, 8192)
    with pytest.raises(ValueError, match="count"):
        expand_image_prompt([99, 99], [(1, 4, 6)], 99, 2, 20)


def make_prefill():
    output = request_image_metadata(
        [{"image_grid_thw": [[1, 4, 6]]}],
        [7, 99, 8],
        image_token_id=99,
        merge_size=2,
        limit=100,
    )
    item = output.mm_items[0]
    # A real prefill identity hashes pixels and processor config, not just grid.
    item.set_identity(bytes(range(32)))
    padded = [7] + item.padding_sequence(6) + [8]
    mm = MultimodalInputs(
        mm_items=[item],
        im_token_id=99,
        mrope_positions=output.mrope_positions,
        mrope_position_delta=output.mrope_position_delta,
    )
    return SimpleNamespace(
        origin_input_ids=padded, multimodal_inputs=mm
    ), output.input_ids


def make_decode(ids):
    req = SimpleNamespace(
        origin_input_ids=list(ids), multimodal_inputs=None, output_ids=[]
    )
    req.extend_image_inputs = lambda mm: setattr(req, "multimodal_inputs", mm)
    return req


def test_transfer_restores_prefill_identity_and_positions_without_pixels():
    prefill, ids = make_prefill()
    payload = export_metadata(prefill, merge_size=2, capacity=4096)
    req = make_decode(ids)
    import_metadata(payload, req, merge_size=2, image_token_id=99)
    assert list(req.origin_input_ids) == list(prefill.origin_input_ids)
    assert req.multimodal_inputs.mm_items[0].identity == bytes(range(32))
    assert req.multimodal_inputs.mm_items[0].feature is None
    torch.testing.assert_close(
        req.multimodal_inputs.mrope_positions, prefill.multimodal_inputs.mrope_positions
    )


def test_bad_transfer_is_rejected_before_request_mutation():
    prefill, ids = make_prefill()
    payload = export_metadata(prefill, merge_size=2, capacity=4096)
    req = make_decode([123] + ids[1:])
    original = req.origin_input_ids.copy()
    with pytest.raises(ValueError, match="prompt"):
        import_metadata(payload, req, merge_size=2, image_token_id=99)
    assert req.origin_input_ids == original and req.multimodal_inputs is None
    meta = msgspec.msgpack.decode(payload, type=MediaMetadata)
    meta.images[0].span = (1, 7)
    with pytest.raises(ValueError, match="span"):
        import_metadata(
            msgspec.msgpack.encode(meta),
            make_decode(ids),
            merge_size=2,
            image_token_id=99,
        )
    with pytest.raises(ValueError, match="capacity"):
        export_metadata(prefill, merge_size=2, capacity=1)


def test_empty_transfer_cannot_silently_drop_image_metadata():
    with pytest.raises(ValueError, match="did not send"):
        import_metadata(b"", make_decode([1, 99, 2]), merge_size=2, image_token_id=99)
    req = make_decode([1, 2])
    import_metadata(b"", req, merge_size=2, image_token_id=99)
    assert req.multimodal_inputs is None


def test_request_metadata_rejects_mixed_input():
    with pytest.raises(ValueError, match="mix"):
        request_image_metadata(
            [{"image_grid_thw": [[1, 4, 6]]}, "data:image/png;base64,AA=="],
            [99, 99],
            image_token_id=99,
            merge_size=2,
            limit=100,
        )


def test_registered_metadata_buffers_roundtrip_and_slot_reuse():
    from sglang.srt.disaggregation.utils import MetadataBuffers
    from sglang.srt.environ import envs

    with (
        envs.SGLANG_PD_MEDIA_METADATA_MODE.override("transfer"),
        envs.SGLANG_PD_MEDIA_METADATA_MAX_BYTES.override(4096),
    ):
        src = MetadataBuffers(2, 16, torch.float32, 16)
        dst = MetadataBuffers(2, 16, torch.float32, 16)
    prefill, ids = make_prefill()
    prefill.metadata_buffer_index = 0
    src._set_media_metadata(prefill)
    # A whole registered slot is copied by a P/D transport, including its
    # explicit used length; trailing bytes may belong to a previous request.
    dst.media_metadata[1].copy_(src.media_metadata[0])
    dst.media_metadata_len[1].copy_(src.media_metadata_len[0])
    decode = make_decode(ids)
    import_metadata(dst.get_media_metadata(1), decode, merge_size=2, image_token_id=99)
    assert list(decode.origin_input_ids) == list(prefill.origin_input_ids)
    ptrs, _, widths = src.get_buf_infos()
    assert src.media_metadata.data_ptr() in ptrs and 4096 in widths
    dst.clear_slot(1)
    assert dst.get_media_metadata(1) == b""
    dst.media_metadata_len[1, 0] = 4097
    with pytest.raises(ValueError, match="length"):
        dst.get_media_metadata(1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x"]))
