"""Version 1 of the private, single-receiver image handoff."""

import asyncio
import re
from urllib.parse import urlsplit

import msgspec

# Keep one recovery task per ambiguous registered destination until the original
# write completes or E fences it. The receiver's admission limit bounds this set
# for each scheduler, including when an encoder remains unreachable.
_fence_tasks: set[asyncio.Task] = set()


class ImageLayout(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    t: int
    h: int
    w: int
    identity: str


class EncoderHandoff(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    version: int
    peer: str
    request_id: str
    transfer_id: str
    part_id: str
    part_idx: int
    num_parts: int
    image_count: int
    images: list[ImageLayout]

    @classmethod
    def parse(cls, data, placeholder_count: int) -> "EncoderHandoff":
        transfer = msgspec.json.decode(data, type=cls)
        if transfer.version != 1:
            raise ValueError("unsupported encoder_handoff version")
        peer = urlsplit(transfer.peer)
        if (
            peer.scheme != "grpc"
            or not peer.hostname
            or not peer.port
            or peer.username is not None
            or peer.path
            or peer.query
            or peer.fragment
        ):
            raise ValueError(
                "encoder_handoff requires an explicit grpc://host:port peer"
            )
        if not transfer.request_id or not re.fullmatch(
            "[0-9a-f]{32}", transfer.transfer_id
        ):
            raise ValueError("invalid staged request or transfer ID")
        if (transfer.num_parts, transfer.part_idx, transfer.part_id) != (
            1,
            0,
            transfer.transfer_id + "_local_part_0",
        ):
            raise ValueError("invalid staged image part")
        if (
            not transfer.images
            or len(transfer.images) != transfer.image_count
            or transfer.image_count != placeholder_count
        ):
            raise ValueError(
                "staged image count differs from layout or prompt placeholders"
            )
        for image in transfer.images:
            if min(image.t, image.h, image.w) < 1 or not re.fullmatch(
                "[0-9a-f]{64}", image.identity
            ):
                raise ValueError("invalid staged image grid or identity")
        return transfer

    def validate_layout(
        self, merge_size: int, prompt_len: int, max_length: int
    ) -> None:
        rows = 0
        for image in self.images:
            if image.h % merge_size or image.w % merge_size:
                raise ValueError("staged image grid is incompatible with the model")
            rows += image.t * (image.h // merge_size) * (image.w // merge_size)
        if prompt_len - self.image_count + rows >= max_length:
            raise ValueError("staged images exceed the model's input length limit")

    def grids(self):
        return [[image.t, image.h, image.w] for image in self.images]

    def bind_identities(self, mm_items, routing_hashes):
        from sglang.srt.environ import envs

        if len(mm_items) != self.image_count:
            raise ValueError("staged image count differs from rebuilt prompt")
        for index, (image, item) in enumerate(zip(self.images, mm_items)):
            identity = bytes.fromhex(image.identity)
            if (
                item.precomputed_embeddings is not None
                and not envs.SGLANG_MM_SKIP_COMPUTE_HASH.get()
                and item.cache_key != identity
            ):
                raise ValueError("staged image embedding identity mismatch")
            item.set_identity(identity)
            if routing_hashes and len(routing_hashes) == self.image_count:
                hint = routing_hashes[index]
                if hint is not None:
                    # The legacy scalar is for routing. Prefix keys and padded
                    # token IDs continue to use the full embedding digest.
                    item.hash = hint

    def native_transfer(self):
        from sglang.encoder.v1.encoder_pb2 import StagedImageTransfer

        fields = msgspec.to_builtins(self)
        del fields["peer"]
        return StagedImageTransfer(**fields)

    async def deliver(self, receive_url: str, timeout: float):
        import grpc
        from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc

        async with grpc.aio.insecure_channel(urlsplit(self.peer).netloc) as channel:
            stub = encoder_pb2_grpc.SglangEncoderStub(channel)
            await stub.Deliver(
                encoder_pb2.DeliverRequest(
                    transfer=self.native_transfer(), receive_url=receive_url
                ),
                timeout=timeout,
            )

    async def receive_nixl(self, receiver, shape: tuple[int, int], timeout: float):
        import grpc
        import torch
        from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc

        deadline = asyncio.get_running_loop().time() + timeout
        transfer = self.native_transfer()
        async with grpc.aio.insecure_channel(urlsplit(self.peer).netloc) as channel:
            stub = encoder_pb2_grpc.SglangEncoderStub(channel)
            spec = await stub.Describe(
                encoder_pb2.DescribeRequest(transfer=transfer), timeout=timeout
            )
            if tuple(spec.shape) != shape:
                raise ValueError(
                    "NIXL embedding shape differs from staged image layout/model"
                )
            if spec.dtype not in ("bfloat16", "float16", "float32"):
                raise ValueError(f"unsupported staged embedding dtype: {spec.dtype}")
            dtype = getattr(torch, spec.dtype)
            nbytes = shape[0] * shape[1] * dtype.itemsize
            target = receiver.allocate(nbytes)
            try:
                tensor = target.tensor.view(dtype).view(shape)
                destination = encoder_pb2.DeliverRequest(
                    transfer=transfer, nixl_target=target.descriptor
                )
            except BaseException:
                target.close()
                raise
            try:
                await stub.Deliver(
                    destination,
                    timeout=max(0, deadline - asyncio.get_running_loop().time()),
                )
            except BaseException:
                # A lost Deliver response can leave E writing this memory. Release
                # retires the transfer and fences any writer that already claimed it.
                fenced = False
                try:
                    await stub.Release(
                        encoder_pb2.ReleaseRequest(transfer_id=self.transfer_id),
                        timeout=timeout,
                    )
                    fenced = True
                finally:
                    if fenced:
                        target.close()
                    else:
                        target.abandon()
                        task = asyncio.create_task(
                            self._refence_target(target, timeout)
                        )
                        _fence_tasks.add(task)
                        task.add_done_callback(_fence_tasks.discard)
                raise
            target.close()
            return tensor

    async def _refence_target(self, target, timeout: float):
        import grpc

        delay = 0.25
        while not target.reaped():
            await asyncio.sleep(delay)
            if target.reaped():
                return
            try:
                await self.release(timeout)
            except grpc.RpcError:
                delay = min(delay * 2, 30)
            else:
                target.close()
                return

    async def release(self, timeout: float):
        import grpc
        from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc

        async with grpc.aio.insecure_channel(urlsplit(self.peer).netloc) as channel:
            stub = encoder_pb2_grpc.SglangEncoderStub(channel)
            await stub.Release(
                encoder_pb2.ReleaseRequest(transfer_id=self.transfer_id),
                timeout=timeout,
            )
