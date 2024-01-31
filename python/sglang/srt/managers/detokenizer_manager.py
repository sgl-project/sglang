import asyncio

import uvloop
import zmq
import zmq.asyncio
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import BatchStrOut, BatchTokenIDOut
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class DetokenizerManager:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        context = zmq.asyncio.Context(2)
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{port_args.detokenizer_port}")

        self.send_to_tokenizer = context.socket(zmq.PUSH)
        self.send_to_tokenizer.connect(f"tcp://127.0.0.1:{port_args.tokenizer_port}")

        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )

    async def handle_loop(self):
        while True:
            recv_obj = await self.recv_from_router.recv_pyobj()

            if isinstance(recv_obj, BatchTokenIDOut):
                output_tokens = recv_obj.output_tokens

                # TODO(lmzheng): handle skip_special_tokens per request
                output_strs = self.tokenizer.batch_decode(
                    output_tokens,
                    skip_special_tokens=recv_obj.skip_special_tokens[0],
                )

                # Trim stop str
                # TODO(lmzheng): handle the case where multiple stop strs are hit
                for i in range(len(output_strs)):
                    if recv_obj.hit_stop_str[i] is not None:
                        pos = output_strs[i].find(recv_obj.hit_stop_str[i])
                        if pos != -1:
                            output_strs[i] = output_strs[i][:pos]

                    if len(output_tokens[i]) > 0:
                        first_token = self.tokenizer.convert_ids_to_tokens(
                            int(output_tokens[i][0])
                        )
                        if not isinstance(first_token, str):
                            first_token = first_token.decode("utf-8", errors="ignore")
                        if first_token.startswith("▁"):
                            output_strs[i] = " " + output_strs[i]

                    output_strs[i] = (
                        recv_obj.output_and_fast_forward_strs[i] + output_strs[i]
                    )

                self.send_to_tokenizer.send_pyobj(
                    BatchStrOut(
                        recv_obj.rids,
                        output_strs,
                        recv_obj.meta_info,
                        recv_obj.finished,
                    )
                )
            else:
                raise ValueError(f"Invalid object: {recv_obj}")


def start_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    try:
        manager = DetokenizerManager(server_args, port_args)
    except Exception as e:
        pipe_writer.send(get_exception_traceback())
        raise
    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(manager.handle_loop())
