"""Exercise the actual routed-turn coroutine against an HTTP stub."""

import asyncio
import dataclasses
import unittest

from aiohttp import web

from sglang.srt.entrypoints.openai.pd_responses import PDResponsesError, routed_turn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclasses.dataclass
class Turn:
    rid: str = "resp_owner"
    input_ids: list = dataclasses.field(default_factory=lambda: [10, 20, 30])
    sampling_params: dict = dataclasses.field(default_factory=lambda: {"top_p": 0.95})
    stream: bool = False
    background: bool = False
    bootstrap_host: str = "stale"
    bootstrap_port: int = 8998
    bootstrap_room: int = 7


class RoutedResponsesTest(unittest.IsolatedAsyncioTestCase):
    async def start(self, handler):
        app = web.Application()
        app.router.add_post("/generate", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.addAsyncCleanup(runner.cleanup)
        return f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def test_expanded_payload_and_split_events(self):
        seen = []

        async def handler(request):
            seen.append(await request.json())
            response = web.StreamResponse()
            await response.prepare(request)
            for part in (
                b'data: {"text":"',
                b'ok","meta_info":{}}\n\n',
                b"data: [DONE]\n\n",
            ):
                await response.write(part)
            return response

        url = await self.start(handler)
        turn = Turn()
        outputs = [x async for x in routed_turn(turn, url)]
        self.assertEqual(outputs[0]["text"], "ok")
        self.assertEqual(seen[0]["input_ids"], turn.input_ids)
        self.assertEqual(seen[0]["sampling_params"], turn.sampling_params)
        self.assertNotEqual(seen[0]["rid"], turn.rid)
        self.assertNotIn("bootstrap_host", seen[0])
        self.assertTrue(seen[0]["stream"])
        self.assertEqual(turn.bootstrap_host, "stale")

    async def test_nonstream_only_yields_final_cumulative_output(self):
        async def handler(request):
            return web.Response(
                text=(
                    'data: {"text":"a","output_ids":[1]}\n\n'
                    'data: {"text":"ab","output_ids":[1,2]}\n\n'
                    "data: [DONE]\n\n"
                )
            )

        url = await self.start(handler)
        outputs = [x async for x in routed_turn(Turn(), url)]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["output_ids"], [1, 2])

    async def test_error_and_incomplete_stream(self):
        for status, body in (
            (503, "busy"),
            (200, 'data: {"text":"partial"}\n\n'),
            (200, 'data: {"error":{}}\n\n'),
        ):

            async def handler(request):
                return web.Response(status=status, text=body)

            url = await self.start(handler)
            with self.assertRaises(PDResponsesError):
                _ = [x async for x in routed_turn(Turn(), url)]

    async def test_frontend_disconnect_cancels_waiting_turn(self):
        arrived = asyncio.Event()
        disconnected = asyncio.Event()

        async def handler(request):
            response = web.StreamResponse()
            await response.prepare(request)
            arrived.set()
            try:
                while True:
                    await asyncio.sleep(0.01)
                    await response.write(b": heartbeat\n\n")
            except (ConnectionResetError, asyncio.CancelledError):
                disconnected.set()
            return response

        class Client:
            async def is_disconnected(self):
                return arrived.is_set()

        url = await self.start(handler)

        async def consume():
            return [x async for x in routed_turn(Turn(), url, Client())]

        task = asyncio.create_task(consume())
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        await asyncio.wait_for(disconnected.wait(), 2)

    async def test_background_turn_ignores_original_client_disconnect(self):
        class Client:
            async def is_disconnected(self):
                raise AssertionError("Background work must outlive original client")

        async def handler(request):
            return web.Response(text='data: {"text":"ok"}\n\ndata: [DONE]\n\n')

        url = await self.start(handler)
        outputs = [x async for x in routed_turn(Turn(background=True), url, Client())]
        self.assertEqual(outputs[0]["text"], "ok")

    async def test_close_disconnects_upstream(self):
        disconnected = asyncio.Event()

        async def handler(request):
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b'data: {"text":"start"}\n\n')
            try:
                while True:
                    await asyncio.sleep(0.01)
                    await response.write(b": heartbeat\n\n")
            except (ConnectionResetError, asyncio.CancelledError):
                disconnected.set()
            return response

        url = await self.start(handler)
        generator = routed_turn(Turn(stream=True), url)
        await anext(generator)
        await generator.aclose()
        await asyncio.wait_for(disconnected.wait(), 2)


class CoordinatorConfigurationTest(unittest.TestCase):
    def test_router_origin_validation(self):
        from sglang.srt.arg_groups.validation_hook import validate_response_store
        from sglang.srt.server_args import ServerArgs

        for value in (
            "file:///tmp/router",
            "http://host/path",
            "http://u:p@host",
            "http://host?x=1",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_response_store(
                    ServerArgs(model_path="dummy", responses_generation_url=value)
                )

        args = ServerArgs(
            model_path="dummy", responses_generation_url="http://router:30000"
        )
        validate_response_store(args)
        self.assertEqual(args.responses_generation_url, "http://router:30000")

    def test_media_preprocessing_options_are_preserved(self):
        from sglang.srt.entrypoints.openai.pd_responses import _media_reference_json
        from sglang.srt.utils import ImageData

        image = ImageData(
            url="data:image/png;base64,AA==",
            detail="high",
            preprocess_kwargs={"max_pixels": 4096},
        )
        self.assertEqual(_media_reference_json([image]), [dataclasses.asdict(image)])


if __name__ == "__main__":
    unittest.main()
