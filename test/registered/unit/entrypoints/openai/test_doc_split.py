"""Regression tests for server-side Unlimited-OCR document splitting."""

import asyncio
import itertools
import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai import doc_split, serving_chat_doc_split
from sglang.srt.entrypoints.openai.doc_kv_state import DocumentKVState
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.runtime_context import get_context
from sglang.srt.session.document_session import DocumentSessionSlot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _plan_batches(
    *,
    n_pages,
    image_mode,
    context_length=10**6,
    max_completion_tokens=1,
    max_batch_pages=None,
    overlap_pages=1,
):
    return doc_split.plan_batches(
        n_pages=n_pages,
        image_mode=image_mode,
        context_length=context_length,
        max_completion_tokens=max_completion_tokens,
        max_batch_pages=max_batch_pages,
        overlap_pages=overlap_pages,
    )


class TestDocSplit(CustomTestCase):
    def test_batches_overlap_by_one_page(self):
        # 964 - reserve 64 - one completion token fits three 274-token pages.
        batches = _plan_batches(n_pages=5, image_mode="base", context_length=964)
        self.assertEqual(batches, [(0, 3, 0), (2, 5, 3)])

    def test_batches_overlap_is_configurable(self):
        batches = _plan_batches(
            n_pages=8,
            image_mode="base",
            context_length=10**6,
            max_batch_pages=4,
            overlap_pages=2,
        )
        self.assertEqual(batches, [(0, 4, 0), (2, 6, 4), (4, 8, 6)])

    def test_zero_overlap_is_contiguous(self):
        batches = _plan_batches(
            n_pages=8,
            image_mode="base",
            context_length=10**6,
            max_batch_pages=4,
            overlap_pages=0,
        )
        self.assertEqual(batches, [(0, 4, 0), (4, 8, 4)])

    def test_overlap_must_leave_progress(self):
        self.assertIsNone(
            _plan_batches(
                n_pages=8,
                image_mode="base",
                context_length=10**6,
                max_batch_pages=2,
                overlap_pages=2,
            )
        )

    def test_a_document_that_fits_is_not_split(self):
        self.assertIsNone(
            _plan_batches(n_pages=5, image_mode="base", context_length=10**6)
        )

    def test_page_cap_forces_a_split_that_the_budget_would_allow(self):
        self.assertIsNone(
            _plan_batches(
                n_pages=4, image_mode="base", context_length=10**6, max_batch_pages=4
            )
        )
        self.assertEqual(
            _plan_batches(
                n_pages=5, image_mode="base", context_length=10**6, max_batch_pages=4
            ),
            [(0, 4, 0), (3, 5, 4)],
        )

    def test_no_room_for_a_context_page_means_no_split(self):
        # One page plus its context page must fit, or splitting cannot help.
        self.assertIsNone(
            _plan_batches(n_pages=3, image_mode="base", context_length=400)
        )

    def test_a_long_gundam_document_splits_at_the_page_limit(self):
        """Windows are sized by input pages; a large output budget does not
        shrink them below the page limit."""
        self.assertEqual(
            _plan_batches(
                n_pages=17,
                image_mode="gundam",
                context_length=131072,
                max_completion_tokens=126496,
                max_batch_pages=10,
            ),
            [(0, 10, 0), (9, 17, 10)],
        )

    def test_input_window_ignores_global_output_budget(self):
        expected = [(0, 4, 0), (3, 7, 4), (6, 10, 7)]
        for budget in (1, 126496, 10**6):
            with self.subTest(budget=budget):
                self.assertEqual(
                    _plan_batches(
                        n_pages=10,
                        image_mode="base",
                        context_length=32768,
                        max_completion_tokens=budget,
                        max_batch_pages=4,
                    ),
                    expected,
                )

    def test_small_context_splits_below_the_policy_page_limit(self):
        self.assertEqual(
            _plan_batches(
                n_pages=5,
                image_mode="gundam",
                context_length=4096,
                max_completion_tokens=126496,
                max_batch_pages=10,
            ),
            [(0, 2, 0), (1, 3, 2), (2, 4, 3), (3, 5, 4)],
        )

    def test_select_batch_output_drops_the_overlap_page(self):
        owned, observed, expected = doc_split.select_batch_output(
            text="<PAGE>context<PAGE>owned<PAGE>extra",
            input_pages=2,
            context_pages=1,
        )
        self.assertEqual(owned, "<PAGE>owned")
        self.assertEqual((observed, expected), (3, 2))

    def test_blank_first_overlap_page_keeps_owned_pages_aligned(self):
        """A blank first overlap page must not shift ownership and drop an owned page."""
        owned, observed, expected = doc_split.select_batch_output(
            text="<PAGE><PAGE>owned1<PAGE>owned2", input_pages=3, context_pages=1
        )
        self.assertEqual(owned, "<PAGE>owned1<PAGE>owned2")
        self.assertEqual((observed, expected), (3, 3))

    def test_a_reused_window_keeps_a_blank_first_page(self):
        """A reused window's first marker is spliced in and hidden, so its answer
        opens with a body; dropping an empty one would shift every page up."""
        owned, observed, expected = doc_split.select_batch_output(
            text="<PAGE>owned2<PAGE>owned3",
            input_pages=3,
            context_pages=0,
            leads_with_body=True,
        )
        self.assertEqual(owned, "<PAGE><PAGE>owned2<PAGE>owned3")
        self.assertEqual((observed, expected), (3, 3))
        # A non-blank first page is unaffected.
        owned, observed, expected = doc_split.select_batch_output(
            text="owned1<PAGE>owned2",
            input_pages=2,
            context_pages=0,
            leads_with_body=True,
        )
        self.assertEqual(owned, "<PAGE>owned1<PAGE>owned2")
        self.assertEqual((observed, expected), (2, 2))

    def test_ten_pages_survive_three_batches_exactly_once(self):
        pages = [f"page-{i}-body" for i in range(10)]
        plan = _plan_batches(
            n_pages=10, image_mode="base", context_length=20000, max_batch_pages=4
        )
        self.assertEqual(plan, [(0, 4, 0), (3, 7, 4), (6, 10, 7)])
        chunks = []
        for lo, hi, emit_from in plan:
            text = "<PAGE>" + "<PAGE>".join(pages[lo:hi])
            owned, _, _ = doc_split.select_batch_output(
                text=text, input_pages=hi - lo, context_pages=emit_from - lo
            )
            chunks.append(owned)
        self.assertEqual("".join(chunks).split("<PAGE>")[1:], pages)

    def test_page_introduced_before_its_marker_is_counted(self):
        """The model can omit the marker that opens its first page."""
        owned, observed, expected = doc_split.select_batch_output(
            text="overlap<PAGE>owned", input_pages=2, context_pages=1
        )
        self.assertEqual(owned, "<PAGE>owned")
        self.assertEqual((observed, expected), (2, 2))

    def test_empty_body_is_visible_to_the_caller(self):
        """Marker counting alone cannot see a page the model skipped."""
        owned, observed, expected = doc_split.select_batch_output(
            text="first<PAGE><PAGE>last", input_pages=3, context_pages=1
        )
        self.assertEqual(len(doc_split.owned_page_bodies(owned)), 2)
        self.assertEqual((observed, expected), (3, 3))

    def test_extra_page_changes_the_body_count(self):
        """A repeated first page or over-generation must be visible to the caller."""
        for text in ("p3<PAGE>p3<PAGE>p4", "p3<PAGE>p4<PAGE>extra"):
            with self.subTest(text=text):
                _, observed, expected = doc_split.select_batch_output(
                    text=text, input_pages=2, context_pages=1
                )
                self.assertEqual((observed, expected), (3, 2))

    def test_invalid_page_range_is_rejected(self):
        for input_pages, context_pages in ((0, 0), (2, 2), (2, -1)):
            with self.subTest(input_pages=input_pages, context_pages=context_pages):
                with self.assertRaises(ValueError):
                    doc_split.select_batch_output(
                        text="x", input_pages=input_pages, context_pages=context_pages
                    )


class TestStreamOwnedText(CustomTestCase):
    """Streamed text cannot be taken back, so every prefix of a window's output
    must own a prefix of what the finished window owns."""

    CASES = (
        # (output, input_pages, context_pages, leads_with_body)
        ("<PAGE>ctx<PAGE>a1<PAGE>b2<PAGE>extra", 3, 1, False),
        ("body1<PAGE>body2<PAGE>surplus", 2, 0, True),
        ("<PAGE>a<PAGE><PAGE>c", 3, 0, False),
    )

    def test_every_prefix_streams_a_prefix_of_the_final_answer(self):
        for text, pages, ctx, leads in self.CASES:
            with self.subTest(text=text):
                final, _, _ = doc_split.select_batch_output(
                    text=text,
                    input_pages=pages,
                    context_pages=ctx,
                    leads_with_body=leads,
                )
                sent = ""
                for n in range(len(text) + 1):
                    owned = doc_split.stream_owned_text(
                        text=text[:n],
                        input_pages=pages,
                        context_pages=ctx,
                        leads_with_body=leads,
                    )
                    self.assertTrue(owned.startswith(sent), (n, owned, sent))
                    self.assertTrue(final.startswith(owned), (n, owned, final))
                    sent = owned
                self.assertEqual(sent, final)

    def test_a_half_written_marker_is_held_back(self):
        """A partial <PAGE> streamed as body text would duplicate the marker."""
        owned = doc_split.stream_owned_text(
            text="<PAGE>a<PA", input_pages=2, context_pages=0, leads_with_body=False
        )
        self.assertEqual(owned, "<PAGE>a")


class TestWindowCompleteness(CustomTestCase):
    """Only missing pages fail a window.

    Masking the <PAGE> token once a window has opened all its pages makes a
    model that wants another page write the characters instead, so a surplus
    body is the guard's own artifact; the pages the window owns are all there.
    """

    def test_a_missing_page_fails_and_a_surplus_body_does_not(self):
        self.assertTrue(doc_split.window_incomplete(9, 10))
        self.assertFalse(doc_split.window_incomplete(10, 10))
        self.assertFalse(doc_split.window_incomplete(11, 10))


class TestPageCapParams(CustomTestCase):
    """The per-page cap is a server setting but rides in the request's
    custom_params, so these merge rules decide whether it applies at all."""

    ARGS = {
        "max_page_tokens": 8000,
        "page_token_id": 128827,
        "eos_token_id": 1,
        "custom_params": None,
    }

    def test_the_environment_threshold_reaches_the_sampler(self):
        self.assertEqual(
            doc_split.page_guard_params(**self.ARGS),
            {"max_page_tokens": 8000, "page_token_id": 128827, "eos_token_id": 1},
        )

    def test_a_request_keeps_the_threshold_it_asked_for(self):
        args = {**self.ARGS, "custom_params": {"max_page_tokens": 6000}}
        self.assertEqual(doc_split.page_guard_params(**args), {})

    def test_a_reuse_window_keeps_its_stop_without_a_threshold(self):
        """An instruction-less reuse window stops only through the guard, so a
        zero threshold must not drop the token ids it stops on."""
        args = {**self.ARGS, "max_page_tokens": 0, "stop_only": True}
        self.assertEqual(
            doc_split.page_guard_params(**args),
            {"page_token_id": 128827, "eos_token_id": 1, "page_stop_only": True},
        )

    def test_nothing_applies_without_a_threshold_or_the_token_ids(self):
        for key, value in (
            ("max_page_tokens", 0),
            ("page_token_id", None),
            ("eos_token_id", None),
        ):
            with self.subTest(**{key: value}):
                self.assertEqual(
                    doc_split.page_guard_params(**{**self.ARGS, key: value}), {}
                )

    def test_a_request_cannot_raise_the_server_threshold(self):
        """A client cap above the server's would let a window outgrow the
        context sized from the server's cap and write past its req_to_token row."""
        args = {**self.ARGS, "custom_params": {"max_page_tokens": 20000}}
        params = doc_split.page_guard_params(**args)
        self.assertEqual(params["max_page_tokens"], 8000)
        self.assertEqual(
            doc_split.effective_max_page_tokens(
                server_max_page_tokens=8000, custom_params={"max_page_tokens": 20000}
            ),
            8000,
        )

    def test_window_cap_follows_the_threshold_the_guard_enforces(self):
        """A lower client cap is what the guard applies; the window is sized by it."""
        page_tokens = doc_split.effective_max_page_tokens(
            server_max_page_tokens=20000, custom_params={"max_page_tokens": 8000}
        )
        self.assertEqual(page_tokens, 8000)
        self.assertEqual(
            doc_split.effective_max_page_tokens(
                server_max_page_tokens=0, custom_params={"max_page_tokens": 20000}
            ),
            20000,
        )
        for cp in (None, {}, {"max_page_tokens": 0}, {"ngram_size": 35}):
            with self.subTest(custom_params=cp):
                self.assertEqual(
                    doc_split.effective_max_page_tokens(
                        server_max_page_tokens=8000, custom_params=cp
                    ),
                    8000,
                )


class TestServerSidePageCap(CustomTestCase):
    """--uocr-max-page-tokens has to reach the sampler without the client
    asking: the cap only runs inside a CustomLogitProcessor."""

    def _window(self, cap, *, enabled=True, custom_params=None, processor=None):
        hf_config = SimpleNamespace(
            architectures=["UnlimitedOCRV2ForCausalLM"],
            sparse_attention_config={"page_token_id": 128827},
            eos_token_id=1,
        )
        sub = SimpleNamespace(
            custom_params=custom_params, custom_logit_processor=processor
        )
        with get_context().override_server_args(
            uocr_max_page_tokens=cap, enable_custom_logit_processor=enabled
        ):
            serving_chat_doc_split._apply_page_cap(sub, hf_config=hf_config)
        return sub

    def test_a_window_gets_the_cap_and_a_processor_to_run_it(self):
        sub = self._window(8000, custom_params={"uocr_doc_page_base": 3})
        self.assertEqual(
            sub.custom_params,
            {
                "uocr_doc_page_base": 3,
                "max_page_tokens": 8000,
                "page_token_id": 128827,
                "eos_token_id": 1,
                "ngram_output_only": True,
            },
        )
        self.assertTrue(sub.custom_logit_processor)

    def test_the_cap_is_off_by_default_and_without_the_server_flag(self):
        self.assertIsNone(self._window(0).custom_params)
        # Attaching a processor the server rejects would fail the whole request.
        self.assertIsNone(self._window(8000, enabled=False).custom_params)

    def test_a_request_that_brought_its_own_cap_is_left_alone(self):
        sub = self._window(
            8000, custom_params={"max_page_tokens": 6000}, processor="client-sent"
        )
        # Only the Unlimited-OCR v2 n-gram domain is added to its own processor.
        self.assertEqual(
            sub.custom_params, {"max_page_tokens": 6000, "ngram_output_only": True}
        )
        self.assertEqual(sub.custom_logit_processor, "client-sent")


class TestSplitterIsOnlyBuiltForUnlimitedOCRV2(CustomTestCase):
    """Chat serving routes every request through the splitter when it exists,
    so any other model must get none and keep the base request path."""

    def _splitter(self, architectures):
        serving = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=architectures)
                )
            )
        )
        return serving_chat_doc_split.ChatDocSplitter.for_model(
            serving, handle_window=None
        )

    def test_other_architectures_get_no_splitter(self):
        for architectures in (["LlamaForCausalLM"], ["UnlimitedOCRForCausalLM"], None):
            with self.subTest(architectures=architectures):
                self.assertIsNone(self._splitter(architectures))
        self.assertIsNotNone(self._splitter(["UnlimitedOCRV2ForCausalLM"]))


class TestOnePageWindows(CustomTestCase):
    def test_a_one_page_window_splits_every_page(self):
        plan = doc_split.plan_batches(
            n_pages=3,
            image_mode="base",
            context_length=10**6,
            max_completion_tokens=1000,
            max_batch_pages=1,
        )
        self.assertEqual(plan, [(0, 1, 0), (1, 2, 1), (2, 3, 2)])

    def test_overlap_still_needs_a_page_of_its_own(self):
        self.assertIsNone(
            doc_split.plan_batches(
                n_pages=3,
                image_mode="base",
                context_length=10**6,
                max_completion_tokens=1000,
                max_batch_pages=1,
                overlap_pages=1,
            )
        )


class TestWindowLookahead(CustomTestCase):
    """Every window but the last reads the next window's first page; the last
    own page of a window needs it to keep its cross-page markers."""

    def _messages(self, n):
        content = [
            {"type": "image_url", "image_url": {"url": f"img{i}"}} for i in range(n)
        ]
        content.append(
            {"type": "text", "text": "<PAGE><image>" * n + "Multi page parsing."}
        )
        return [{"role": "user", "content": content}]

    def _sub(self, plan, bi, n, lookahead=1):
        from sglang.srt.session import uocr_doc_reuse

        request = ChatCompletionRequest(
            model="m", messages=self._messages(n), max_tokens=1000
        )
        hf_config = SimpleNamespace(
            architectures=["UnlimitedOCRV2ForCausalLM"],
            sparse_attention_config={"page_token_id": 128827},
            eos_token_id=1,
        )
        with get_context().override_server_args(
            uocr_doc_lookahead_pages=lookahead, enable_custom_logit_processor=True
        ):
            sub = serving_chat_doc_split._window_request(
                request,
                hf_config=hf_config,
                plan=plan,
                bi=bi,
                max_tokens=100,
                doc_reuse_id="d",
            )
        images = [
            part.image_url.url
            for part in sub.messages[0].content
            if part.type == "image_url"
        ]
        return sub, images, sub.custom_params.get(uocr_doc_reuse.KEY_LOOKAHEAD)

    def test_windows_read_the_next_page_and_the_last_does_not(self):
        plan = [(0, 2, 0), (2, 4, 2), (4, 5, 4)]
        sub, images, ahead = self._sub(plan, 0, 5)
        self.assertEqual((images, ahead), (["img0", "img1", "img2"], 1))
        # The first window stops on the lookahead marker, not on its own EOS.
        self.assertTrue(sub.custom_params.get("page_stop_only"))
        _, images, ahead = self._sub(plan, 1, 5)
        self.assertEqual((images, ahead), (["img2", "img3", "img4"], 1))
        _, images, ahead = self._sub(plan, 2, 5)
        self.assertEqual((images, ahead), (["img4"], 0))

    def test_zero_lookahead_keeps_the_old_windows(self):
        sub, images, ahead = self._sub([(0, 2, 0), (2, 4, 2)], 0, 4, lookahead=0)
        self.assertEqual((images, ahead), (["img0", "img1"], 0))
        self.assertIsNone(sub.custom_params.get("page_stop_only"))


class TestDocumentKVState(CustomTestCase):
    def test_plan_must_have_configured_overlap(self):
        with self.assertRaises(ValueError):
            DocumentKVState.from_plan(
                page_count=8,
                overlap_pages=1,
                plan=((0, 4, 0), (2, 6, 4), (4, 8, 6)),
            )

    def test_document_session_slot_release_is_idempotent(self):
        slot = DocumentSessionSlot()
        self.assertTrue(slot.mark_released())
        self.assertFalse(slot.mark_released())


_HF_CONFIG = SimpleNamespace(
    architectures=["UnlimitedOCRV2ForCausalLM"],
    sparse_attention_config={"page_token_id": 128827},
    eos_token_id=1,
)


class _ScriptedWindows:
    """Answers each window of a split document with the next scripted text,
    as the unary chat handler and the tokenizer manager's stream would."""

    def __init__(self, answers):
        self._answers = list(answers)
        self._served = 0

    def _next_answer(self) -> str:
        answer = self._answers[self._served]
        self._served += 1
        return answer

    async def handle_window(self, sub, raw_request):
        answer = self._next_answer()
        return ChatCompletionResponse(
            id=sub.rid,
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=answer),
                    finish_reason="stop",
                    meta_info={},
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        )

    async def generate_request(self, adapted, raw_request):
        answer = self._next_answer()
        # Cumulative text, as the stream reports it without incremental output.
        cuts = sorted({len(answer) // 3, 2 * len(answer) // 3, len(answer)})
        for cut in cuts:
            done = cut == len(answer)
            yield {
                "text": answer[:cut],
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "finish_reason": {"type": "stop", "matched": 1} if done else None,
                },
            }


def _splitter(windows: _ScriptedWindows):
    tokenizer_manager = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_HF_CONFIG),
        max_req_input_len=10**6,
        rid_to_state={},
        generate_request=windows.generate_request,
        release_uocr_doc=lambda doc_id: None,
    )
    serving = SimpleNamespace(
        tokenizer_manager=tokenizer_manager,
        _validate_request=lambda request, **kwargs: None,
        create_error_response=lambda message: message,
        create_streaming_error_response=lambda message: message,
        _request_id_prefix=lambda: "chatcmpl-",
        _reported_prompt_tokens=lambda meta: meta["prompt_tokens"],
        _convert_to_internal_request=lambda sub, raw: (
            SimpleNamespace(rid=sub.rid),
            None,
        ),
    )
    return serving_chat_doc_split.ChatDocSplitter(
        serving, handle_window=windows.handle_window
    )


def _document_request(n_pages: int, *, stream: bool) -> ChatCompletionRequest:
    content = [
        {"type": "image_url", "image_url": {"url": f"img{i}"}} for i in range(n_pages)
    ]
    content.append(
        {"type": "text", "text": "<PAGE><image>" * n_pages + "Multi page parsing."}
    )
    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": content}],
        max_tokens=10**5,
        stream=stream,
    )


async def _serve_unary(plan, answers, n_pages):
    response = await _splitter(_ScriptedWindows(answers))._handle_split_document(
        _document_request(n_pages, stream=False), None, plan, doc_reuse_id=None
    )
    return (
        response.choices[0].message.content,
        response.choices[0].finish_reason,
        response.metadata,
    )


async def _serve_stream(plan, answers, n_pages):
    text, finish = "", None
    async for chunk in _splitter(_ScriptedWindows(answers))._split_document_stream(
        _document_request(n_pages, stream=True), None, plan, doc_reuse_id=None
    ):
        payload = chunk[len("data: ") :].strip()
        if payload == "[DONE]":
            break
        choice = json.loads(payload)["choices"][0]
        text += choice["delta"].get("content") or ""
        finish = choice.get("finish_reason") or finish
    return text, finish


def _answers(lo, hi, *, malformed):
    marker = doc_split.PAGE_MARKER
    clean = [f"p{i}" for i in range(lo, hi)]
    cases = [marker.join(clean)]
    if malformed:
        cases += [
            marker.join([clean[0] + "dup", *clean]),  # repeated first page
            marker.join([*clean, "extra"]),  # over-generation
            "stray" + marker + marker.join(clean),  # text before the first marker
            marker + marker.join(clean),  # one marker too many
            marker.join([clean[0], "", *clean[1:]]),  # a page with no body
        ]
    return cases


def _plans():
    for n_pages, cap in ((3, 2), (4, 2), (5, 3), (10, 4)):
        plan = _plan_batches(
            n_pages=n_pages,
            image_mode="base",
            context_length=10**6,
            max_batch_pages=cap,
        )
        yield n_pages, plan


class TestSplitDocumentAssembly(CustomTestCase):
    """Runs the server's window loop over scripted window answers."""

    MARKER = doc_split.PAGE_MARKER

    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_no_combination_returns_a_wrong_document_silently(self):
        """A wrong assembly must say so: not complete, or extra or empty pages."""

        async def silent_wrong_documents():
            silent = []
            for n_pages, plan in _plans():
                truth = [f"p{i}" for i in range(n_pages)]
                for answers in itertools.product(
                    *[_answers(lo, hi, malformed=True) for lo, hi, _ in plan]
                ):
                    text, _, meta = await _serve_unary(plan, answers, n_pages)
                    flagged = (
                        not meta["doc_split_complete"]
                        or any(meta["doc_split_batch_extra_pages"])
                        or meta["doc_split_empty_pages"]
                    )
                    if text.split(self.MARKER)[1:] != truth and not flagged:
                        silent.append((n_pages, answers))
            return silent

        self.assertEqual(asyncio.run(silent_wrong_documents()), [])

    def test_clean_answers_assemble_exactly(self):
        async def assembled():
            for n_pages, plan in _plans():
                answers = [_answers(lo, hi, malformed=False)[0] for lo, hi, _ in plan]
                yield n_pages, await _serve_unary(plan, answers, n_pages)

        async def check():
            async for n_pages, (text, finish, meta) in assembled():
                with self.subTest(n_pages=n_pages):
                    self.assertEqual(
                        text.split(self.MARKER)[1:], [f"p{i}" for i in range(n_pages)]
                    )
                    self.assertEqual(finish, "stop")
                    self.assertTrue(meta["doc_split_complete"])

        asyncio.run(check())

    def test_a_stream_carries_the_unary_answer(self):
        """Streamed text cannot be taken back, so each window's deltas must add
        up to exactly what the unary path assembles for the same answers."""

        async def mismatches():
            found = []
            for n_pages, plan in _plans():
                for answers in itertools.islice(
                    itertools.product(
                        *[_answers(lo, hi, malformed=True) for lo, hi, _ in plan]
                    ),
                    0,
                    None,
                    7,
                ):
                    unary_text, unary_finish, _ = await _serve_unary(
                        plan, answers, n_pages
                    )
                    stream = await _serve_stream(plan, answers, n_pages)
                    if stream != (unary_text, unary_finish or "stop"):
                        found.append((n_pages, answers, stream, unary_text))
            return found

        self.assertEqual(asyncio.run(mismatches()), [])


class TestPromptKeepsInlinePagePlaceholders(CustomTestCase):
    """Unlimited-OCR v2 prompts carry one "<PAGE><image>" per page in the text;
    every other server builds the prompt exactly as before."""

    def _user_turn(self, *, inline_image_placeholders):
        request = _document_request(2, stream=False)
        conv = generate_chat_conv(
            request,
            "unlimited-ocr",
            inline_image_placeholders=inline_image_placeholders,
        )
        return conv.messages[0][1]

    def test_the_v2_prompt_keeps_the_client_placeholders(self):
        self.assertEqual(
            self._user_turn(inline_image_placeholders=True),
            "<PAGE><image><PAGE><image>Multi page parsing.",
        )

    def test_other_servers_still_prepend_one_token_per_image(self):
        self.assertEqual(
            self._user_turn(inline_image_placeholders=False),
            "<image><image><PAGE><image><PAGE><image>Multi page parsing.",
        )


class TestPerModePageLimit(CustomTestCase):
    """Use separate caps for gundam and lighter image modes."""

    def test_mode_token_estimates_match_processor_presets(self):
        self.assertEqual(
            {
                mode: doc_split.tokens_per_page(mode)
                for mode in ("tiny", "small", "base", "large", "gundam")
            },
            {"tiny": 74, "small": 112, "base": 274, "large": 422, "gundam": 1514},
        )

    def test_mode_matching_ignores_surrounding_whitespace(self):
        """Lookups must match the processor's `.strip().lower()`, or "base " is
        budgeted and capped as gundam while running at base resolution."""
        for mode in ("base ", " base", "BASE ", "base\n"):
            with self.subTest(mode=mode):
                self.assertEqual(doc_split.tokens_per_page(mode), 274)
                self.assertEqual(doc_split.page_limit_for_mode(mode, 10, 40), 40)

    def test_absent_mode_takes_the_gundam_cap(self):
        """Absent modes use gundam for both token budgets and page caps."""
        self.assertEqual(doc_split.page_limit_for_mode(None, 10, 40), 10)
        self.assertEqual(doc_split.page_limit_for_mode("", 10, 40), 10)

    def test_unknown_mode_takes_the_gundam_cap(self):
        self.assertEqual(doc_split.page_limit_for_mode("weird", 10, 40), 10)

    def test_light_modes_take_the_base_cap(self):
        for mode in ("base", "large", "small", "tiny"):
            with self.subTest(mode=mode):
                self.assertEqual(doc_split.page_limit_for_mode(mode, 10, 40), 40)

    def test_unset_limits_stay_unset_so_the_budget_decides(self):
        self.assertIsNone(doc_split.page_limit_for_mode("gundam", None, None))
        self.assertIsNone(doc_split.page_limit_for_mode("base", None, None))
        self.assertIsNone(doc_split.page_limit_for_mode(None, None, None))

    def test_cap_and_token_paths_agree_on_which_mode_is_expensive(self):
        """Both paths must classify an absent mode the same way, or one request is
        budgeted as gundam and capped as base."""
        for mode in (None, "", "gundam", "base", "tiny", "weird"):
            with self.subTest(mode=mode):
                expensive = doc_split.tokens_per_page(mode) > doc_split.tokens_per_page(
                    "base"
                )
                takes_gundam_cap = doc_split.page_limit_for_mode(mode, 10, 40) == 10
                self.assertEqual(expensive, takes_gundam_cap)

    def test_server_cap_splits_a_document_that_would_have_fitted(self):
        # 4 base pages fit the budget, yet a cap of 2 must still split.
        budget = 10**6
        self.assertIsNone(
            _plan_batches(n_pages=4, image_mode="base", context_length=budget)
        )
        plan = _plan_batches(
            n_pages=4, image_mode="base", context_length=budget, max_batch_pages=2
        )
        self.assertEqual(plan, [(0, 2, 0), (1, 3, 2), (2, 4, 3)])

    def test_mm_processor_kwargs_mode_reaches_the_splitter(self):
        """The splitter must plan with the mode the processor runs, including
        vLLM's mm_processor_kwargs spelling, which images_config overrides."""
        cases = [
            ({"crop_mode": False}, None, "base"),
            ({"image_mode": "tiny"}, None, "tiny"),
            ({"crop_mode": False}, {"image_mode": "gundam"}, "gundam"),
            (None, None, None),
        ]
        for mm_processor_kwargs, images_config, mode in cases:
            with self.subTest(kwargs=mm_processor_kwargs, images_config=images_config):
                request = SimpleNamespace(
                    images_config=images_config,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                self.assertEqual(serving_chat_doc_split._doc_image_mode(request), mode)


if __name__ == "__main__":
    unittest.main()
