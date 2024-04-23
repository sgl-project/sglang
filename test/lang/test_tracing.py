import unittest

from sglang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template

import sglang as sgl


class TestTracing(unittest.TestCase):
    def test_few_shot_qa(self):
        @sgl.function
        def few_shot_qa(s, question):
            s += "The following are questions with answers.\n\n"
            s += "Q: What is the capital of France?\n"
            s += "A: Paris\n"
            s += "Q: " + question + "\n"
            s += "A:" + sgl.gen("answer", stop="\n")

        tracer = few_shot_qa.trace()
        print(tracer.last_node.print_graph_dfs() + "\n")

    def test_select(self):
        @sgl.function
        def capital(s):
            s += "The capital of France is"
            s += sgl.select("capital", ["Paris. ", "London. "])
            s += "It is a city" + sgl.gen("description", stop=".")

        tracer = capital.trace()
        print(tracer.last_node.print_graph_dfs() + "\n")

    def test_raise_warning(self):
        @sgl.function
        def wrong(s, question):
            s += f"I want to ask {question}"

        try:
            tracer = wrong.trace()
            raised = False
        except TypeError:
            raised = True

        assert raised

    def test_multi_function(self):
        @sgl.function
        def expand(s, tip):
            s += (
                "Please expand the following tip into a detailed paragraph:"
                + tip
                + "\n"
            )
            s += sgl.gen("detailed_tip")

        @sgl.function
        def tip_suggestion(s, topic):
            s += "Here are 2 tips for " + topic + ".\n"

            s += "1." + sgl.gen("tip_1", stop=["\n", ":", "."]) + "\n"
            s += "2." + sgl.gen("tip_2", stop=["\n", ":", "."]) + "\n"

            branch1 = expand(tip=s["tip_1"])
            branch2 = expand(tip=s["tip_2"])

            s += "Tip 1: " + branch1["detailed_tip"] + "\n"
            s += "Tip 2: " + branch2["detailed_tip"] + "\n"
            s += "In summary" + sgl.gen("summary")

        compiled = tip_suggestion.compile()
        compiled.print_graph()

        sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo-instruct"))
        state = compiled.run(topic="staying healthy")
        print(state.text() + "\n")

        states = compiled.run_batch(
            [
                {"topic": "staying healthy"},
                {"topic": "staying happy"},
                {"topic": "earning money"},
            ],
            temperature=0,
        )
        for s in states:
            print(s.text() + "\n")

    def test_role(self):
        @sgl.function
        def multi_turn_chat(s):
            s += sgl.user("Who are you?")
            s += sgl.assistant(sgl.gen("answer_1"))
            s += sgl.user("Who created you?")
            s += sgl.assistant(sgl.gen("answer_2"))

        backend = BaseBackend()
        backend.chat_template = get_chat_template("llama-2-chat")

        compiled = multi_turn_chat.compile(backend=backend)
        compiled.print_graph()

    def test_fork(self):
        @sgl.function
        def tip_suggestion(s):
            s += (
                "Here are three tips for staying healthy: "
                "1. Balanced Diet; "
                "2. Regular Exercise; "
                "3. Adequate Sleep\n"
            )

            forks = s.fork(3)
            for i in range(3):
                forks[i] += f"Now, expand tip {i+1} into a paragraph:\n"
                forks[i] += sgl.gen(f"detailed_tip")

            s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
            s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
            s += "Tip 3:" + forks[2]["detailed_tip"] + "\n"
            s += "In summary" + sgl.gen("summary")

        tracer = tip_suggestion.trace()
        print(tracer.last_node.print_graph_dfs())

        a = tip_suggestion.run(backend=sgl.OpenAI("gpt-3.5-turbo-instruct"))
        print(a.text())


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # t = TestTracing()
    # t.test_multi_function()
