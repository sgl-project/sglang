import unittest

from sglang.backend.runtime_endpoint import RuntimeEndpoint

import sglang as sgl


class TestBind(unittest.TestCase):
    backend = None

    def setUp(self):
        cls = type(self)

        if cls.backend is None:
            cls.backend = RuntimeEndpoint(base_url="http://localhost:30000")

    def test_bind(self):
        @sgl.function
        def few_shot_qa(s, prompt, question):
            s += prompt
            s += "Q: What is the capital of France?\n"
            s += "A: Paris\n"
            s += "Q: " + question + "\n"
            s += "A:" + sgl.gen("answer", stop="\n")

        few_shot_qa_2 = few_shot_qa.bind(
            prompt="The following are questions with answers.\n\n"
        )

        tracer = few_shot_qa_2.trace()
        print(tracer.last_node.print_graph_dfs() + "\n")

    def test_pin(self):
        @sgl.function
        def few_shot_qa(s, prompt, question):
            s += prompt
            s += "Q: What is the capital of France?\n"
            s += "A: Paris\n"
            s += "Q: " + question + "\n"
            s += "A:" + sgl.gen("answer", stop="\n")

        few_shot_qa_2 = few_shot_qa.bind(
            prompt="Answer the following questions as if you were a 5-year-old kid.\n\n"
        )
        few_shot_qa_2.pin(self.backend)
        few_shot_qa_2.unpin(self.backend)


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # t = TestBind()
    # t.setUp()
    # t.test_pin()
