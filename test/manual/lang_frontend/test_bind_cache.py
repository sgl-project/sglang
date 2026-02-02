import unittest

import sglang as sgl
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase


class TestBind(CustomTestCase):
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.backend = sgl.Runtime(model_path=DEFAULT_MODEL_NAME_FOR_TEST)
        sgl.set_default_backend(cls.backend)

    @classmethod
    def tearDownClass(cls):
        cls.backend.shutdown()

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

    def test_cache(self):
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
        few_shot_qa_2.cache(self.backend)


if __name__ == "__main__":
    unittest.main()
