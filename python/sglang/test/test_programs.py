"""
This file contains the SGL programs used for unit testing.
"""

import json
import re

import sglang as sgl


def test_few_shot_qa():
    @sgl.function
    def few_shot_qa(s, question):
        s += "The following are questions with answers.\n\n"
        s += "Q: What is the capital of France?\n"
        s += "A: Paris\n"
        s += "Q: What is the capital of Germany?\n"
        s += "A: Berlin\n"
        s += "Q: What is the capital of Italy?\n"
        s += "A: Rome\n"
        s += "Q: " + question + "\n"
        s += "A:" + sgl.gen("answer", stop="\n", temperature=0)

    ret = few_shot_qa.run(question="What is the capital of the United States?")
    assert "washington" in ret["answer"].strip().lower(), f"answer: {ret['answer']}"

    rets = few_shot_qa.run_batch(
        [
            {"question": "What is the capital of Japan?"},
            {"question": "What is the capital of the United Kingdom?"},
            {"question": "What is the capital city of China?"},
        ],
        temperature=0.1,
    )
    answers = [x["answer"].strip().lower() for x in rets]
    assert answers == ["tokyo", "london", "beijing"], f"answers: {answers}"


def test_mt_bench():
    @sgl.function
    def answer_mt_bench(s, question_1, question_2):
        s += sgl.system("You are a helpful assistant.")
        s += sgl.user(question_1)
        s += sgl.assistant(sgl.gen("answer_1"))
        with s.user():
            s += question_2
        with s.assistant():
            s += sgl.gen("answer_2")

    question_1 = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
    question_2 = (
        "Rewrite your previous response. Start every sentence with the letter A."
    )
    ret = answer_mt_bench.run(
        question_1=question_1, question_2=question_2, temperature=0.7, max_new_tokens=64
    )
    assert len(ret.messages()) in [4, 5]


def test_select(check_answer):
    @sgl.function
    def true_or_false(s, statement):
        s += "Determine whether the statement below is True, False, or Unknown.\n"
        s += "Statement: The capital of France is Pairs.\n"
        s += "Answer: True\n"
        s += "Statement: " + statement + "\n"
        s += "Answer:" + sgl.select("answer", ["True", "False", "Unknown"])

    ret = true_or_false.run(
        statement="The capital of Germany is Berlin.",
    )
    if check_answer:
        assert ret["answer"] == "True", ret.text
    else:
        assert ret["answer"] in ["True", "False", "Unknown"]

    ret = true_or_false.run(
        statement="The capital of Canada is Tokyo.",
    )
    if check_answer:
        assert ret["answer"] == "False", ret.text
    else:
        assert ret["answer"] in ["True", "False", "Unknown"]

    ret = true_or_false.run(
        statement="Purple is a better color than green.",
    )
    if check_answer:
        assert ret["answer"] == "Unknown", ret.text
    else:
        assert ret["answer"] in ["True", "False", "Unknown"]


def test_decode_int():
    @sgl.function
    def decode_int(s):
        s += "The number of hours in a day is " + sgl.gen_int("hours") + "\n"
        s += "The number of days in a year is " + sgl.gen_int("days") + "\n"

    ret = decode_int.run(temperature=0.1)
    assert int(ret["hours"]) == 24, ret.text
    assert int(ret["days"]) == 365, ret.text


def test_decode_json_regex():
    @sgl.function
    def decode_json(s):
        from sglang.lang.ir import REGEX_FLOAT, REGEX_INT, REGEX_STRING

        s += "Generate a JSON object to describe the basic information of a city.\n"

        with s.var_scope("json_output"):
            s += "{\n"
            s += '  "name": ' + sgl.gen(regex=REGEX_STRING + ",") + "\n"
            s += '  "population": ' + sgl.gen(regex=REGEX_INT + ",") + "\n"
            s += '  "area": ' + sgl.gen(regex=REGEX_INT + ",") + "\n"
            s += '  "latitude": ' + sgl.gen(regex=REGEX_FLOAT + ",") + "\n"
            s += '  "country": ' + sgl.gen(regex=REGEX_STRING + ",") + "\n"
            s += '  "timezone": ' + sgl.gen(regex=REGEX_STRING) + "\n"
            s += "}"

    ret = decode_json.run()
    js_obj = json.loads(ret["json_output"])
    assert isinstance(js_obj["name"], str)
    assert isinstance(js_obj["population"], int)


def test_decode_json():
    @sgl.function
    def decode_json(s):
        s += "Generate a JSON object to describe the basic information of a city.\n"

        with s.var_scope("json_output"):
            s += "{\n"
            s += '  "name": ' + sgl.gen_string() + ",\n"
            s += '  "population": ' + sgl.gen_int() + ",\n"
            s += '  "area": ' + sgl.gen(dtype=int) + ",\n"
            s += '  "country": ' + sgl.gen_string() + ",\n"
            s += '  "timezone": ' + sgl.gen(dtype=str) + "\n"
            s += "}"

    ret = decode_json.run()
    js_obj = json.loads(ret["json_output"])
    assert isinstance(js_obj["name"], str)
    assert isinstance(js_obj["population"], int)


def test_expert_answer():
    @sgl.function
    def expert_answer(s, question):
        s += "Question: " + question + "\n"
        s += (
            "A good person to answer this question is"
            + sgl.gen("expert", stop=[".", "\n"])
            + ".\n"
        )
        s += (
            "For example,"
            + s["expert"]
            + " would answer that "
            + sgl.gen("answer", stop=".")
            + "."
        )

    ret = expert_answer.run(question="What is the capital of France?", temperature=0.1)
    assert "paris" in ret.text().lower()


def test_tool_use():
    def calculate(expression):
        return f"{eval(expression)}"

    @sgl.function
    def tool_use(s, lhs, rhs):
        s += "Please perform computations using a calculator. You can use calculate(expression) to get the results.\n"
        s += "For example,\ncalculate(1+2)=3\ncalculate(3*4)=12\n"
        s += "Question: What is the product of " + str(lhs) + " and " + str(rhs) + "?\n"
        s += (
            "Answer: The answer is calculate("
            + sgl.gen("expression", stop=")")
            + ") = "
        )
        with s.var_scope("answer"):
            s += calculate(s["expression"])

    lhs, rhs = 257, 983
    ret = tool_use(lhs=lhs, rhs=rhs, temperature=0)
    assert int(ret["answer"]) == lhs * rhs


def test_react():
    @sgl.function
    def react(s, question):
        s += """
Question: Which country does the founder of Microsoft live in?
Thought 1: I need to search for the founder of Microsoft.
Action 1: Search [Founder of Microsoft].
Observation 1: The founder of Microsoft is Bill Gates.
Thought 2: I need to search for the country where Bill Gates lives in.
Action 2: Search [Where does Bill Gates live].
Observation 2: Bill Gates lives in the United States.
Thought 3: The answer is the United States.
Action 3: Finish [United States].\n
"""

        s += "Question: " + question + "\n"

        for i in range(1, 5):
            s += f"Thought {i}:" + sgl.gen(stop=[".", "\n"]) + ".\n"
            s += f"Action {i}: " + sgl.select(f"action_{i}", ["Search", "Finish"])

            if s[f"action_{i}"] == "Search":
                s += " [" + sgl.gen(stop="]") + "].\n"
                s += f"Observation {i}:" + sgl.gen(stop=[".", "\n"]) + ".\n"
            else:
                s += " [" + sgl.gen("answer", stop="]") + "].\n"
                break

    ret = react.run(
        question="What country does the creator of Linux live in?",
        temperature=0.1,
    )
    answer = ret["answer"].lower()
    assert "finland" in answer or "states" in answer


def test_parallel_decoding():
    max_tokens = 64
    number = 5

    @sgl.function
    def parallel_decoding(s, topic):
        s += "Act as a helpful assistant.\n"
        s += "USER: Give some tips for " + topic + ".\n"
        s += (
            "ASSISTANT: Okay. Here are "
            + str(number)
            + " concise tips, each under 8 words:\n"
        )

        # Generate skeleton
        for i in range(1, 1 + number):
            s += f"{i}." + sgl.gen(max_tokens=16, stop=[".", "\n"]) + ".\n"

        # Generate detailed tips
        forks = s.fork(number)
        for i in range(number):
            forks[
                i
            ] += f"Now, I expand tip {i+1} into a detailed paragraph:\nTip {i+1}:"
            forks[i] += sgl.gen("detailed_tip", max_tokens, stop=["\n\n"])
        forks.join()

        # Concatenate tips and summarize
        s += "Here are these tips with detailed explanation:\n"
        for i in range(number):
            s += f"Tip {i+1}:" + forks[i]["detailed_tip"] + "\n"

        s += "\nIn summary," + sgl.gen("summary", max_tokens=512)

    ret = parallel_decoding.run(topic="writing a good blog post", temperature=0.3)


def test_parallel_encoding(check_answer=True):
    max_tokens = 64

    @sgl.function
    def parallel_encoding(s, question, context_0, context_1, context_2):
        s += "USER: I will ask a question based on some statements.\n"
        s += "ASSISTANT: Sure. I will give the answer.\n"
        s += "USER: Please memorize these statements.\n"

        contexts = [context_0, context_1, context_2]

        forks = s.fork(len(contexts))
        forks += lambda i: f"Statement {i}: " + contexts[i] + "\n"
        forks.join(mode="concate_and_append")

        s += "Now, please answer the following question. " "Do not list options."
        s += "\nQuestion: " + question + "\n"
        s += "ASSISTANT:" + sgl.gen("answer", max_tokens=max_tokens)

    ret = parallel_encoding.run(
        question="Who is the father of Julian?",
        context_0="Ethan is the father of Liam.",
        context_1="Noah is the father of Julian.",
        context_2="Oliver is the father of Carlos.",
        temperature=0,
    )
    answer = ret["answer"]

    if check_answer:
        assert "Noah" in answer


def test_image_qa():
    @sgl.function
    def image_qa(s, question):
        s += sgl.user(sgl.image("test_image.png") + question)
        s += sgl.assistant(sgl.gen("answer"))

    state = image_qa.run(
        question="Please describe this image in simple words.",
        temperature=0,
        max_new_tokens=64,
    )
    assert (
        "taxi" in state.messages()[-1]["content"]
        or "car" in state.messages()[-1]["content"]
    )


def test_stream():
    @sgl.function
    def qa(s, question):
        s += sgl.user(question)
        s += sgl.assistant(sgl.gen("answer"))

    ret = qa(
        question="Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        stream=True,
    )
    out = ""
    for chunk in ret.text_iter():
        out += chunk

    ret = qa(
        question="Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        stream=True,
    )
    out = ""
    for chunk in ret.text_iter("answer"):
        out += chunk


def test_regex():
    regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"

    @sgl.function
    def regex_gen(s):
        s += "Q: What is the IP address of the Google DNS servers?\n"
        s += "A: " + sgl.gen(
            "answer",
            temperature=0,
            regex=regex,
        )

    state = regex_gen.run()
    answer = state["answer"]
    assert re.match(regex, answer)
