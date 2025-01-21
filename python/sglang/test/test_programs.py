"""This file contains the SGL programs used for unit testing."""

import json
import re
import time

import numpy as np

import sglang as sgl
from sglang.utils import download_and_cache_file, read_jsonl


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
        assert ret["answer"] == "True", ret.text()
    else:
        assert ret["answer"] in ["True", "False", "Unknown"]

    ret = true_or_false.run(
        statement="The capital of Canada is Tokyo.",
    )
    if check_answer:
        assert ret["answer"] == "False", ret.text()
    else:
        assert ret["answer"] in ["True", "False", "Unknown"]

    ret = true_or_false.run(
        statement="Purple is a better color than green.",
    )
    if check_answer:
        assert ret["answer"] == "Unknown", ret.text()
    else:
        assert ret["answer"] in ["True", "False", "Unknown"]


def test_decode_int():
    @sgl.function
    def decode_int(s):
        s += "The number of hours in a day is " + sgl.gen_int("hours") + "\n"
        s += "The number of days in a year is " + sgl.gen_int("days") + "\n"

    ret = decode_int.run(temperature=0.1)
    assert int(ret["hours"]) == 24, ret.text()
    assert int(ret["days"]) == 365, ret.text()


def test_decode_json_regex():
    @sgl.function
    def decode_json(s):
        from sglang.lang.ir import REGEX_FLOAT, REGEX_INT, REGEX_STR

        s += "Generate a JSON object to describe the basic city information of Paris.\n"
        s += "Here are the JSON object:\n"

        # NOTE: we recommend using dtype gen or whole regex string to control the output

        with s.var_scope("json_output"):
            s += "{\n"
            s += '  "name": ' + sgl.gen(regex=REGEX_STR) + ",\n"
            s += '  "population": ' + sgl.gen(regex=REGEX_INT, stop=[" ", "\n"]) + ",\n"
            s += '  "area": ' + sgl.gen(regex=REGEX_INT, stop=[" ", "\n"]) + ",\n"
            s += '  "latitude": ' + sgl.gen(regex=REGEX_FLOAT, stop=[" ", "\n"]) + "\n"
            s += "}"

    ret = decode_json.run(temperature=0.0)
    try:
        js_obj = json.loads(ret["json_output"])
    except json.decoder.JSONDecodeError:
        print("JSONDecodeError", ret["json_output"])
        raise
    assert isinstance(js_obj["name"], str)
    assert isinstance(js_obj["population"], int)


def test_decode_json():
    @sgl.function
    def decode_json(s):
        s += "Generate a JSON object to describe the basic city information of Paris.\n"

        with s.var_scope("json_output"):
            s += "{\n"
            s += '  "name": ' + sgl.gen_string() + ",\n"
            s += '  "population": ' + sgl.gen_int() + ",\n"
            s += '  "area": ' + sgl.gen(dtype=int) + ",\n"
            s += '  "country": ' + sgl.gen_string() + ",\n"
            s += '  "timezone": ' + sgl.gen(dtype=str) + "\n"
            s += "}"

    ret = decode_json.run(max_new_tokens=64)
    try:
        js_obj = json.loads(ret["json_output"])
    except json.decoder.JSONDecodeError:
        print("JSONDecodeError", ret["json_output"])
        raise
    assert isinstance(js_obj["name"], str)
    assert isinstance(js_obj["population"], int)


def test_expert_answer(check_answer=True):
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

    if check_answer:
        assert "paris" in ret.text().lower(), f"Answer: {ret.text()}"


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
    fork_size = 5

    @sgl.function
    def parallel_decoding(s, topic):
        s += "Act as a helpful assistant.\n"
        s += "USER: Give some tips for " + topic + ".\n"
        s += (
            "ASSISTANT: Okay. Here are "
            + str(fork_size)
            + " concise tips, each under 8 words:\n"
        )

        # Generate skeleton
        for i in range(1, 1 + fork_size):
            s += f"{i}." + sgl.gen(max_tokens=16, stop=[".", "\n"]) + ".\n"

        # Generate detailed tips
        forks = s.fork(fork_size)
        for i in range(fork_size):
            forks[
                i
            ] += f"Now, I expand tip {i+1} into a detailed paragraph:\nTip {i+1}:"
            forks[i] += sgl.gen("detailed_tip", max_tokens, stop=["\n\n"])
        forks.join()

        # Concatenate tips and summarize
        s += "Here are these tips with detailed explanation:\n"
        for i in range(fork_size):
            s += f"Tip {i+1}:" + forks[i]["detailed_tip"] + "\n"

        s += "\nIn summary," + sgl.gen("summary", max_tokens=512)

    ret = parallel_decoding.run(topic="writing a good blog post", temperature=0.3)
    assert isinstance(ret["summary"], str)


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
        s += sgl.user(sgl.image("example_image.png") + question)
        s += sgl.assistant(sgl.gen("answer"))

    state = image_qa.run(
        question="Please describe this image in simple words.",
        temperature=0,
        max_new_tokens=64,
    )

    assert (
        "taxi" in state.messages()[-1]["content"]
        or "car" in state.messages()[-1]["content"]
    ), f"{state.messages()[-1]['content']}"


def test_stream():
    @sgl.function
    def qa(s, question):
        s += sgl.system("You are a helpful assistant.")
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


def test_dtype_gen():
    @sgl.function
    def dtype_gen(s):
        s += "Q: What is the full name of DNS?\n"
        s += "A: The full nams is " + sgl.gen("str_res", dtype=str, stop="\n") + "\n"
        s += "Q: Which year was DNS invented?\n"
        s += "A: " + sgl.gen("int_res", dtype=int) + "\n"
        s += "Q: What is the value of pi?\n"
        s += "A: " + sgl.gen("float_res", dtype=float) + "\n"
        s += "Q: Is the sky blue?\n"
        s += "A: " + sgl.gen("bool_res", dtype=bool) + "\n"

    state = dtype_gen.run()

    try:
        state["int_res"] = int(state["int_res"])
        state["float_res"] = float(state["float_res"])
        state["bool_res"] = bool(state["bool_res"])
        # assert state["str_res"].startswith('"') and state["str_res"].endswith('"')
    except ValueError:
        print(state)
        raise


def test_completion_speculative():
    @sgl.function(num_api_spec_tokens=64)
    def gen_character_spec(s):
        s += "Construct a character within the following format:\n"
        s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n"
        s += "\nPlease generate new Name, Birthday and Job.\n"
        s += (
            "Name:"
            + sgl.gen("name", stop="\n")
            + "\nBirthday:"
            + sgl.gen("birthday", stop="\n")
        )
        s += "\nJob:" + sgl.gen("job", stop="\n") + "\n"

    @sgl.function
    def gen_character_no_spec(s):
        s += "Construct a character within the following format:\n"
        s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n"
        s += "\nPlease generate new Name, Birthday and Job.\n"
        s += (
            "Name:"
            + sgl.gen("name", stop="\n")
            + "\nBirthday:"
            + sgl.gen("birthday", stop="\n")
        )
        s += "\nJob:" + sgl.gen("job", stop="\n") + "\n"

    token_usage = sgl.global_config.default_backend.token_usage

    token_usage.reset()
    gen_character_spec().sync()
    usage_with_spec = token_usage.prompt_tokens

    token_usage.reset()
    gen_character_no_spec().sync()
    usage_with_no_spec = token_usage.prompt_tokens

    assert (
        usage_with_spec < usage_with_no_spec
    ), f"{usage_with_spec} vs {usage_with_no_spec}"


def test_chat_completion_speculative():
    @sgl.function(num_api_spec_tokens=256)
    def gen_character_spec(s):
        s += sgl.system("You are a helpful assistant.")
        s += sgl.user("Construct a character within the following format:")
        s += sgl.assistant(
            "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n"
        )
        s += sgl.user("Please generate new Name, Birthday and Job.\n")
        s += sgl.assistant(
            "Name:"
            + sgl.gen("name", stop="\n")
            + "\nBirthday:"
            + sgl.gen("birthday", stop="\n")
            + "\nJob:"
            + sgl.gen("job", stop="\n")
        )

    gen_character_spec().sync()


def test_hellaswag_select():
    """Benchmark the accuracy of sgl.select on the HellaSwag dataset."""

    def get_one_example(lines, i, include_answer):
        ret = lines[i]["activity_label"] + ": " + lines[i]["ctx"] + " "
        if include_answer:
            ret += lines[i]["endings"][lines[i]["label"]]
        return ret

    def get_few_shot_examples(lines, k):
        ret = ""
        for i in range(k):
            ret += get_one_example(lines, i, True) + "\n\n"
        return ret

    # Read data
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    filename = download_and_cache_file(url)
    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = 200
    num_shots = 20
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    choices = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        choices.append(lines[i]["endings"])
        labels.append(lines[i]["label"])
    arguments = [{"question": q, "choices": c} for q, c in zip(questions, choices)]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_hellaswag(s, question, choices):
        s += few_shot_examples + question
        s += sgl.select("answer", choices=choices)

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.time()
    rets = few_shot_hellaswag.run_batch(
        arguments,
        temperature=0,
        num_threads=64,
        progress_bar=True,
        generator_style=False,
    )
    preds = []
    for i, ret in enumerate(rets):
        preds.append(choices[i].index(ret["answer"]))
    latency = time.time() - tic

    # Compute accuracy
    accuracy = np.mean(np.array(preds) == np.array(labels))

    # Test generator style of run_batch
    tic = time.time()
    rets = few_shot_hellaswag.run_batch(
        arguments,
        temperature=0,
        num_threads=64,
        progress_bar=True,
        generator_style=True,
    )
    preds_gen = []
    for i, ret in enumerate(rets):
        preds_gen.append(choices[i].index(ret["answer"]))
    latency_gen = time.time() - tic

    # Compute accuracy
    accuracy_gen = np.mean(np.array(preds_gen) == np.array(labels))
    print(f"{accuracy=}, {accuracy_gen=}")
    assert np.abs(accuracy_gen - accuracy) < 0.05
    assert np.abs(latency_gen - latency) < 1

    return accuracy, latency


def test_gen_min_new_tokens():
    """
    Validate sgl.gen(min_tokens) functionality.

    The test asks a question where, without a min_tokens constraint, the generated answer is expected to be short.
    By enforcing the min_tokens parameter, we ensure the generated answer has at least the specified number of tokens.
    We verify that the number of tokens in the answer is >= the min_tokens threshold.
    """
    import sglang as sgl
    from sglang.srt.hf_transformers_utils import get_tokenizer

    model_path = sgl.global_config.default_backend.endpoint.get_model_name()
    MIN_TOKENS, MAX_TOKENS = 64, 128

    @sgl.function
    def convo_1(s):
        s += sgl.user("What is the capital of the United States?")
        s += sgl.assistant(
            sgl.gen("answer", min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS)
        )

    def assert_min_tokens(tokenizer, text):
        token_ids = tokenizer.encode(text)
        assert (
            len(token_ids) >= MIN_TOKENS
        ), f"Generated {len(token_ids)} tokens, min required: {MIN_TOKENS}. Text: {text}"

    tokenizer = get_tokenizer(model_path)

    state = convo_1.run()
    assert_min_tokens(tokenizer, state["answer"])
