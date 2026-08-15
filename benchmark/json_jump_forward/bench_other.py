import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import guidance
from tqdm import tqdm

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_generate
from sglang.utils import dump_state_text, read_jsonl

# there are some FSM bugs with json regex converted from pydantic model
# here use a string regex instead
# regex_string = build_regex_from_object(HarryPoterRole)
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)

city_regex = (
    r"""\{\n"""
    + r"""  "name": "[\w\d\s]{1,16}",\n"""
    + r"""  "country": "[\w\d\s]{1,16}",\n"""
    + r"""  "latitude": [-+]?[0-9]*\.?[0-9]{0,2},\n"""
    + r"""  "population": [-+]?[0-9]{1,9},\n"""
    + r"""  "top 3 landmarks": \["[\w\d\s]{1,16}", "[\w\d\s]{1,16}", "[\w\d\s]{1,16}"\]\n"""
    + r"""\}"""
)

# fmt: off
def character_gen(name, generate):
    s = name + " is a character in Harry Potter. Please fill in the following information about this character.\n"
    s += generate(s, max_tokens=256, regex=character_regex)
    return s
# fmt: on

# fmt: off
def city_gen(document, generate):
    s = "Please extract the information of a city from the following wikipedia page.\n"
    s += "Page begin.\n" + document + "Page end.\n"
    s += "Here is the name, country, and symbol of the city in JSON format.\n"
    s += generate(s, max_tokens=256, regex=city_regex)
    return s
# fmt: on


@guidance
def character_maker(lm, name):
    regex_str_no_quote = r"[\w\d\s]+"
    regex_float = r"[0-9]+\.[0-9]+"
    lm += f"""\
    {name} is a character in Harry Potter. Please fill in the following information about this character.
    {{
        "name": "{guidance.gen("name", max_tokens=16, regex=regex_str_no_quote)}",
        "house": "{guidance.select(options=['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'], name='house')}",
        "blood status": "{guidance.select(options=['Pure-blood', 'Half-blood', 'Muggle-born'], name='blood status')}",
        "occupation": "{guidance.select(options=['student', 'teacher', 'auror', 'ministry of magic', 'death eater', 'order of the phoenix'], name='occupation')}",
        "wand": {{
            "wood": "{guidance.gen("wood", max_tokens=16, regex=regex_str_no_quote)}",
            "core": "{guidance.gen('core', max_tokens=16, regex=regex_str_no_quote)}",
            "length": {guidance.gen('length', max_tokens=10, regex=regex_float)}
        }},
        "alive": "{guidance.select(options=['Alive', 'Deceased'], name='alive')}",
        "patronus": "{guidance.gen('patronus', max_tokens=16, regex=regex_str_no_quote)}",
        "bogart": "{guidance.gen('bogart', max_tokens=16, regex=regex_str_no_quote)}"
    }}
    """

    return lm


async def call_generate_lmql(
    prompt, temperature, max_tokens, regex, max_len=4096, model=None, **kwargs
):
    assert model is not None
    import lmql

    @lmql.query(model=model)
    async def program(question, max_tokens, regex):
        '''lmql
        """{question}[ANSWER]""" where len(TOKENS(ANSWER)) < max_tokens and REGEX(ANSWER, regex)
        return ANSWER
        '''

    return await program(
        question=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        max_len=max_len,
        regex=regex,
        **kwargs,
    )


@guidance
def city_maker(lm, document):
    regex_str_no_quote = r"[\w\d\s]+"
    regex_float = r"[0-9]+\.[0-9]+"
    lm += f"""\
    Please extract the information of a city from the following wikipedia page.
    Page begin.
    {document}
    Page end.
    Here is the name, country, and symbol of the city in JSON format.
    {{
        "name": "{guidance.gen("name", max_tokens=16, regex=regex_str_no_quote)}",
        "country": "{guidance.gen("country", max_tokens=16, regex=regex_str_no_quote)}",
        "latitude": {guidance.gen("latitude", max_tokens=10, regex=regex_float)},
        "population": {guidance.gen("population", max_tokens=10, regex=r"[0-9]+")},
        "top 3 landmarks": [
            "{guidance.gen("landmark1", max_tokens=16, regex=regex_str_no_quote)}", "{guidance.gen("landmark2", max_tokens=16, regex=regex_str_no_quote)}", "{guidance.gen("landmark3", max_tokens=16, regex=regex_str_no_quote)}"
        ]
    }}
    """

    return lm


def bench_character(args):
    arguments = []
    with open(args.data_path, "r") as f:
        for line in f:
            arguments.append({"name": line.strip()})
    arguments = arguments[: args.num_jsons]

    states = [None] * len(arguments)

    # Select backend
    if args.backend == "outlines":
        call_generate = partial(get_call_generate(args), temperature=0)

        def get_one_answer(i):
            states[i] = character_gen(**arguments[i], generate=call_generate)

    elif args.backend == "guidance":
        model = guidance.models.LlamaCpp(
            args.model_path,
            n_gpu_layers=-1,
            n_ctx=args.n_ctx,
        )

        def get_one_answer(i):
            lm = model + character_maker(**arguments[i])
            states[i] = lm

    elif args.backend == "lmql":
        import asyncio

        import lmql

        model = lmql.model(args.model_path, endpoint=f"{args.host}:{args.port}")
        call_generate = partial(
            call_generate_lmql,
            model=model,
            max_tokens=256,
            regex=character_regex,
        )

        async def get_one_answer_async(i):
            states[i] = await call_generate(prompt=arguments[i]["name"], temperature=0)

    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    tic = time.perf_counter()

    if args.backend != "lmql":
        if args.parallel == 1:
            for i in tqdm(range(len(arguments))):
                get_one_answer(i)
        else:
            with ThreadPoolExecutor(args.parallel) as executor:
                rets = list(
                    tqdm(
                        executor.map(get_one_answer, list(range(len(arguments)))),
                        total=len(arguments),
                    )
                )
                for _ in rets:
                    pass
    else:
        batches = []
        for i in range(0, len(arguments), args.parallel):
            batches.append(list(range(i, min(i + args.parallel, len(arguments)))))
        loop = asyncio.get_event_loop()

        for bt in tqdm(batches):
            loop.run_until_complete(
                asyncio.gather(*[get_one_answer_async(i) for i in bt])
            )

    latency = time.perf_counter() - tic

    return states, latency


def bench_city_doc(args):
    arguments = []
    for line in read_jsonl(args.data_path):
        arguments.append({"document": line["document"]})
    arguments = arguments[: args.num_jsons]

    states = [None] * len(arguments)

    # Select backend
    if args.backend == "outlines":
        call_generate = partial(get_call_generate(args), temperature=0)

        def get_one_answer(i):
            states[i] = city_gen(**arguments[i], generate=call_generate)

    elif args.backend == "guidance":
        model = guidance.models.LlamaCpp(
            args.model_path,
            n_gpu_layers=-1,
            n_ctx=args.n_ctx,
        )

        def get_one_answer(i):
            lm = model + city_maker(**arguments[i])
            states[i] = lm

    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    tic = time.perf_counter()
    if args.parallel == 1:
        for i in tqdm(range(len(arguments))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            rets = executor.map(get_one_answer, list(range(len(arguments))))
            for _ in rets:
                pass

    latency = time.perf_counter() - tic

    return states, latency


def main(args):
    if args.mode == "character":
        args.data_path = "dataset.txt"
        states, latency = bench_character(args)
    elif args.mode == "city":
        args.data_path = "questions.jsonl"
        states, latency = bench_city_doc(args)

    # Compute accuracy
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}_{args.mode}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "json_jump_forward",
            "backend": args.backend,
            "latency": round(latency, 3),
            "num_jsons": args.num_jsons,
            "mode": args.mode,
            "parallel": args.parallel,
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--num-jsons", type=int, default=50)
    parser.add_argument(
        "--mode", type=str, default="character", choices=["character", "city"]
    )
    args = add_common_other_args_and_parse(parser)
    main(args)
