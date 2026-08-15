"""
Usage:
python3 openai_speculative.py
"""

from sglang import OpenAI, function, gen, set_default_backend


@function(num_api_spec_tokens=64)
def gen_character_spec(s):
    s += "Construct a character within the following format:\n"
    s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n"
    s += "\nPlease generate new Name, Birthday and Job.\n"
    s += "Name:" + gen("name", stop="\n") + "\nBirthday:" + gen("birthday", stop="\n")
    s += "\nJob:" + gen("job", stop="\n") + "\n"


@function
def gen_character_no_spec(s):
    s += "Construct a character within the following format:\n"
    s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n"
    s += "\nPlease generate new Name, Birthday and Job.\n"
    s += "Name:" + gen("name", stop="\n") + "\nBirthday:" + gen("birthday", stop="\n")
    s += "\nJob:" + gen("job", stop="\n") + "\n"


@function(num_api_spec_tokens=64)
def gen_character_spec_no_few_shot(s):
    # s += "Construct a character with name, birthday, and job:\n"
    s += "Construct a character:\n"
    s += "Name:" + gen("name", stop="\n") + "\nBirthday:" + gen("birthday", stop="\n")
    s += "\nJob:" + gen("job", stop="\n") + "\n"


if __name__ == "__main__":
    backend = OpenAI("gpt-3.5-turbo-instruct")
    set_default_backend(backend)

    for function in [
        gen_character_spec,
        gen_character_no_spec,
        gen_character_spec_no_few_shot,
    ]:
        backend.token_usage.reset()

        print(f"function: {function.func.__name__}")

        state = function.run()

        print("...name:", state["name"])
        print("...birthday:", state["birthday"])
        print("...job:", state["job"])
        print(backend.token_usage)
        print()
