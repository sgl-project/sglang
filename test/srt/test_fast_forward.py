import argparse
from enum import Enum

import sglang as sgl
from pydantic import BaseModel, constr
from sglang.srt.constrained.json_schema import build_regex_from_object
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)

IP_REGEX = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"

ip_fast_forward = (
    r"The google's DNS sever address is "
    + IP_REGEX
    + r" and "
    + IP_REGEX
    + r". "
    + r"The google's website domain name is "
    + r"www\.(\w)+\.(\w)+"
    + r"."
)


# fmt: off
@sgl.function
def regex_gen(s):
    s += "Q: What is the IP address of the Google DNS servers?\n"
    s += "A: " + sgl.gen(
        "answer",
        max_tokens=128,
        temperature=0,
        regex=ip_fast_forward,
    )
# fmt: on

json_fast_forward = (
    r"""The information about Hogwarts is in the following JSON format\.\n"""
    + r"""\n\{\n"""
    + r"""  "name": "[\w\d\s]*",\n"""
    + r"""  "country": "[\w\d\s]*",\n"""
    + r"""  "latitude": [-+]?[0-9]*\.?[0-9]+,\n"""
    + r"""  "population": [-+]?[0-9]+,\n"""
    + r"""  "top 3 landmarks": \["[\w\d\s]*", "[\w\d\s]*", "[\w\d\s]*"\],\n"""
    + r"""\}\n"""
)

# fmt: off
@sgl.function
def json_gen(s):
    s += sgl.gen(
        "json",
        max_tokens=128,
        temperature=0,
        regex=json_fast_forward,
    )
# fmt: on


class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    weapon: Weapon
    strength: int


@sgl.function
def character_gen(s):
    s += "Give me a character description who is a wizard.\n"
    s += sgl.gen(
        "character",
        max_tokens=128,
        temperature=0,
        regex=build_regex_from_object(Character),
    )


def main(args):
    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    state = regex_gen.run(temperature=0)

    print("=" * 20, "IP TEST", "=" * 20)
    print(state.text())

    state = json_gen.run(temperature=0)

    print("=" * 20, "JSON TEST", "=" * 20)
    print(state.text())

    state = character_gen.run(temperature=0)

    print("=" * 20, "CHARACTER TEST", "=" * 20)
    print(state.text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_common_sglang_args_and_parse(parser)
    main(args)

# ==================== IP TEST ====================
# Q: What is the IP address of the Google DNS servers?
# A: The google's DNS sever address is 8.8.8.8 and 8.8.4.4. The google's website domain name is www.google.com.
# ==================== JSON TEST ====================
# The information about Hogwarts is in the following JSON format.

# {
#   "name": "Hogwarts School of Witchcraft and Wizardry",
#   "country": "Scotland",
#   "latitude": 55.566667,
#   "population": 1000,
#   "top 3 landmarks": ["Hogwarts Castle", "The Great Hall", "The Forbidden Forest"],
# }

# ==================== CHARACTER TEST ====================
# Give me a character description who is a wizard.
# { "name" : "Merlin", "age" : 500, "armor" : "chainmail" , "weapon" : "sword" , "strength" : 10 }
