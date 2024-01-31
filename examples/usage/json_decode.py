"""
Usage:
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
python json_decode.py
"""
from enum import Enum

from pydantic import BaseModel, constr
import sglang as sgl
from sglang.srt.constrained.json_schema import build_regex_from_object


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


@sgl.function
def character_gen(s, name):
    s += name + " is a character in Harry Potter. Please fill in the following information about this character.\n"
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)


def driver_character_gen():
    state = character_gen.run(name="Hermione Granger")
    print(state.text())


class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Wizard(BaseModel):
    name: str
    age: int
    weapon: Weapon


@sgl.function
def pydantic_wizard_gen(s):
    s += "Give me a description about a wizard in the JSON format.\n"
    s += sgl.gen(
        "character",
        max_tokens=128,
        temperature=0,
        regex=build_regex_from_object(Wizard),  # Requires pydantic >= 2.0
    )


def driver_character_gen():
    state = character_gen.run(name="Hermione Granger")
    print(state.text())


def driver_pydantic_wizard_gen():
    state = pydantic_wizard_gen.run()
    print(state.text())


if __name__ == "__main__":
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    driver_character_gen()
    # driver_pydantic_wizard_gen()
