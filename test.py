import sglang as sgl

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
def regex_chinese(s):
    s += "如果你想学习魔法，你需要去:"
    s += sgl.gen("res", regex=r"霍格沃茨魔法学校|霍比特人的洞穴") + "\n"
    s += "分院帽说，甘道夫是一个:"
    s += sgl.gen("res", regex=r"(格兰芬多|斯莱特林|拉文克劳|赫奇帕奇)!") + "\n"


@sgl.function
def character_gen(s, name):
    s += (
        name
        + " is a character in Harry Potter. Please fill in the following information about this character.\n"
    )
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)


def main():
    backend = sgl.RuntimeEndpoint("http://localhost:30000")
    sgl.set_default_backend(backend)
    ret = regex_chinese.run(temperature=0)
    print(ret.text())
    ret = character_gen.run("Harry Potter", temperature=0)
    print(ret.text())


if __name__ == "__main__":
    main()
