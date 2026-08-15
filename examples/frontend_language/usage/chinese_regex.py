import sglang as sgl

character_regex = (
    r"""\{\n"""
    + r"""    "姓名": "[^"]{1,32}",\n"""
    + r"""    "学院": "(格兰芬多|赫奇帕奇|拉文克劳|斯莱特林)",\n"""
    + r"""    "血型": "(纯血|混血|麻瓜)",\n"""
    + r"""    "职业": "(学生|教师|傲罗|魔法部|食死徒|凤凰社成员)",\n"""
    + r"""    "魔杖": \{\n"""
    + r"""        "材质": "[^"]{1,32}",\n"""
    + r"""        "杖芯": "[^"]{1,32}",\n"""
    + r"""        "长度": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "存活": "(存活|死亡)",\n"""
    + r"""    "守护神": "[^"]{1,32}",\n"""
    + r"""    "博格特": "[^"]{1,32}"\n"""
    + r"""\}"""
)


@sgl.function
def character_gen(s, name):
    s += name + " 是一名哈利波特系列小说中的角色。请填写以下关于这个角色的信息。"
    s += """\
这是一个例子
{
    "姓名": "哈利波特",
    "学院": "格兰芬多",
    "血型": "混血",
    "职业": "学生",
    "魔杖": {
        "材质": "冬青木",
        "杖芯": "凤凰尾羽",
        "长度": 11.0
    },
    "存活": "存活",
    "守护神": "麋鹿",
    "博格特": "摄魂怪"
}
"""
    s += f"现在请你填写{name}的信息：\n"
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)


def main():
    backend = sgl.RuntimeEndpoint("http://localhost:30000")
    sgl.set_default_backend(backend)
    ret = character_gen.run(name="赫敏格兰杰", temperature=0)
    print(ret.text())


if __name__ == "__main__":
    main()
