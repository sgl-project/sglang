from sglang import OpenAI, function, gen, set_default_backend


@function()
def gen_character_default(s):
    s += "Construct a character within the following format:\n"
    s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\nWelcome.\n"
    s += "\nPlease generate new Name, Birthday and Job.\n"
    s += "Name:" + gen("name", stop="\n") + "\nBirthday:" + gen("birthday", stop="\n")
    s += "\nJob:" + gen("job", stop="\n") + "\nWelcome.\n"


@function(api_num_spec_tokens=512)
def gen_character_spec(s):
    s += "Construct a character within the following format:\n"
    s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\nWelcome.\n"
    s += "\nPlease generate new Name, Birthday and Job.\n"
    s += "Name:" + gen("name", stop="\n") + "\nBirthday:" + gen("birthday", stop="\n")
    s += "\nJob:" + gen("job", stop="\n") + "\nWelcome.\n"


@function(api_num_spec_tokens=512)
def gen_character_no_stop(s):
    s += "Construct a character within the following format:\n"
    s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\nWelcome.\n"
    s += "\nPlease generate new Name, Birthday and Job.\n"
    s += "Name:" + gen("name") + "\nBirthday:" + gen("birthday")
    s += "\nJob:" + gen("job") + "\nWelcome.\n"


@function(api_num_spec_tokens=512)
def gen_character_multi_stop(s):
    s += "Construct a character within the following format:\n"
    s += (
        "Name: Steve Jobs.###Birthday: February 24, 1955.###Job: Apple CEO.\nWelcome.\n"
    )
    s += "\nPlease generate new Name, Birthday and Job.\n"
    s += "Name:" + gen("name", stop=["\n", "###"])
    s += "###Birthday:" + gen("birthday", stop=["\n", "###"])
    s += "###Job:" + gen("job", stop=["\n", "###"]) + "\nWelcome.\n"


set_default_backend(OpenAI("gpt-3.5-turbo-instruct"))

state = gen_character_default.run()
print(state.text())

print("=" * 60)

state = gen_character_no_stop.run()

print("name###", state["name"])
print("birthday###:", state["birthday"])
print("job###", state["job"])

print("=" * 60)

state = gen_character_multi_stop.run()
print(state.text())

print("=" * 60)

state = gen_character_spec.run()
print(state.text())

print("name###", state["name"])
print("birthday###", state["birthday"])
print("job###", state["job"])
