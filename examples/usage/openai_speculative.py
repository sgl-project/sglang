from sglang import function, gen, set_default_backend, OpenAI


@function(api_num_spec_tokens=512)
def gen_character_spec(s):
    s += "Construct a character within the following format:\n"
    s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\n"
    s += "\nPlease generate new Name, Birthday and Job.\n"
    s += "Name:" + gen("name", stop="\n") + "\nBirthday:" + gen("birthday", stop="\n")
    s += "\nJob:" + gen("job", stop="\n") + "\n"


set_default_backend(OpenAI("gpt-3.5-turbo-instruct"))

state = gen_character_spec.run()

print("name:", state["name"])
print("birthday:", state["birthday"])
print("job:", state["job"])
