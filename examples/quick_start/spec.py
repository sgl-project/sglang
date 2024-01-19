from sglang import function, gen, set_default_backend, OpenAI, user, assistant


@function
def example(s):
    with s.user():
        s += "Construct a character. Here are two examples:\n"
        s += "Name: Steve Jobs. Birthday: February 24, 1955. Job: Apple CEO.\n"
        s += "Name: Bill Gates. Birthday: January 4, 1954. Job: Microsoft CEO.\n"
        s += "Please generate a new character with Name, Birthday and Job in the same format.\n"
    s += assistant("Name: " + gen("name") + " Birthday: " + gen("birthday") + " Job: " + gen("job"))

set_default_backend(OpenAI("gpt-3.5-turbo-instruct"))

state = example.run()
print(state.text())
