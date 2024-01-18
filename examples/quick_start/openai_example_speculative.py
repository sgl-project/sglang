from sglang import function, gen, set_default_backend, OpenAI
@function
def example(s):
  s += "Construct a character. Here is an example:\n"
  s += "Name: Steve Jobs. Birthday: February 24, 1955. Job: Apple CEO.\n"
  s += "Please generate new Name, Birthday and Job.\n"
  s += "Name: " + gen("name") + " Birthday: " + gen("birthday") + " Job: " + gen("job")

set_default_backend(OpenAI("gpt-3.5-turbo-instruct"))

state = example.run()
print(state.text())