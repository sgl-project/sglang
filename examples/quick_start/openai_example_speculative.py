from sglang import function, gen, set_default_backend, OpenAI

@function(api_num_spec_tokens=512)
def example(s):
  s += "Construct a character within the following format:\n"
  s += "Name: Steve Jobs.\nBirthday: February 24, 1955.\nJob: Apple CEO.\nWelcome.\n"
  s += "\nPlease generate new Name, Birthday and Job.\n"
  s += "Name: " + gen("name", stop="\n") + "Birth" + "day: " + gen("birthday", stop="\n")
  s += "Job: " + gen("job", stop="\n") + "Welcome.\n"
  # Sometimes the first generation for "name" will start with the stop symbol "\n" which will fully break the whole generation (always start with "\n"), results in a single "\n" for each gen.
  # This is not a challenge that expect to be handled in SGLang, since SGLang seeks for exact execution for now. It should be handled by a better prompt.
  # We keep the example to showcase this subtle issue.

set_default_backend(OpenAI("gpt-3.5-turbo-instruct"))

state = example.run()
print(state.text())

print("name:", state["name"])
print("birthday:", state["birthday"])
print("job:", state["job"])
