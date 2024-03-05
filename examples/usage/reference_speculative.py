import sglang as sgl

ref_text = """\
The location of Hogwarts is Scotland, UK.
The headmaster of Hogwarts is Albus Dumbledore.
The potions teacher in Hogwarts is Severus Snape.
The transfiguration teacher in Hogwarts is Minerva McGonagall.
The herbology teacher in Hogwarts is Pomona Sprout.
The defense against the dark arts teacher in Hogwarts is Gilderoy Lockhart."""

ref_code = """\
```
import numpy as np
import matplotlib.pyplot as plt

# Calculate the average
average_throughput = np.mean(tokens_per_sec_arr)
print(f"Average Throughput: {average_throughput} tokens/sec")

# Plotting the histogram
plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Throughput Values')
plt.xlabel('Tokens per Second')
plt.ylabel('Frequency')
plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
plt.show()
```"""


@sgl.function
def simple_qa(s, question, answer, ref_text):
    s += "According to the reference text, answer the following question:\n"
    s += "Reference text Begins.\n"
    s += ref_text + "\n"
    s += "Reference text Ends.\n"
    s += question + "\n"
    s += answer + sgl.gen("answer", stop="\n")
    # s += answer + sgl.gen("answer", stop="\n", ref_text=ref_text)


@sgl.function
def code_modification(s, ref_code, question):
    s += "Question: Below is a python code:\n"
    s += ref_code + "\n"
    s += question + "\n"
    s += "Answer: Sure, here is the modified code:\n"
    s += "```"
    s += sgl.gen("code", ref_text=ref_code, max_tokens=1024, temperature=0, stop="```")


def main():
    backend = sgl.RuntimeEndpoint("http://localhost:30000")
    sgl.set_default_backend(backend)
    arguments = [
        {
            "ref_text": ref_text,
            "question": "Where is Hogwarts located?\n",
            "answer": "The location of Hogwarts",
        },
        {
            "ref_text": ref_text,
            "question": "Who is the headmaster of Hogwarts?\n",
            "answer": "The headmaster of Hogwarts",
        },
        {
            "ref_text": ref_text,
            "question": "Who is the potions teacher in Hogwarts?\n",
            "answer": "The potions teacher in Hogwarts",
        },
        {
            "ref_text": ref_text,
            "question": "Who is the transfiguration teacher in Hogwarts?\n",
            "answer": "The transfiguration teacher in Hogwarts",
        },
        {
            "ref_text": ref_text,
            "question": "Who is the herbology teacher in Hogwarts?\n",
            "answer": "The herbology teacher in Hogwarts",
        },
        {
            "ref_text": ref_text,
            "question": "Who is the defense against the dark arts teacher in Hogwarts?\n",
            "answer": "The defense against the dark arts teacher in Hogwarts",
        },
    ]
    states = simple_qa.run_batch(arguments, temperature=0)
    for state in states:
        print(state["answer"])
        print("=" * 50)

    state = code_modification.run(
        ref_code=ref_code,
        question="Can you please change x axis to start from 0?",
        stream=True,
    )
    for chunk in state.text_iter():
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()
