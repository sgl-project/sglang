import sglang as sgl

ref_text = """\
The location of Hogwarts is Scotland, UK.
The headmaster of Hogwarts is Albus Dumbledore.
The potions teacher in Hogwarts is Severus Snape.
The transfiguration teacher in Hogwarts is Minerva McGonagall.
The herbology teacher in Hogwarts is Pomona Sprout.
The defense against the dark arts teacher in Hogwarts is Gilderoy Lockhart."""


@sgl.function
def spec_ref_gen(s, question, answer, ref_text):
    s += "According to the reference text, answer the following question:\n"
    s += "Reference text Begins.\n"
    s += ref_text + "\n"
    s += "Reference text Ends.\n"
    s += question + "\n"
    s += answer + sgl.gen("answer", ref_text=ref_text, stop="\n")


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
    states = spec_ref_gen.run_batch(arguments, temperature=0)
    for state in states:
        print(state["answer"])
        print("=" * 50)


if __name__ == "__main__":
    main()
