"""
Adapted from
https://github.com/stanfordnlp/dspy/blob/34d8420383ec752037aa271825c1d3bf391e1277/intro.ipynb#L9
"""

import argparse

import dspy
from dspy.datasets import HotPotQA


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def main(args):
    # lm = dspy.OpenAI(model='gpt-3.5-turbo')
    if args.backend == "tgi":
        lm = dspy.HFClientTGI(
            model="meta-llama/Llama-2-7b-chat-hf",
            port=args.port,
            url="http://localhost",
        )
    elif args.backend == "sglang":
        lm = dspy.HFClientSGLang(
            model="meta-llama/Llama-2-7b-chat-hf",
            port=args.port,
            url="http://localhost",
        )
    elif args.backend == "vllm":
        lm = dspy.HFClientVLLM(
            model="meta-llama/Llama-2-7b-chat-hf",
            port=args.port,
            url="http://localhost",
        )
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    colbertv2_wiki17_abstracts = dspy.ColBERTv2(
        url="http://20.102.90.50:2017/wiki17_abstracts"
    )
    dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

    # Load the dataset.
    dataset = HotPotQA(
        train_seed=1, train_size=20, eval_seed=2023, dev_size=args.dev_size, test_size=0
    )

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]

    print(len(trainset), len(devset))

    train_example = trainset[0]
    print(f"Question: {train_example.question}")
    print(f"Answer: {train_example.answer}")

    dev_example = devset[18]
    print(f"Question: {dev_example.question}")
    print(f"Answer: {dev_example.answer}")
    print(f"Relevant Wikipedia Titles: {dev_example.gold_titles}")

    print(
        f"For this dataset, training examples have input keys {train_example.inputs().keys()} and label keys {train_example.labels().keys()}"
    )
    print(
        f"For this dataset, dev examples have input keys {dev_example.inputs().keys()} and label keys {dev_example.labels().keys()}"
    )

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    pred = generate_answer(question=dev_example.question)

    # Print the input and the prediction.
    print(f"Question: {dev_example.question}")
    print(f"Predicted Answer: {pred.answer}")

    lm.inspect_history(n=1)

    # Define the predictor. Notice we're just changing the class. The signature BasicQA is unchanged.
    generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)

    # Call the predictor on the same input.
    pred = generate_answer_with_chain_of_thought(question=dev_example.question)

    # Print the input, the chain of thought, and the prediction.
    print(f"Question: {dev_example.question}")
    print(f"Thought: {pred.rationale.split('.', 1)[1].strip()}")
    print(f"Predicted Answer: {pred.answer}")

    retrieve = dspy.Retrieve(k=3)
    topK_passages = retrieve(dev_example.question).passages

    print(
        f"Top {retrieve.k} passages for question: {dev_example.question} \n",
        "-" * 30,
        "\n",
    )

    for idx, passage in enumerate(topK_passages):
        print(f"{idx+1}]", passage, "\n")

    retrieve("When was the first FIFA World Cup held?").passages[0]

    from dspy.teleprompt import BootstrapFewShot

    # Validation logic: check that the predicted answer is correct.
    # Also check that the retrieved context does actually contain that answer.
    def validate_context_and_answer(example, pred, trace=None):
        answer_EM = dspy.evaluate.answer_exact_match(example, pred)
        answer_PM = dspy.evaluate.answer_passage_match(example, pred)
        return answer_EM and answer_PM

    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

    # Compile!
    compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

    # Ask any question you like to this simple RAG program.
    my_question = "What castle did David Gregory inherit?"

    # Get the prediction. This contains `pred.context` and `pred.answer`.
    pred = compiled_rag(my_question)

    # Print the contexts and the answer.
    print(f"Question: {my_question}")
    print(f"Predicted Answer: {pred.answer}")
    print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

    from dspy.evaluate.evaluate import Evaluate

    # Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
    evaluate_on_hotpotqa = Evaluate(
        devset=devset,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=5,
    )

    # Evaluate the `compiled_rag` program with the `answer_exact_match` metric.
    metric = dspy.evaluate.answer_exact_match
    evaluate_on_hotpotqa(compiled_rag, metric=metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--dev-size", type=int, default=150)
    parser.add_argument(
        "--backend", type=str, choices=["sglang", "tgi", "vllm"], default="sglang"
    )
    args = parser.parse_args()

    if args.port is None:
        default_port = {
            "vllm": 21000,
            "lightllm": 22000,
            "tgi": 24000,
            "sglang": 30000,
        }
        args.port = default_port.get(args.backend, None)

    main(args)
