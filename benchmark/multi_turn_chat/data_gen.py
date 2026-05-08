import random
import string

random.seed(42)


def gen_prompt(tokenizer, token_num):
    cha_set = string.ascii_letters + string.digits
    ret = "".join(random.choices(cha_set, k=token_num))
    while len(tokenizer(ret).input_ids) < token_num:
        ret += random.choice(cha_set)
    return ret


def gen_arguments(args, tokenizer):
    multi_qas = [{"qas": []} for _ in range(args.num_qa)]
    for i in range(args.num_qa):
        qas = multi_qas[i]["qas"]
        for _ in range(args.turns):
            prompt_len = random.randint(args.min_len_q, args.max_len_q)
            new_tokens = random.randint(args.min_len_a, args.max_len_a)
            qas.append(
                {
                    "prompt": gen_prompt(tokenizer, prompt_len),
                    "new_tokens": new_tokens,
                }
            )

    return multi_qas
