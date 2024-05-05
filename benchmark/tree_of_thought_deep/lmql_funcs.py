from bench_other import (
    ASSISTANT_PREFIX,
    ASSISTANT_SUFFIX,
    USER_PREFIX,
    USER_SUFFIX,
    temp,
)


async def propose_plan_async(s, question, num_branches, call_generate):
    s += (
        USER_PREFIX
        + """Please generate a high-level plan for solving the following question. As the first step, just say what method and idea you will use to solve the question. You can reorganize the information in the question. Do not do the actual calculation. Keep your response concise and within 80 words. Question: """
        + question
        + USER_SUFFIX
    )

    s += ASSISTANT_PREFIX
    comps = await call_generate(
        s, max_tokens=256, temperature=temp, stop=None, n=num_branches
    )
    return [s + comp + ASSISTANT_SUFFIX for comp in comps]


async def execute_plan_async(s, num_branches, call_generate):
    s += (
        USER_PREFIX
        + """The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the high-level plan. Give me the final answer. Make your response short."""
        + USER_SUFFIX
    )
    s += ASSISTANT_PREFIX
    comps = await call_generate(
        s, max_tokens=256, temperature=temp, stop=None, n=num_branches
    )
    return [s + comp + ASSISTANT_SUFFIX for comp in comps]


async def reflect_solution_async(s, num_branches, call_generate):
    s += (
        USER_PREFIX
        + """Okay. Now, evaluate your own solution and give it a score on a scale of 1 to 5. Please do rigorous check of the correctness."""
        + USER_SUFFIX
    )
    s += ASSISTANT_PREFIX
    comps = await call_generate(
        s, max_tokens=256, temperature=temp, stop=None, n=num_branches
    )
    return [s + comp + ASSISTANT_SUFFIX for comp in comps]


async def get_final_answer_async(s, num_branches, call_generate):
    s += (
        USER_PREFIX
        + """Based on your reflection, do you change your mind? Now, give me the final answer after careful consideration."""
        + USER_SUFFIX
    )
    s += ASSISTANT_PREFIX
    comps = await call_generate(
        s, max_tokens=256, temperature=temp, stop=None, n=num_branches
    )
    return [s + comp + ASSISTANT_SUFFIX for comp in comps]


async def tree_search_async(question, num_branches, call_generate):
    plan_forks = await propose_plan_async("", question, num_branches, call_generate)

    sol_states = []
    for plan in plan_forks:
        forks = await execute_plan_async(plan, num_branches, call_generate)
        sol_states.extend(forks)

    ref_states = []
    for sol in sol_states:
        forks = await reflect_solution_async(sol, num_branches, call_generate)
        ref_states.extend(forks)

    solutions = []
    for sol in ref_states:
        ans = await get_final_answer_async(sol, num_branches, call_generate)
        solutions.append(ans)

    return solutions
