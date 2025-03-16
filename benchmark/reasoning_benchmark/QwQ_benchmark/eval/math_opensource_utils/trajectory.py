#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trajectory:
[
    {"role": "rationale", "content": "..."},
    {"role": "program", "content": "..."},
    {"role": "output", "content": "..."},
    {"role": "rationale", "content": "..."},
    ...
]
"""

import re
from typing import List


def text_to_trajectory(traj_str: str) -> List[dict]:
    """parse the above interleaved string of raionale, program, output, raionale, program, output, ...
    output a list of dict.
    """
    trajectory = []
    cur_role = "rationale"
    cur_content = ""

    for i, line in enumerate(traj_str.split("\n")):
        if line == "```python":  # program begin
            assert cur_role == "rationale"
            if cur_content:
                trajectory.append({"role": cur_role, "content": cur_content})
                cur_content = ""
            cur_role = "program"
        elif cur_role == "program" and line == "```":  # program end
            assert cur_content
            trajectory.append({"role": cur_role, "content": cur_content})
            cur_content = ""
            cur_role = "output"
        elif cur_role == "output" and line.startswith("```output"):  # output begin
            assert cur_content == ""
        elif cur_role == "output" and line == "```":  # output end
            trajectory.append({"role": cur_role, "content": cur_content})
            cur_content = ""
            cur_role = "rationale"
        else:  # content
            cur_content += line
            if i < len(traj_str.split("\n")) - 1:
                cur_content += "\n"
    # the last content
    if cur_content:
        trajectory.append({"role": cur_role, "content": cur_content})
    return trajectory


def trajectory_to_text(trajectory: list) -> str:
    text = ""
    for item in trajectory:
        content = item["content"]
        if item["role"] == "program":
            content = f"```python\n{content}```\n"
        elif item["role"] == "output":
            content = f"```output\n{content}```\n"
        text += content
    return text


def is_execution_success(output):
    error_key_words = [
        "error",
        "exception",
        "no algorithms",
        "no algorithms",
        "cannot",
        "nan",
        "...",
    ]
    success = all([k not in output.lower() for k in error_key_words])
    return success


def extract_program(text: str = None, trajectory: list = None, last_only=False) -> str:
    assert (
        text is not None or trajectory is not None
    ), "Either text or trajectory should be provided."
    if trajectory is None:
        try:
            trajectory = text_to_trajectory(text)
        except:
            return "raise ValueError('Invalid trajectory')"

    program_list = []
    import_lines = []
    for i, item in enumerate(trajectory):
        if item["role"] == "program":
            cur_program = item["content"]
            if i < len(trajectory) - 1:
                assert trajectory[i + 1]["role"] == "output"
                output = trajectory[i + 1]["content"].strip()
                if is_execution_success(output):
                    program_list.append(cur_program)
                else:
                    # extract import lines only
                    for line in cur_program.split("\n"):
                        if line.startswith("import") or line.startswith("from"):
                            import_lines.append(line)
            else:
                program_list.append(cur_program)
    # add import lines to the first program
    if len(program_list) == 0:
        program_list.append("")
    if len(import_lines) > 0:
        program_list[0] = "\n".join(import_lines) + "\n" + program_list[0]
    for i, program in enumerate(program_list[:-1]):
        program_list[i] = "\n".join(
            [
                line
                for line in program.split("\n")
                if not line.strip().startswith("print(")
            ]
        )

    if last_only:
        program = program_list[-1]
    else:
        program = "\n".join(program_list)
    return program


def extract_program_output(pred_str, last_only=True):
    r"""
    extract output between ```output\n...\n```, use regex, there might be multiple outputs, each output may have multiple lines
    """
    outputs = re.findall(r"```output\n(.*?)\n```", pred_str, re.DOTALL)
    if last_only:
        return outputs[-1] if len(outputs) > 0 else ""
    else:
        return outputs


def _test_str_to_trajectory():
    traj_text = r"""We can use the Chinese Remainder Theorem to solve this system of congruences. The Chinese Remainder Theorem states that if $n_1, n_2, \ldots, n_k$ are pairwise coprime (i.e., the greatest common divisor of any two of them is 1), and if $a_1, a_2, \ldots, a_k$ are any integers, then there is a number $x$ that satisfies the system of congruences:

$x \equiv a_1 \pmod{n_1}$
$x \equiv a_2 \pmod{n_2}$
$\vdots$
$x \equiv a_k \pmod{n_k}$

Moreover, any two solutions of this system are congruent modulo the product $n_1n_2 \cdots n_k$.

Let's write a function to solve this:

```python
from sympy import symbols, solve

def smallest_x():
    x = symbols('x')
    # Solve the system of congruences
    solution = solve([x - 7, x - 17], x, domain='Z')
    # Return the smallest positive solution
    return min([sol for sol in solution if sol >= 0])

result = smallest_x()
print(result)
```
```output
ValueError: min() arg is an empty sequence
```
The error in the last example is due to the incorrect usage of the `solve` function. The `solve` function is used to solve equations, not congruences. To solve a system of congruences, we can use the `solve_congruence` function from the `sympy.ntheory.modular` module.

Here is the corrected code:

```python
from sympy.ntheory.modular import solve_congruence

def smallest_x():
    # Solve the system of congruences
    solution = solve_congruence(('x', 7, 12), ('x', 17, 11))
    # Return the smallest positive solution
    return solution[0][0]

result = smallest_x()
print(result)
```
```output
good
```
ValueError: too many values to unpack (expected 2)
The error in the last example is due to the incorrect usage of the `solve_congruence` function. The `solve_congruence` function returns a tuple of two elements: the solution and the modulus. We need to unpack these two values correctly.

Here is the corrected code:

```python
from sympy.ntheory.modular import solve_congruence

def smallest_x():
    # Solve the system of congruences
    solution, modulus = solve_congruence(('x', 7, 12), ('x', 17, 11))
    # Return the smallest positive solution
    return solution

result = smallest_x()
print(result)
```"""

    import pprint

    trajectory = text_to_trajectory(traj_text)
    pprint.pprint(trajectory)

    text = trajectory_to_text(trajectory)
    assert text == traj_text

    # print(extract_program(traj_text))


if __name__ == "__main__":
    _test_str_to_trajectory()
