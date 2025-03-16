#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from collections import Counter

import sympy as sp
import timeout_decorator
from sympy.parsing.latex import parse_latex


def compare_numerical_ans(ans_p, ans_l):
    if ans_p is None:
        return False
    ans_p = ans_p.replace(",", "").replace("$", "")
    ans_l = ans_l.replace(",", "").replace("$", "")
    try:
        if ans_p.endswith("%"):
            ans_p = float(ans_p.rstrip("%")) / 100
        if isinstance(ans_p, str):
            ans_p = float(ans_p)
        if isinstance(ans_l, str):
            ans_l = float(ans_l)
    except Exception as e:
        return False
    return abs(ans_p - float(ans_l)) < 1e-3


def my_parse_latex(expr_str):
    expr_str = expr_str.replace("dfrac", "frac")
    expr = parse_latex(expr_str)
    if "\\pi" in expr_str:
        expr = expr.subs({sp.Symbol("pi"): sp.pi})
    expr = expr.subs({sp.Symbol("i"): sp.I})
    return expr


def is_number(element: str) -> bool:
    try:
        float(element.replace(" ", ""))
        return True
    except ValueError:
        return False


def percentage_to_fraction(text):
    pattern = r"(\d+(\.\d+)?%)"
    matches = re.findall(pattern, text)
    for match in matches:
        percentage_str = match[0]
        percentage = float(percentage_str.strip("%")) / 100
        fraction = str(percentage)
        text = text.replace(percentage_str, fraction)
    return text


def clean_expr_str(expr_str):
    expr_str = (
        expr_str.replace(" . ", ".")
        .replace(". ", ".")
        .replace("**", "^")
        .replace("\\pm", "")
        .replace("*", "\\times ")
        .replace("\\\\", "\\")
        .replace("\\ne ", "\\neq ")
        .replace("!=", "\\neq")
        .replace(">=", "\\ge")
        .replace("<=", "\\le")
        .replace("≠", "\\neq")
        .replace("dfrac", "frac")
        .replace("tfrac", "frac")
        .replace("\\$", "")
        .replace("$", "")
        .replace("\\%", "")
        .replace("%", "")
        .replace("\\!", "")
        .replace("^\circ", "\\times \\pi / 180")
        .replace("//", "/")
        .replace('"', "")
        # .replace(",", "") # TODO
    )
    # expr_str = re.sub(r"\^\s(.*)", r"\^\s{\1}", expr_str)
    expr_str = re.sub(r"\\+", r"\\", expr_str)
    expr_str = re.sub(r"\^\s?\((.*?)\)", r"^{\1}", expr_str)
    expr_str = re.sub(r"\\frac\s?(\d)\s?(\d+)", r"\\frac{\1}{\2}", expr_str)
    expr_str = re.sub(r"\\log_\s?(\d)\s?(\d+)", r"\\log_{\1}{\2}", expr_str)
    expr_str = re.sub(r"\\frac\s?{(.*?)}\s?(\d)", r"\\frac{\1}{\2}", expr_str)
    expr_str = re.sub(r"\\frac\s?(\d)\s?{(.*?)}", r"\\frac{\1}{\2}", expr_str)
    expr_str = re.sub(r"\\sqrt\s?(\d)", r"\\sqrt{\1}", expr_str)
    expr_str = re.sub(r"sqrt\s?\((\d+)\)", r"\\sqrt{\1}", expr_str)
    expr_str = re.sub(r"sqrt\s?\((.*?)\)", r"\\sqrt{\1}", expr_str)
    expr_str = expr_str.replace(" sqrt", "\\sqrt")
    expr_str = (
        expr_str.replace("\\left", "").replace("\\right.", "").replace("\\right", "")
    )
    return expr_str


def parse_latex_answer(sample):
    if isinstance(sample, int) or isinstance(sample, float):
        sample = str(sample)
    #     return sample
    sample = clean_expr_str(sample)
    try:
        expr = my_parse_latex(sample)
    except:
        print("[parse failed]", sample)
        return None
    return expr


def my_equals(ans_p, ans_l):
    return ans_p.equals(ans_l)


def is_expr_equal(ans_p, ans_l, is_strict=False):
    def is_equ_num_equal(equation, number):
        if (
            isinstance(equation, sp.Eq)
            # and isinstance(equation.lhs, sp.Symbol)
            and equation.rhs.is_number
            and number.is_number
        ):
            try:
                ret = my_equals(equation.rhs, number)
                return bool(ret)
            except:
                return equation.rhs == number

    if ans_p is None or ans_l is None:
        return False
    if isinstance(ans_l, str):
        return ans_p == ans_l

    if (
        not is_strict
        and is_equ_num_equal(ans_l, ans_p)
        or is_equ_num_equal(ans_p, ans_l)
    ):
        return True

    if ans_p.free_symbols != ans_l.free_symbols:
        return False

    if ans_p == ans_l:
        return True

    if isinstance(ans_l, sp.core.relational.Relational):
        try:
            if (
                type(ans_l) == type(ans_p)
                and my_equals(ans_p.lhs, ans_l.lhs)
                and my_equals(ans_p.rhs, ans_l.rhs)
            ):
                return True
        except Exception as e:
            print(ans_p, ans_l, e)
    try:
        ret = my_equals(ans_p, ans_l)
        return bool(ret)
    except:
        return False


# @timeout_decorator.timeout(5)
# def compare_ans(ans_p_str, ans_l_str, is_strict=False):
#     ans_p_str = clean_expr_str(ans_p_str)
#     ans_p_str = ans_p_str.replace(",", "").replace("$", "")
#     ans_l_str = clean_expr_str(ans_l_str)
#     ans_l_str = ans_l_str.replace(",", "").replace("$", "")
#     if ans_p_str is None:
#         return False
#     if ans_p_str.replace(" ", "") == ans_l_str.replace(" ", ""):
#         return True
#     ans_p = parse_latex_answer(ans_p_str)
#     if ans_p is None:
#         return False
#     ans_l = parse_latex_answer(ans_l_str)
#     if ans_l is None:
#         return False
#     return is_expr_equal(ans_p, ans_l, is_strict=is_strict)


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return ""
    return pred[-1]


@timeout_decorator.timeout(5)
def compare_ans(ans_p_str, ans_l_str, is_strict=False):
    ans_p_str = clean_expr_str(ans_p_str)
    ans_p_str = ans_p_str.replace(",", "").replace("$", "")
    ans_l_str = clean_expr_str(ans_l_str)
    ans_l_str = ans_l_str.replace(",", "").replace("$", "")
    if ans_p_str is None:
        return False
    if ans_p_str.replace(" ", "") == ans_l_str.replace(" ", ""):
        return True
    ans_p = parse_latex_answer(ans_p_str)
    if ans_p is None:
        return False
    ans_l = parse_latex_answer(ans_l_str)
    if ans_l is None:
        return False
    if is_expr_equal(ans_p, ans_l, is_strict=is_strict):
        return True
    # TODO not suitable
    ans_p_str = extract_answer_number(ans_p_str)
    if is_number(ans_p_str):
        ans_p = parse_latex_answer(ans_p_str)
        if is_expr_equal(ans_p, ans_l, is_strict=is_strict):
            return True
    return False


def vote(answers):
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def contains_number(s):
    return any(i.isdigit() for i in s)


def rough_compare_ans(generation, answer):
    for line in generation.split("\n")[::-1]:
        if contains_number(line):
            break
    words = line.split()
    for i, w in enumerate(words):
        if i > 0 and words[i - 1] in ["+", "-", "*", "/", "^"]:
            continue
        if i < len(words) - 1 and words[i + 1] in ["+", "-", "*", "/", "^"]:
            continue
        if not contains_number(w):
            continue
        if compare_numerical_ans(w.replace("$", ""), answer) and "=" not in " ".join(
            w[i:]
        ):
            return 1
    return 0


if __name__ == "__main__":
    pred = "To find the minimum value of $f(x)$, we need to find the critical points where the derivative of $f(x)$ is either zero or undefined. Let's write a function to solve this problem:\n\n```python\nfrom sympy import symbols, diff, solve\n\ndef find_minimum_value():\n    x, p = symbols('x p')\n    f = abs(x - p) + abs(x - 15) + abs(x - p - 15)\n    f_prime = diff(f, x)\n    critical_points = solve(f_prime, x)\n    \n    # Evaluate f(x) at the critical points and endpoints\n    values = [f.subs(x, p), f.subs(x, 15)]\n    for point in critical_points:\n        values.append(f.subs(x, point))\n    \n    # Find the minimum value\n    minimum_value = min(values)\n    \n    return minimum_value\n\nresult = find_minimum_value()\nprint(result)\n```\n```output\nValueError(\"\\nCan't calculate derivative wrt re(x) + I*im(x).\")\n```\nIt seems that there is an error in the code. Let's modify the code to handle the case when the derivative is undefined.\n\n```python\nfrom sympy import symbols, diff, solve, oo\n\ndef find_minimum_value():\n    x, p = symbols('x p')\n    f = abs(x - p) + abs(x - 15) + abs(x - p - 15)\n    f_prime = diff(f, x)\n    critical_points = solve(f_prime, x)\n    \n    # Evaluate f(x) at the critical points and endpoints\n    values = [f.subs(x, p), f.subs(x, 15)]\n    for point in critical_points:\n        if point.is_real:\n            values.append(f.subs(x, point))\n    \n    # Find the minimum value\n    minimum_value = min(values)\n    \n    return minimum_value\n\nresult = find_minimum_value()\nprint(result)\n```\n```output\nValueError(\"\\nCan't calculate derivative wrt re(x) + I*im(x).\")\n```\nApologies for the error in the code. It seems that the derivative calculation is causing an issue. Let's modify the code to handle the absolute value function and find the minimum value using a different approach.\n\n```python\nfrom sympy import symbols, Piecewise, Min, oo\n\ndef find_minimum_value():\n    x, p = symbols('x p')\n    f = abs(x - p) + abs(x - 15) + abs(x - p - 15)\n    \n    # Define the piecewise function for f(x)\n    f_piecewise = Piecewise((f, x <= p), (f.subs(x, p), p < x <= 15), (f.subs(x, 15), x > 15))\n    \n    # Find the minimum value\n    minimum_value = Min(f_piecewise, (x, p, 15))\n    \n    return minimum_value\n\nresult = find_minimum_value()\nprint(result)\n```\n```output\nTypeError('cannot determine truth value of Relational')\n```\nApologies for the error in the code. It seems that the piecewise function is causing an issue. Let's modify the code to handle the absolute value function and find the minimum value using a different approach.\n\n```python\nfrom sympy import symbols, Min, oo\n\ndef find_minimum_value():\n    x, p = symbols('x p')\n    f = abs(x - p) + abs(x - 15) + abs(x - p - 15)\n    \n    # Find the minimum value\n    minimum_value = Min(f, (x, p, 15))\n    \n    return minimum_value\n\nresult = find_minimum_value()\nprint(result)\n```\n"
    label = "15"
    print(compare_ans(pred, label))
