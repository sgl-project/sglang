"""
Sort the test case by name alphabetically for run_suite.py
"""

from dataclasses import dataclass


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


suites = {}


if __name__ == "__main__":
    for key in suites:
        cases = suites[key]
        names = [x.name for x in cases]
        names.sort()

        print(f'    "{key}": [')
        for name in names:
            estimated_time = [x.estimated_time for x in cases if x.name == name][0]
            print(f'        TestFile("{name}", {estimated_time}),')
        print(f"    ],\n")
