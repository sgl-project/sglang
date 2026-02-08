#################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################


"""
Given a set of test files to be included in a CMake target, this script extracts
the TEST definitions from each file, writes them into new files, and prints the names
of the new files so that they can be processed as part of a new CMake target.

For example, given a set of --src_files test_a.cu test_b.cu containing 3 and 2 TEST
definitions, respectively, this script would produce:
    test_a_000.cu
    test_a_001.cu
    test_a_002.cu
    test_b_000.cu
    test_b_001.cu

The splitting follows a fairly rudimentary algorithm that does not support all valid C++ programs.
We walk through a given input test file line by line. Any lines that are not within a TEST definition is added to a running
"filler" text. When a TEST definition is encountered, the current filler text becomes the prefix
for that test. All subsequent lines are considered to be part of the TEST definition until the
number of starting function braces ('{') match the number of closing function braces ('}'). When
these counts are equal, the TEST definition is considered to be completed. At this point, we return
to adding lines to the "filler" text until a new TEST definition is encountered. Any "filler" text
following a TEST definition is added to the suffix of that TEST definition (this is useful for finishing
off #if statements, as is common in unit tests.).

A state machine illustrating this algorithm at a high level is provided in the source below.

Example: Suppose an input test `test.cu` has the following source:
    // COPYRIGHT
    #include <iostream>

    #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    // Test #1
    TEST(SM90_a, 256x128x64_2x2x1) {
        std::cout << "Test #1" << std::endl;
    }

    // Test #2
    TEST(SM90_b, 256x128x64_1x1x1) {
        std::cout << "Test #2" << std::endl;
    }

    #endif defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

The contents of the two resulting test files will be:
  $ cat test_000.cu
    // COPYRIGHT
    #include <iostream>

    #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    // Test #1
    TEST(SM90_a, 256x128x64_2x2x1) {
        std::cout << "Test #1" << std::endl;
    }

    // Test #2

    #endif defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  $ cat test_001.cu
    // COPYRIGHT
    #include <iostream>

    #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    // Test #1

    // Test #2
    TEST(SM90_b, 256x128x64_1x1x1) {
        std::cout << "Test #2" << std::endl;
    }

    #endif defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

Notice that each of test_000.cu and test_001.cu contain comments that appear outside
the TEST definitions not included in each file. This is by design, as these
would be considered "filler" text.

As expected, some cases can't be handled. Below is a non-exhaustive list:
    1. New TEST following the closing '}' of a TEST case on the same line:
        TEST(x, y) {
            // Do stuff
        } TEST(a, b) {

        In this case, "TEST(a, b) {" will be ignored

    2. Preprocessor macros that occur midway through a test case and extend
       beyond the conclusion of a testcase

       Example:
            TEST(a, b) {
                // Do stuff
        #if X
                // Do more stuff
            }
        #else
                // Do other stuff
            }
        #endif
"""


import argparse
import enum
import os


parser = argparse.ArgumentParser()
parser.add_argument("cmake_target", type=str,
                    help="Name of the CMake target being generated.")
parser.add_argument("src_dir", type=str,
                    help="Path to the directory containing test files.")
parser.add_argument("--src_files", nargs='+',
                    help="Files containing TEST instances to split.")
parser.add_argument("--max_tests_per_file", type=int, default=1,
                    help="Maximum number of TEST instances per file.")
parser.add_argument("--dst_dir", type=str,
                    help="Path to the directory to which to write new test files. If not set, uses src_dir.")
args = parser.parse_args()


if args.dst_dir == None:
    args.dst_dir = args.src_dir


class Testcase:
    """
    Lightweight tracker of test-case processing status
    """
    def __init__(self, prefix_text):
        # Any text that preceded the TEST definition that was
        # not part of another TEST definition
        self.prefix = prefix_text

        # Any text within the TEST definition
        self.test = ""

        # Any text that follows the completion of the TEST definition
        # and is not included in other TEST definitions
        self.suffix = ""

        # Whether the test's definition has concluded
        self.completed = False

        # Current balance of opening and closing curly brackets in
        # the TEST definition. '{' increments the count and '}' decrements it.
        # A value of 0 (when self.completed == False) indicates that the test
        # has completed.
        self.curly_bracket_balance = 0


class ParseState(enum.Enum):
    """
      State machine for processing.
      Transitions occur on each line encountered in the soruce file


      Line does not contain 'TEST('
                 +----+
                 |    |
                 |    v          'TEST('
               +--------+      encountered         +--------------------------+
        ------>| Filler | -----------------------> | TestDeclaredWaitingStart |
               +--------+                          +--------------------------+
                   ^                                         |
 Number of '{'     |                                         | First '{' encountered
 equals number of  |           +--------+                    |
 '}' encountered   +-----------| InTest | <------------------+
                               +--------+
                                 |    ^
                                 |    |
                                 +----+
                      Number of '{' encountered
                      exceeds number of '}' encountered
    """


    # Any text that is not part of a TEST case
    Filler = 0

    # Processing text within the first { of the TEST case
    # and before the en of the final } of the TEST case
    InTest = 1

    # Processing text from the start of the TEST definition
    # but before the first {. This could occur if the opening {
    # occurs on a separate line than the TEST definition.
    TestDeclaredWaitingStart = 2


cmake_src_list = []
for filename in args.src_files:
    if '.' not in filename:
        # Add any non-filename arguments to the command list by default
        cmake_src_list.append(filename)
        continue

    if '/' in filename:
        raise Exception(
            f"Source files passed to {__file__} must be within the same directory "
            "as the CMakeLists defining the target using the files. "
            f"Provided path {filename} is in a different directory.")

    full_filename = os.path.join(args.src_dir, filename)
    with open(full_filename, 'r') as infile:
        lines = infile.readlines()

    # Find the number of instances of "TEST("
    ntest = sum([1 for line in lines if "TEST(" in line])

    if ntest <= args.max_tests_per_file:
        # File contains fewer than max_tests_per_file TEST instances. It does
        # not need to be split
        cmake_src_list.append(filename)
        continue

    # Current state of the parsing state machine. We start with filler text
    state = ParseState.Filler

    # List of individual TESTs found
    tests = []

    # Ongoing text that is not included in a TEST definition. This will serve
    # as the prefix for any yet-to-be encountered TEST definitions.
    filler_text = ""

    def add_filler_text(text):
        global filler_text
        # Add new text to the ongoing filler text and to the suffixes of
        # any completed tests
        filler_text += text
        for i in range(len(tests)):
            if tests[i].completed:
                tests[i].suffix += text

    for line in lines:
        if state == ParseState.Filler:
            # We are not currently within a TEST definition.

            if 'TEST(' in line:
                # We have encountered a new TEST( case. Any text preceding this
                # must be added to the filler text (e.g., if we have a line of the form:
                #   "static constexpr int Val = 4; TEST(blah) {"
                #   then "static constexpr int Val = 4;" needs to be included in filler
                #   text, as it could be used by subsequent tests.)
                splits = line.split('TEST')

                # There should not be more than one TEST definition on a given line
                assert len(splits) <= 2

                if len(splits) > 1:
                    if not splits[0].isspace():
                        # Only add text to filler if there are non-whitespace charcters
                        # preceding the TEST definition in the line
                        filler_text += splits[0]

                        # The new line is just the TEST-related line
                        line = 'TEST' + splits[-1]

                # Add tests and transtion to TestDeclaredWaitingStart state.
                # Do not add the line to the test text of the new test case; this
                # will be done in either the TestDeclaredWaitingStart state processing
                # below or in the InTest state processing below.
                tests.append(Testcase(filler_text))
                state = ParseState.TestDeclaredWaitingStart
            else:
                # Any remaining filler text is added to the running filler_text
                # which will be used as the prefix for any new tests, and to the
                # suffix of any completed tests
                add_filler_text(line)

        if state == ParseState.TestDeclaredWaitingStart:
            # We have seen a TEST definition but have not yet seen its opening {.

            if '{' in line:
                # The first curly bracket for the TEST definition has been found.
                # Advance to state InTests. Do not add the line to the test's text
                # or change the curly-brace balance of the test; these will be done
                # when processing the state == ParseState.InTest condition below.
                state = ParseState.InTest
            else:
                tests[-1].test += line

        if state == ParseState.InTest:
            # We are currently within a TEST definition.
            # Process lines character-by-character looking for opening and closing
            # braces. If we reach parity between opening and closing braces, the
            # test is considered done.
            filler_text_to_add = ""
            for char in line:
                if not tests[-1].completed:
                    tests[-1].test += char
                    if char == '{':
                        tests[-1].curly_bracket_balance += 1
                    elif char == '}':
                        tests[-1].curly_bracket_balance -= 1
                        if tests[-1].curly_bracket_balance == 0:
                            tests[-1].completed = True
                else:
                    filler_text_to_add += char

            if filler_text_to_add != "" and (not filler_text_to_add.isspace() or '\n' in filler_text_to_add):
                add_filler_text('\n' + filler_text_to_add)

            if tests[-1].completed:
                state = ParseState.Filler

    # Write out the new files for tests
    filename_prefix, filename_suffix = filename.split('.')
    for i, test in enumerate(tests):
        assert test.completed
        new_filename = filename_prefix + '_' + str(i).zfill(3) + '.' + filename_suffix
        full_new_filename = os.path.join(args.dst_dir, new_filename)

        # Replace any '\' with '/'. CMake doesn't like '\'.
        full_new_filename = full_new_filename.replace('\\', '/')

        with open(full_new_filename, 'w') as outfile:
            outfile.write(test.prefix + test.test + test.suffix)
        cmake_src_list.append(full_new_filename)


for cmake_file in cmake_src_list:
    print(cmake_file)
