.. _debugging:

Debugging
=========

This page provides an overview of debugging techniques and tools for CuTe DSL programs.


Getting Familiar with the Limitations
-------------------------------------

Before diving into comprehensive debugging capabilities, it's important to understand the limitations of CuTe DSL.
Understanding these limitations will help you avoid potential pitfalls from the start.

Please refer to :doc:`../limitations` for more details.


DSL Debugging
-------------

CuTe DSL provides built-in logging mechanisms to help you understand the code execution flow and
some of the internal state.

Enabling Logging
~~~~~~~~~~~~~~~~

CuTe DSL provides environment variables to control logging level:

.. code:: bash

    # Enable console logging (default: False)
    export CUTE_DSL_LOG_TO_CONSOLE=1

    # Log to file instead of console (default: False)
    export CUTE_DSL_LOG_TO_FILE=my_log.txt

    # Control log verbosity (0, 10, 20, 30, 40, 50, default: 10)
    export CUTE_DSL_LOG_LEVEL=20


Log Categories and Levels
~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to standard Python logging, different log levels provide varying degrees of detail:

+--------+-------------+
| Level  | Description |
+========+=============+
| 0      | Disabled    |
+--------+-------------+
| 10     | Debug       |
+--------+-------------+
| 20     | Info        |
+--------+-------------+
| 30     | Warning     |
+--------+-------------+
| 40     | Error       |
+--------+-------------+
| 50     | Critical    |
+--------+-------------+


Dump the generated IR
~~~~~~~~~~~~~~~~~~~~~

For users familiar with MLIR and compilers, CuTe DSL supports dumping the Intermediate Representation (IR).
This helps you verify whether the IR is generated as expected.

.. code:: bash

    # Dump Generated CuTe IR (default: False)
    export CUTE_DSL_PRINT_IR=1

    # Keep Generated CuTe IR in a file (default: False)
    export CUTE_DSL_KEEP_IR=1



Kernel Functional Debugging
----------------------------

Using Python's ``print`` and CuTe's ``cute.printf``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTe DSL programs can use both Python's native ``print()`` as well as our own ``cute.printf()``  to
print debug information during kernel generation and execution. They differ in a few key ways:

- Python's ``print()`` executes during compile-time only (no effect on the generated kernel) and is
  typically used for printing static values (e.g. a fully static layouts).
- ``cute.printf()`` executes at runtime on the GPU itself and changes the PTX being generated. This
  can be used for printing values of tensors at runtime for diagnostics, but comes at a performance
  overhead similar to that of `printf()` in CUDA C.

For detailed examples of using these functions for debugging, please refer to the associated
notebook referenced in :doc:`notebooks`.

Handling Unresponsive/Hung Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a kernel becomes unresponsive and ``SIGINT`` (``CTRL+C``) fails to terminate it,
you can follow these steps to forcefully terminate the process:

1. Use ``CTRL+Z`` to suspend the unresponsive kernel
2. Execute the following command to terminate the suspended process:

.. code:: bash

    # Terminate the most recently suspended process
    kill -9 $(jobs -p | tail -1)


CuTe DSL can also be debugged using standard NVIDIA CUDA tools.

Using Compute-Sanitizer
~~~~~~~~~~~~~~~~~~~~~~~

For detecting memory errors and race conditions:

.. code:: bash

    compute-sanitizer --some_options python your_dsl_code.py

Please refer to the `compute-sanitizer documentation <https://developer.nvidia.com/compute-sanitizer>`_ for more details.

Conclusion
----------

This page covered several key methods for debugging CuTe DSL programs. Effective debugging typically requires a combination of these approaches.
If you encounter issues with DSL, you can enable logging and share the logs with the CUTLASS team as a GitHub issue to report a bug.
