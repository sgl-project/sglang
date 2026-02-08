.. _dsl_code_generation:
.. |DC|  replace:: dynamic compilation
.. |DSL| replace:: CuTe DSL
.. |IR|  replace:: intermediate representation (IR)

End-to-End Code Generation
==========================


1. Techniques for Turning Python into |IR|
------------------------------------------

1.1 AST rewrite
^^^^^^^^^^^^^^^^
The function’s abstract-syntax tree is analysed **before** execution.
Python control-flow (``for``/``while``, ``if``/``else``) and built-ins are converted to structured |IR|
constructs.  Computation inside each region is left untouched at this stage.

*Advantages*

*  Sees the entire program, so every branch and loop is preserved.
*  Keeps loop structure intact for optimization such as tiling, vectorisation
   or GPU thread mapping.

*Disadvantages*

*  Requires a well-defined Python subset that the rewriter understands.


1.2 Tracing
^^^^^^^^^^^
The decorated function is executed once with *proxy* arguments; overloaded
operators record every tensor operation that actually runs and produce a flat
trace that is lowered to |IR|.

*Advantages*

*  Near-zero compile latency, ideal for straight-line arithmetic.
*  No need to parse Python source, so it supports many dynamic Python
   features, and Python has many features.

*Disadvantages*

*  Untaken branches vanish, so the generated kernel may be wrong for other
   inputs.
*  Loops are flattened to the iteration count observed during tracing.
*  Data-dependent control-flow freezes to a single execution path.


2. |DSL| Code-Generation Modes
------------------------------

CuTe’s Python front-end combines the techniques above into **two mutually
exclusive modes**, selectable with the ``preprocessor`` flag of the
``@jit`` decorator:

1. Tracing mode ``@jit(preprocess=False)`` – tracing only.
This results in the fastest compilation path and is recommended only for kernels that are guaranteed to be
straight-line arithmetic. It suffers from all tracing limitations listed in the previous section.

2.  Preprocessor mode (**default**) ``@jit(preprocess=True)`` – **AST rewrite + tracing**.
The AST pass captures every loop and branch, eliminating the correctness and
optimisation problems of pure tracing; tracing then fills in the arithmetic.
This hybrid “preprocessor” pipeline is unique to |DSL| and was designed
specifically to overcome the disadvantages identified above.

.. figure:: dsl_modes.png
   :width: 400
   :align: center

   *Left*: tracing mode records only the path that executed.
   *Right*: preprocessor mode emits structured |IR| for every branch and loop
   before tracing the arithmetic.


Why Tracing-Only Is Insufficient for Control-Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Branch loss** – The untaken side of an ``if``/``else`` is never lowered.
* **Loop unrolling** – Loops are flattened to the iteration count observed,
  destroying structure needed for parallel mapping and tiling.
* **Data-dependent paths** – Control-flow that depends on tensor values freezes
  to a single execution path at trace time.

The preprocessor mode fixes all of these by lowering control-flow first and delegating
only the arithmetic to the tracer.
