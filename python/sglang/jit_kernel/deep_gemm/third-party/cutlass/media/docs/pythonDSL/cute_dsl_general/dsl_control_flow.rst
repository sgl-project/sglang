.. _dsl_control_flow:
.. |DC|        replace:: dynamic compilation
.. |IR|        replace:: intermediate representation (IR)
.. |DSL|       replace:: CuTe DSL
.. |Constexpr| replace:: **Constexpr** (compile-time Python value)

Control Flow
==================


Overview
--------
|DSL| walks Python's AST and converts each control-flow construct it finds into
structured |IR|.  You can therefore write ordinary Python loops and branches
while the compiler decides—statement by statement—whether to

* **evaluate at compile time** if it's a native Python control flow, or
* **emit intermediate representation (IR)** when the control flow is marked as dynamic.

Passing |IR| values to a native Python control flow will result in an error.

For a high-level discussion of the overall pipeline, see
:doc:`the code-generation overview <dsl_code_generation>`.


For Loops
---------
|DSL| recognises three kinds of ranges for ``for`` loops:

* ``range`` – the Python built-in, always lowered to |IR|
* ``cutlass.range`` - Same as Python built-in ``range``, but supports advanced unrolling and pipelining control
* ``cutlass.range_constexpr`` – unrolled at compile time


range(...)/cutlass.range(...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use when you *always* want a loop in the generated |IR|, even if the inputs
are Python values.

cutlass.range_constexpr(...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Runs in the Python interpreter and is fully unrolled before code generation.
All loop indices must be |Constexpr|.


**Example:**

.. code-block:: python

    @cute.jit
    def control_flow_examples(bound: cutlass.Int32):
        n = 10

        # ✅ This loop is Python loop, evaluated at compile time.
        for i in cutlass.range_constexpr(n):
            cute.printf("%d\\n", i)

        # ✅ This loop is dynamic, even when bound is Python value.
        for i in range(n):
            cute.printf("%d\\n", i)

        # ❌ This loop bound is a dynamic value, not allowed in Python loop.
        # Should use `range` instead.
        for i in cutlass.range_constexpr(bound):
            cute.printf("%d\\n", i)

        # ✅ This loop is dynamic, emitted IR loop.
        for i in range(bound):
            cute.printf("%d\\n", i)

        # ✅ This loop is dynamic, emitted IR loop with unrolling
        for i in cutlass.range(bound, unroll=2):
            cute.printf("%d\\n", i)

Software Pipelining
~~~~~~~~~~~~~~~~~~~

Software pipelining is a technique used to optimize loops. Typically, this involves writing a prefetch loop and a main loop.

.. code-block:: python

    @cute.jit
    def example():
        ...
        # build a circular buffer
        buffer = ...

        # prefetch loop
        for i in range(prefetch_stages):
            cute.copy(atom, gmem[i], buffer[i], ...)

        # main loop
        for i in range(bound):
            if i + prefetch_stages < bound:
                cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)

            use(buffer[i % total_stages])

        ...

This can be tedious to write and tune. |DSL| provides a loop attribute to ask the compiler to do this.

.. code-block:: python

    @cute.jit
    def example():
        ...
        # build a circular buffer
        buffer = ... 

        for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
            # Compiler automatically handles the pipelining:
            # - Generates prefetch loop for initial stages
            # - In main loop, prefetches future data while using current data
            cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
            use(buffer[i % total_stages])  # Uses data from previous iterations
        
        ...

Compiler will automatically generate the prefetch loop with `prefetch_stages` iterations and a corresponding main loop.

This feature is experimental and only supported on sm90 and above.


If-Else Statements
------------------

Standard Python ``if``/``elif``/``else`` is supported.

* **Predicate without annotation** → lowered to |IR|.
* **Predicate annotated with `cutlass.const_expr`** → evaluated at compile time.

**Example:**

.. code-block:: python

    @cute.jit
    def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
        # ✅ This branch is Python branch, evaluated at compile time.
        if cutlass.const_expr(const_var):
            cute.printf("Const branch\\n")
        else:
            cute.printf("Const else\\n")

        # ✅ This branch is dynamic branch, emitted IR branch.
        if dynamic_var == 10:
            cute.printf("Dynamic True\\n")
        else:
            cute.printf("Dynamic False\\n")

        # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
        if cutlass.const_expr(dynamic_var == 10):
            cute.printf("Bound is 10\\n")


While Loops
-----------

Standard Python ``while`` is supported.

* **Condition without annotation** → lowered to |IR|.
* **Condition annotated with `cutlass.const_expr`** → evaluated at compile time.

**Example:**

.. code-block:: python

    @cute.jit
    def main(dynamic_var: cutlass.Int32):
        n = 0

        # ✅ This is Python while loop, evaluated at compile time.
        while cutlass.const_expr(n < 10):
            cute.printf("Const branch\\n")
            n += 1

        # ✅ This is dynamic while loop, emitted IR while loop.
        while dynamic_var == 10:
            cute.printf("Dynamic True\\n")
            n += 1

        # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
        while cutlass.const_expr(n < dynamic_var):
            n += 1


Compile-Time Metaprogramming
----------------------------

Mix compile-time constructs with normal |DSL| code to generate specialised
kernels without runtime overhead.  A compile-time flag can, for example, toggle
an optional **ReLU** epilogue:

.. code-block:: python

   @cute.kernel
   def gemm(..., do_relu: cutlass.Constexpr):
       # main GEMM work
       ...
       if cutlass.const_expr(do_relu):    # compile-time guard
           # ReLU code is emitted only when do_relu is True
           ...

.. code-block:: text

   gemm(..., False)   # ReLU is omitted from the generated |IR|
   gemm(..., True)    # ReLU is included


Limitations of Dynamic Control Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Early-exit ``break``, ``continue``, ``pass`` or raising exception from
  control flow body are not yet supported.
* Operations in the control flow body are traced only when tracing is active in
  that region.
* Values originating in control flow body are not available outside the control
  flow.
* Changing type of a variable in control flow body is not allowed.

**Example:**

.. code-block:: python

    @cute.jit
    def control_flow_negative_examples(predicate: cutlass.Boolean):
        n = 10

        # ❌ This loop is dynamic, early-exit isn't allowed.
        for i in range(n):
            if i == 5:
                break         # Early-exit

        if predicate:
            val = 10
            # ❌ return from control flow body is not allowed.
            return
            # ❌ Raising exception from control flow body is not allowed.
            raise ValueError("This is not allowed")
            # ❌ Using pass in control flow body is not allowed.
            pass

        # ❌ val is not available outside the dynamic if
        cute.printf("%d\\n", val)

        if predicate:
            # ❌ Changing type of a variable in control flow body is not allowed.
            n = 10.0

