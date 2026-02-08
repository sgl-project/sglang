.. _dsl_dynamic_layout:
.. |DSL| replace:: CuTe DSL
.. |SLAY| replace:: static layout
.. |DLAY| replace:: dynamic layout

Static vs Dynamic layouts
=========================

Static Layout
-------------

When integrating with popular deep learning frameworks, one question is how to deal with the layout of the converted ``cute.Tensor``.
For example, when converting a ``torch.Tensor`` to a ``cute.Tensor``, the shape of the ``torch.Tensor`` is honored for the layout of
``cute.Tensor``.

.. code-block:: python

    import torch
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    @cute.jit
    def foo(tensor):
        print(f"tensor.layout: {tensor.layout}")  # Prints tensor layout at compile time
        cute.printf("tensor: {}", tensor)         # Prints tensor values at runtime

In this example, we define a JIT function ``foo`` that takes a ``cute.Tensor`` as input and prints its layout. Note
that Python print is used to print the layout at compile time. This works fine for |SLAY| whose value is known at
compile time.

Now let's try to run the JIT function ``foo`` with different shapes of the input ``torch.Tensor``.

.. code-block:: python

    a = torch.tensor([1, 2, 3], dtype=torch.uint16)
    a_pack = from_dlpack(a)
    compiled_func = cute.compile(foo, a_pack)
    compiled_func(a_pack)

Here we first convert a 1D ``torch.Tensor`` with 3 elements to a ``cute.Tensor`` using ``from_dlpack``. Then we compile
the JIT function ``foo`` with the converted ``cute.Tensor`` and call the compiled function. 

::

    tensor.layout: (3):(1)
    tensor: raw_ptr(0x00000000079e5100: i16, generic, align<2>) o (3):(1) = 
  ( 1, 2, 3 )

It prints ``(3):(1)`` for the layout because the converted ``cute.Tensor`` has a |SLAY| with shape ``(3)`` which
is the shape of the ``a``.

Now if we call the compiled function with a different shape of the input ``torch.Tensor``, it would result in an unexpected
result at runtime due to the mismatch of the type since ``compiled_func`` expects a ``cute.Tensor`` with layout ``(3):(1)``
while ``b`` has shape ``(5)``.

.. code-block:: python

    b = torch.tensor([11, 12, 13, 14, 15], dtype=torch.uint16)
    b_pack = from_dlpack(b)
    compiled_func(b_pack)  # ❌ This results in an unexpected result at runtime due to type mismatch

Following is the output which is unexpected due to the type mismatch.

::

    tensor: raw_ptr(0x00000000344804c0: i16, generic, align<2>) o (3):(1) = 
  ( 11, 12, 13 )

To fix that, we would have to trigger another code generation and compilation for the new shape for ``b``.

.. code-block:: python

    compiled_func_2 = cute.compile(foo, b_pack)  # This would trigger another compilation
    compiled_func_2(b_pack)                      # ✅ Now this works fine

As shown in the example above, with the newly compiled ``compiled_func_2``,  we can pass in ``b_pack`` to the compiled
JIT function ``compiled_func_2``.

::

    tensor.layout: (5):(1)
    tensor: raw_ptr(0x0000000034bb2840:: i16, generic, align<2>) o (5):(1) = 
  ( 11, 12, 13, 14, 15 )

Now it recompiles and prints the values of ``b`` correctly.

It's obvoius that we need distinct codes generated and compiled for different static layout. In this case, one for layout
``(3):(1)`` and the other for layout ``(5):(1)``.

Dynamic Layout
--------------

In order to avoid generating and compiling multiple times for different shapes of the input ``torch.Tensor``, |DSL| provides a way to
generate and compile JIT function with |DLAY|.

To get dyanmic layout of the ``cute.Tensor``, a ``torch.Tensor`` object can be passed into the JIT function directly which instructs
|DSL| to call ``cute.mark_layout_dynamic`` automatically on the converted ``cute.Tensor`` per the leading dimension of the layout.

.. code-block:: python

    import torch
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    @cute.jit
    def foo(tensor):
        print(tensor.layout)  # Prints (?,?):(?,1) for dynamic layout

    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
    compiled_func = cute.compile(foo, a)
    compiled_func(a)

    b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
    compiled_func(b)  # Reuse the same compiled function for different shape

In the example above, a single compilation of the JIT function ``foo`` is reused for different shapes of the input ``torch.Tensor``.
This is possible because the converted ``cute.Tensor`` has a |DLAY| ``(?,?):(?,1)`` which is compatible with the shape of the
input ``torch.Tensor`` of both calls.

Alternatively, for compact layout, ``cute.mark_compact_shape_dynamic`` can be called for a finer-grained control to specify the mode
of the layout for dynamic and the divisibility constraint for the dynamic dimension.

Refer to :doc:`framework_integration` for more details on ``from_dlpack``, ``mark_layout_dynamic``,
and ``mark_compact_shape_dynamic``.

Static Layout vs. Dynamic Layout
--------------------------------

Per the previous sections, we have seen that |SLAY| leads to distinct JIT code generations while |DLAY| leads to a single
compilation for different shapes.

That said, creating JIT function with |SLAY| is useful when the use cases targeting input data with fixed shapes.
Since more information is available at compile time, the compiler would be able to kick in optimizations that otherwise would not
be possible for the code generated for |DLAY|.

On the other hand, |DLAY| would be more flexible for the cases where the input data has varying shapes. This provides more
scalability of the generated code to deal with varying input data of different shapes.

Programming with Static and Dynamic Layout
------------------------------------------

|DSL| provides intuitive way to program with static and |DLAY| in the codes.

.. code-block:: python

    import torch
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    @cute.jit
    def foo(tensor, x: cutlass.Constexpr[int]):
        print(cute.size(tensor))  # Prints 3 for the 1st call
                                  # Prints ? for the 2nd call
        if cute.size(tensor) > x:
            cute.printf("tensor[2]: {}", tensor[2])
        else:
            cute.printf("tensor size <= {}", x)

    a = torch.tensor([1, 2, 3], dtype=torch.uint16)
    foo(from_dlpack(a), 3)   # First call with static layout

    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
    foo(b, 3)                # Second call with dynamic layout

In this example, the JIT function ``foo`` is compiled with a |SLAY| ``(3):(1)`` for the first call, which means the
size of the tensor is known at compile time. |DSL| makes good use of this and automatically handles the if condition at the
compile time. Hence the generated codes are efficient without the if condition at all.

For the second call, the JIT function ``foo`` is compiled with a |DLAY| ``(?):(1)`` hence the tensor size is only
evaluated at runtime. |DSL| automatically generates the code to handle the |DLAY| and the if condition at runtime.

The same applies to loop as well:

.. code-block:: python

    @cute.jit
    def foo(tensor, x: cutlass.Constexpr[int]):
        for i in range(cute.size(tensor)):
            cute.printf("tensor[{}]: {}", i, tensor[i])

    a = torch.tensor([1, 2, 3], dtype=torch.uint16)
    foo(from_dlpack(a), 3)   # First call with static layout

    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
    foo(b, 3)                # Second call with dynamic layout

With the static layout in the first call, |DSL| is able to fully unroll the loop at compile time. While in the second call,
the generated codes will have the loop executed at runtime based on the |DLAY|.

With the single JIT function implementation, |DSL| is able to handle control-flow constructs and automatically generate
the optimized codes for different cases. This is all possible because |DSL| is able to walk the Python AST and convert
each control-flow construct it finds accordingly.

Please refer to :doc:`dsl_control_flow` for more details.
