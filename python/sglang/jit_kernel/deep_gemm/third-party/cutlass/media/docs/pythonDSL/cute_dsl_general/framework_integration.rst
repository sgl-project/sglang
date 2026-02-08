.. _framework_integration:
.. |DSL| replace:: CuTe DSL

Integration with Frameworks
=============================

In order to facilitate the integration of CUTLASS Python with popular frameworks, we leverage the
`DLPack protocol <https://github.com/dmlc/dlpack>`_ and transform tensors originating from these
frameworks to CuTe tensors. The present page documents the conventions, the API available to the
user, and provide example code snippets for common usage patterns. We also provide a section on how to
bypass the DLPack protocol and directly call the JIT function.

Implicit Conversion
-------------------

Tensors originating from frameworks supporting the DLPack protocol can be directly provided to a
JIT function as a regular parameter. |DSL|'s  runtime implicitly converts the original tensor to a
CuTe tensor with a fully dynamic layout except for the stride element corresponding to the leading
dimension. The example below demonstrates this use case.

.. code-block:: python

    import torch
    import cutlass.cute as cute

    @cute.jit
    def foo(src):
        """
        The following lines print

        ptr<f32, generic> o (?,?,?):(?,?,1)
        <class 'cutlass.cute.core._Tensor'>
        """
        print(src)
        print(type(src))

    a = torch.randn(30, 20, 32, device="cpu")
    foo(a)


Explicit conversion using ``from_dlpack``
------------------------------------------

|DSL|'s runtime provides an interface for converting DLPack-compatible tensors to CuTe tensors,

.. code-block:: python

    b = cute.runtime.from_dlpack(a)

where ``a`` is a tensor supporting the DLPack protocol with the ``__dlpack__``
and ``__dlpack_device__`` methods. The resulting CuTe tensor ``b`` has a fully static layout. This
conversion is performed without copying any tensor data, enabling seamless integration with major
frameworks. Users can create tensors using NumPy, PyTorch, etc. and directly feed them into JIT
functions writtnen using |DSL|.

The resulting CuTe tensor shares the same underlying memory buffer as the original tensor. This
zero-copy approach maximizes performance by eliminating unnecessary data duplication. However, it is
important to note that the CuTe tensor's validity is tied to the lifetime of the original tensor. If
the source tensor is destroyed or goes out of scope, the corresponding CuTe tensor becomes invalid
since it references the original memory location.

The full signature of from_dlpack is as follows:

.. code-block:: python

    def from_dlpack(tensor, assumed_align=None):

The ``assumed_align`` integer parameter specifies the alignment of the tensor in unit of bytes.
The tensor's base address must be divisible by ``assumed_align``. When not provided explicitly,
the alignment is set to the natural alignment of the tensor's element type. Note that the alignment
information is part of the pointer type in the generated IR. Therefore, programs with different
alignments have a different IR and identical IRs are required for hitting the kernel caching
mechanism of |DSL|.

Code Example
~~~~~~~~~~~~

The following code demonstrates how to convert a PyTorch tensor to a CuTe tensor using the
``from_dlpack`` function with default parameters.

.. code-block:: python

    import torch
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    x = torch.randn(30, 20, device="cpu")
    y = from_dlpack(x)

Once converted, we can access the tensor's information through various
attributes. The following list shows the attributes of the converted tensor:

- ``tensor.shape``: the tensor's shape
- ``tensor.stride``: the tensor's stride
- ``tensor.memspace``: the tensor's memory space
- ``tensor.element_type``: the tensor's element data type

.. code-block:: python

    import torch
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    x = torch.randn(30, 20, device="cpu")
    y = from_dlpack(x)

    print(y.shape)        # (30, 20)
    print(y.stride)       # (20, 1)
    print(y.memspace)     # generic (if torch tensor in on device memory, memspace will be gmem)
    print(y.element_type) # Float32
    print(y)              # Tensor<0x000000000875f580@generic o (30, 20):(20, 1)>

The string format of the resulting CuTe tensor is

.. code-block::

    Tensor<0x{tensor.data_ptr:016x}@{tensor.memspace} o {tensor.shape}:{tensor.stride}>

As can be seen in the example above, ``from_dlpack`` first results in a tensor with a static layout.
To obtain dynamic or mixed static/dynamic layouts after calling ``from_dlpack``, the
``mark_layout_dynamic`` and ``mark_compact_shape_dynamic`` functions are used and described in
the following sections.

When to Use Explicit Conversion?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DLPack protocol is a widely used protocol for interoperability between different frameworks.
However, there is some associated overhead. Based on our benchmark, it usually takes between 2 to 3
us per call to ``from_dlpack``.

Explicit conversion allows for caching the converted CuTe tensors in order to avoid the overhead of
repeated calls to ``from_dlpack``.

.. code-block:: python

    x = torch.randn(30, 20, device="cpu")
    if key not in cached_tensors:
        # Do the conversion only for cache misses
        cached_tensors[key] = cute.runtime.from_dlpack(x)
    foo(cached_tensors[key])

Another use case for explicit conversion is to gain fine-grain control over which modes of a tensor
are considered dynamic from the perspective of the generated program.

Mark the Tensor's Layout as Dynamic with ``mark_layout_dynamic``
----------------------------------------------------------------

After calling this function, all shape modes become dynamic. The stride modes also become dynamic
with the following two exceptions:

1. the leading dimension's stride remains fixed at 1;
2. stride elements equal to 0 (which indicates broadcasting) are retained.

The full signature of ``mark_layout_dynamic`` is as follows:

.. code-block:: python

    def mark_layout_dynamic(self, leading_dim: int|None = None):

The ``leading_dim`` parameter specifies the leading dimension of the tensor. The leading dimension's
stride is set to 1 unless inconsistent with the layout of the DLPack tensor. For example,

- For a tensor with layout ``(2,2,3,4):(2,1,4,12)``, if ``leading_dim`` is specified to be 1,
  the layout will be marked as ``(?,?,?,?):(?,1,?,?)``.
- If ``leading_dim`` is specified to be 0, a deduction failure error is raised because the stride of
  dimension 0 is 2 (not 1).

The default value for ``leading_dim`` is ``None``.  In such case, the system
automatically deduces it from the tensor's layout using the following logic:

1. If a dimension's stride is 1, that dimension is marked as the leading dimension.
2. If multiple dimensions satisfy condition 1, an error is thrown indicating deduction failure.
   Note that after converting a **PyTorch** tensor to the DLPack format, the stride for dimensions
   with size 1 are canonicalized to 1. This canonicalization can increase the likelihood of
   deduction failures. This behavior is specific to PyTorch and does not occur with NumPy for
   example.
3. If no dimension satisfies condition 1, all strides are marked as dynamic.

For example:

- For a tensor with layout ``(2,2,3,4):(2,1,4,12)``, the leading dimension is 1.
  The layout will be marked as ``(?,?,?,?):(?,1,?,?)``.
- For a tensor with layout ``(1,5,1):(1,1,1)``, if ``leading_dim`` is not specified,
  a deduction failure error is raised.
- For a tensor with layout ``(2,2):(8,2)``, since no dimension has stride 1,
  all dimensions are marked as dynamic: ``(?,?):(?,?)``.

Code Example
~~~~~~~~~~~~

The following example demonstrates how to use ``mark_layout_dynamic`` to specify dynamic tensor layouts.

* ``t0`` shows the usage of ``mark_layout_dynamic`` with unspecified ``leading_dim`` and the automatic deduction of leading dimension.
* ``t1`` & ``t2`` shows the usage of ``mark_layout_dynamic`` with specified ``leading_dim``.
* ``t3`` shows the usage of ``mark_layout_dynamic`` with no leading dimension.
* ``t4`` shows the usage of ``mark_layout_dynamic`` with broadcasted dimensions.
* ``t5`` demonstrates the deduction failure when the there're more than one dimensions with stride equals to 1.
* ``t6`` & ``t7`` demonstrates incorrect settings for ``leading_dim`` and expected errors.

.. code-block:: python

    import torch
    from cutlass.cute.runtime import from_dlpack

    # (8,4,16,2):(2,16,64,1)
    a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
    # (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
    # resulting in (1,4,1,32,1):(1,1,1,4,1)
    b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)
    # (2,2):(8,2)
    c = torch.empty(3, 4)[::2, ::2]
    # (3,1,1,5):(5,0,0,1)
    d = torch.empty(3, 1, 1, 5).expand(3, 4, 2, 5)

    # auto deduce the leading dimension to be 3
    t0 = from_dlpack(a).mark_layout_dynamic()
    print(t0)
    # (?,?,?,?):(?,?,?,1)

    t1 = from_dlpack(b).mark_layout_dynamic(leading_dim=0)
    print(t2)
    # (?,?,?,?,?):(1,?,?,?,?)

    t2 = from_dlpack(b).mark_layout_dynamic(leading_dim=2)
    print(t3)
    # (?,?,?,?,?):(?,?,1,?,?)

    t3 = from_dlpack(c).mark_layout_dynamic()
    print(t3)
    # (?,?):(?,?)

    t4 = from_dlpack(d).mark_layout_dynamic()
    print(t4)
    # (?,?,?,?):(?,0,0,1)

    t5 = from_dlpack(b).mark_layout_dynamic()
    # Can't decude the leading dimension from layout, please specify the leading_dim explicitly.

    t6 = from_dlpack(a).mark_layout_dynamic(leading_dim=1)
    # Expected strides[leading_dim] == 1, but got 16

    t7 = from_dlpack(b).mark_layout_dynamic(leading_dim=3)
    # Expected strides[leading_dim] == 1, but got 4

Mark the Tensor's Layout as Dynamic with ``mark_compact_shape_dynamic``
-----------------------------------------------------------------------

The ``mark_compact_shape_dynamic`` function provides fine-grain control over dynamic shapes for compact
layouts. The full signature of ``mark_compact_shape_dynamic`` is as follows:

.. code-block:: python

    def mark_compact_shape_dynamic(self, mode: int, stride_order: tuple[int, ...]|None = None, divisibility: int = 1):

The ``mode`` parameter determines which shape dimension becomes dynamic. After calling this function,
the specific shape dimension given by ``mode`` is marked as dynamic immediately. The stride will be
updated accordingly. For modes that have a shape of size 1, their stride are canonicalized to 0.

The ``stride_order`` parameter specifies the ordering of strides in the tensor. It is consistent
with ``torch.Tensor.dim_order()`` and defaults to ``None``. The parameter indicates the order of
modes (dimensions) if the current layout were to be converted to row-major order. It starts from the
outermost to the innermost dimension when reading it from left to right. This parameter must be
explicitly set when the stride order cannot be automatically deduced from the tensor's layout, such
as when multiple dimensions have a stride of 1.

For example:

- Layout ``(4,2):(1,4)`` has a ``stride_order`` of ``(1,0)`` indicates the innermost dimension is
  0 (``4:1``), the outermost dimension is 1 (``2:4``).
- Layout ``(5,3,2,4):(3,1,15,30)`` has a ``stride_order`` of ``(3,2,0,1)`` indicates the innermost
  dimension is 1 (``3:1``), the outermost dimension is 3 (``4:30``).

If ``stride_order`` is not specified, the system automatically deduces it from the tensor's layout
using the following logic:

1. Sort the strides in descending order.
2. If multiple dimensions have a stride of 1, a deduction failure error is raised.

For example:

- For a tensor with layout ``(2,2,3,4):(2,1,4,12)``, the deduced ``stride_order`` is ``[3,2,0,1]``.
- For a tensor with layout ``(1,5,1):(1,1,1)``, ``stride_order``'s deduction fails because
  all dimensions have an identical stride of 1, making it impossible to determine the correct ordering.

If ``stride_order`` is specified, the system validates that the order is consistent with the
tensor's layout.

The ``divisibility`` parameter specifies the divisibility of the dynamic shape. It could be used to
represent the assumption alignment of the input. Defaults to 1.

Note that this API is only available for compact tensors. For non-compact tensors, we can use
``cute.assume`` to attach divisibility information to a specific shape mode in a host JIT function,
as demonstrated in the following example:

.. code-block:: python

    @cute.jit
    def foo(a: cute.Tensor):
        new_shape = a.shape
        # use cute.assume to set shape of mode=0 with divisibility=16
        new_shape[0] = cute.assume(new_shape[0], 16)
        new_layout = cute.make_layout(new_shape, stride=a.stride)
        new_a = cute.make_tensor(a.iterator, new_layout)


Code Example
~~~~~~~~~~~~

The following example demonstrates how to use ``mark_compact_shape_dynamic`` to specify dynamic tensor layouts.

* ``t0`` & ``t1`` show the usage of ``mark_compact_shape_dynamic`` with unspecified ``stride_order`` and different ``mode`` and ``divisibility``.
* ``t2`` shows the usage of consecutive ``mark_compact_shape_dynamic`` with unspecified ``stride_order`` and different ``mode`` and ``divisibility``.
* ``t3`` & ``t4`` show the usage of ``mark_compact_shape_dynamic`` with different specified ``stride_order``.
* ``t5``, ``t6``, ``t7``, ``t8``, ``t9``, ``t10``, ``t11``, and ``t12`` demonstrate incorrect settings for parameters and expected errors.

.. code-block:: python

    import torch
    from cutlass.cute.runtime import from_dlpack

    # (8,4,16,2):(2,16,64,1)
    a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
    # (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
    # resulting in (1,4,1,32,1):(1,1,1,4,1)
    # b.dim_order() is (3,2,4,0,1)
    b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)

    # auto deduce the stride order to be [2,1,0,3]
    t0 = from_dlpack(a).mark_compact_shape_dynamic(
        mode=0, divisibility=2
    )
    # (?{div=2},4,16,2):(2,?{div=4},?{div=16},1)
    print(t0)

    t1 = from_dlpack(a).mark_compact_shape_dynamic(
        mode=1, divisibility=2
    )
    # (8,?{div=2},16,2):(2,16,?{div=32},1)
    print(t1)

    t2 = from_dlpack(a).mark_compact_shape_dynamic(
        mode=1, divisibility=2
    ).mark_compact_shape_dynamic(
        mode=3, divisibility=2
    )
    # (8,?{div=2},16,?{div=2}):(?{div=2},?{div=16},?{div=32},1)
    print(t2)

    t3 = from_dlpack(b).mark_compact_shape_dynamic(
        mode=2, divisibility=1, stride_order=(3, 0, 2, 4, 1)
    )
    # (1,4,?,32,1):(0,1,4,?{div=4},0)
    print(t3)

    t4 = from_dlpack(b).mark_compact_shape_dynamic(
        mode=2, divisibility=1, stride_order=(2, 3, 4, 0, 1)
    )
    # (1,4,?,32,1):(0,1,128,4,0)
    print(t4)

    t5 = t2.mark_compact_shape_dynamic(
        mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
    )
    # The stride_order is not consistent with the last stride_order

    t6 = from_dlpack(a).mark_compact_shape_dynamic(
        mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
    )
    # The stride_order is not consistent with the deduced stride_order

    t7 = from_dlpack(b).mark_compact_shape_dynamic(
        mode=0, divisibility=4
    )
    # The layout could not be deduced, please specify the stride_order explicitly

    t8 = from_dlpack(b).mark_compact_shape_dynamic(
        mode=30, divisibility=5, stride_order=(3, 0, 2, 4, 1)
    )
    # Expected mode value to be in range [0, 5), but got 30

    t9 = from_dlpack(b).mark_compact_shape_dynamic(
        mode=3, divisibility=5, stride_order=(2, 1, 2, 3, 4)
    )
    # Expected stride_order to contain all the dimensions of the tensor, but it doesn't contain 0.

    t10 = from_dlpack(b).mark_compact_shape_dynamic(
        mode=3, divisibility=5, stride_order=(0, 1, 2, 3, 4, 5)
    )
    # Expected stride_order to have 5 elements, but got 6.

    t11 = from_dlpack(b).mark_compact_shape_dynamic(
        mode=0, divisibility=4, stride_order=b.dim_order()
    )
    # The shape(1) of mode(0) is not divisible by the divisibility(4)

    t12 = from_dlpack(b).mark_compact_shape_dynamic(
        mode=0, divisibility=1, stride_order=(2, 1, 3, 0, 4)
    )
    # The stride_order is not consistent with the layout


Bypass the DLPack Protocol
--------------------------

In certain scenarios, users may wish to bypass the DLPack protocol and invoke the JIT function directly.  
This can be accomplished by creating a lightweight JIT wrapper around the existing JIT function, 
utilizing ``cute.ptr`` and ``cute.make_tensor`` to pass pointers and construct tensors directly.

Typical use cases for bypassing DLPack include:
1. Users want to call the JIT function directly to avoid the overhead introduced by the DLPack protocol.
2. DLPack canonicalizes the stride of shape-1 dimensions to 1, which may result in incorrect alignment 
propagation and affect memory access or performance.
3. DLPack may lack support for some narrow data types.

The following example illustrates how to bypass the DLPack protocol when invoking a JIT function.
Assume we have a pre-defined ``TensorOpGemm`` kernel whose JIT interface expects three 
arguments of type ``cute.Tensor``. To enable direct invocation without DLPack, we first define a JIT wrapper 
function that accepts ``cute.Pointer`` types as parameters. Within this wrapper, we use ``cute.make_tensor`` 
to construct tensors from the provided pointers, and then call the ``TensorOpGemm`` kernel as usual.

.. code-block:: python

    @cute.jit
    def tensor_op_gemm_wrapper(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        l: cutlass.Int32,
    ):

        # Assume alignment of shape to call tensorop_gemm example
        m = cute.assume(m, divby=8)
        n = cute.assume(n, divby=8)

        # Torch is row major
        a_layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
        b_layout = cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
        c_layout = cute.make_ordered_layout((m, n, l), order=(1, 0, 2))
        mA = cute.make_tensor(a_ptr, layout=a_layout)
        mB = cute.make_tensor(b_ptr, layout=b_layout)
        mC = cute.make_tensor(c_ptr, layout=c_layout)
        
        # TensorOpGemm is a pre-defined kernel from our example
        tensor_op_gemm = TensorOpGemm(
            a_ptr.value_type, c_ptr.value_type, cutlass.Float32, (2, 2, 1)
        )

        tensor_op_gemm(mA, mB, mC)

To pass a PyTorch tensor to this new JIT wrapper, we retrieve the raw pointer from the PyTorch tensor 
and create a ``cute.Pointer`` instance using ``cute.make_ptr``.
This approach allows us to bypass the DLPack protocol entirely, avoiding its overhead and potential 
issues with shape-1 dimension handling.

.. code-block:: python

    a = torch.randn(
        m, k, l, dtype=torch.float16, device="cuda"
    ).permute(2, 1, 0)
    b = torch.randn(
        n, k, l, dtype=torch.float16, device="cuda"
    ).permute(2, 1, 0)
    c = torch.randn(
        n, m, l, dtype=torch.float16, device="cuda"
    ).permute(1, 2, 0)
    
    # from cutlass.cute.runtime import make_ptr
    a_ptr = make_ptr(
        cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    b_ptr = make_ptr(
        cutlass.Float16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    c_ptr = make_ptr(
        cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    tensor_op_gemm_wrapper(a_ptr, b_ptr, c_ptr, m, n, k, l)
