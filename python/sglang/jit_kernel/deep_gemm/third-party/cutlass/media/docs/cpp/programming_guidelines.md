![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Programming Guidelines")

# Programming Guidelines

## Hierarchical Organization

The [CUTLASS 3.0 GEMM API](./gemm_api_3x.md) document
explains CUTLASS 3.0's hierarchical organization,
based conceptually on parallelization strategy.
This differs from CUTLASS 2.x's approach,
which more closely mirrors the GPU hardware hierarchy
of thread blocks, warps, and threads.

## Design Patterns

CUTLASS aims for the highest performance possible on NVIDIA GPUs.
It also offers flexible components that can be assembled and customized
to solve new problems related to deep learning and linear algebra.
Given a tradeoff between simplicity and performance,
CUTLASS chooses performance.
Consequently, several design patterns are necessary
to yield a composable structure
while also satisfying these performance objectives.

### Templates

CUDA C++ templates and modern generic programming techniques enable CUTLASS device code to span a large design space.

This design space includes:
* Mixed precision arithmetic and data storage
* Kernels specialized for layout and problem size
* Support for kernel fusion

Moreover, templates provided a structured approach to collecting compile-time constants such as tile dimensions. These
must be template arguments to target static array allocation and take advantage of loop unrolling, constant folding,
and function inlining.

### Constant Memory

Several CUTLASS template classes exhibit a pattern in which problem-specific internal state is known at kernel
launch time and remains invariant throughout the execution of a kernel. For example, tile iterators compute several
offsets based on the strides of the input tensor that is added to an internal pointer when loading the elements
of a tile. These are computed from the tensor stride and never updated; the per-thread internal state consists
only of the internal global memory pointer.

CUTLASS can take advantage of this CUDA grid-invariant property by constructing the object in host code and passing
a composed parameters structure to the kernel. This confers two benefits: (1.) invariant state is held in constant
memory, and (2.) there is no overhead to compute the initial state by each thread.

The design pattern in CUTLASS is for classes with nontrivial constructors to define `struct Params` as an inner class
which contains grid-invariant state. These should define a constructor and an `initialize()` method. The `Params`
structure should also include a data member corresponding to each data member in the parent class, so these too can
be properly constructed in host code. The parent class should define a constructor which accepts `Params const &` as
its first argument.

### Composable Shared Memory

Shared memory requires explicit effort by the programmer to allocate and de-allocate. CUTLASS follows the paradigm
introduced by [CUB](https://nvlabs.github.io/cub/) to define composed structures for storing data intended to be held
in shared memory. Any object requiring shared memory storage for itself or its data members should define a child
structure called `SharedStorage`. This holds data needed by the class and also instantiates `SharedStorage`
objects for each data member.

To be consistent, this pattern defines a convention in which classes define internal shared memory storage requirements.
Classes should consider all SharedStorage structures to be opaque other than their own child class. When the lifetimes
of child objects are known to be non-overlapping, `union`s may be used to alias multiple SharedStorage objects to the same
shared memory region and reduce overall shared memory capacity.  Developers should carefully note that C++ `union` rules
require that they only access the most recently written ("active") member of the `union`; this differs from C rules.

For host to device ABI compatibility, inheritance from a class is only permitted if the superclass is unique to the
child class. This is most easily achieved by templating the parent class by the child class (CRTP).

### Loop Unrolling

CUTLASS requires tiles of data to be stored in registers for high-bandwidth access. Simultaneously, high-throughput math instructions
must be issued concurrently with memory instructions to hide latency with relatively few concurrent threads. These objectives are
achieved by unrolling loops whose iteration counts are known at compile time.

Consequently, most loops within the CUTLASS GEMM implementation are specified by constant values and template arguments. The CUDA compiler
is able to unroll the loop bodies, map array elements to registers, and construct an efficient instruction schedule.

All loops expected to be unrolled should be annotated with `CUTLASS_PRAGMA_UNROLL` to explicitly direct the compiler
to unroll them.

```c++
int const kN = 8;
Array<float, kN> x;                       // Array we would like to store in registers

CUTLASS_PRAGMA_UNROLL                     // Directs the CUDA compiler to unroll this loop.
for (int idx = 0; idx < kN; ++idx) {      // Loop has constant number of iterations.

  x[i] = float(idx);                      // Indirect access by induction variable results in
                                          // direct register access.
}
```
## Style

### If you see an issue in code formatting, fix it

You are empowered to reformat code.
Please, however, consider making reformatting changes separately from content-related changes.

### No automatic code formatting

Do not use any kind of automatic code formatting,
like `clang-format`, on CUTLASS code.

### C++ style

#### CUTLASS is a C++ project

CUTLASS is a C++ project.  CUDA C++ is a C++ dialect.
Therefore, we write using standard C++ idioms as much as possible.
We aim for portability to as many compilers as possible,
by writing host code in Standard C++
and device code in CUDA C++
that resembles Standard C++ as much as possible.
This improves usability
for the general community of C++ developers,
and makes it easier for new staff to join the project.

#### Follow Standard C++ idioms where possible

Regarding "standard C++ idioms,"
CUTLASS source code follows the following guidelines,
with deviations only because of compiler limitations
or where performance absolutely requires it.
"Performance requires it" implies measurement.
Deviations should be limited in scope
and we should always strive to eliminate them.

* [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)

* [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

#### C is not a subset of C++

C is not a subset of C++.
Some valid C is not valid C++, and some valid "C-looking" C++ is not valid C.
See e.g., the informative C++ Standard Committee (WG21) document
[P2735R0](https://isocpp.org/files/papers/P2735R0.pdf),
which explains ways in which the same code has different behavior in C vs. C++.
In some cases, code that compiles in both C and C++,
and is correct in C, has undefined behavior (can crash or worse) in C++.
The "type.punning" section of P2735R0 specifically relates to unions.

#### Spacing and line length

* Use spaces, not tabs.

* Use 2 spaces to indent.

* Use at most 100 characters per line.

(Right-align tensor shape layout comments at column 120.
Please see below.)
Lines longer than 100 characters typically wrap unfavorably
when viewed in Github's pretty printer.

#### Formatting function declarations and definitions

Short function headers can go on one line.

Do not insert a newline between the parenthesis
that closes the function's parameters and
the curly bracket that opens the function's body.

```c++
int short_name(int x, int y) {
  return x + y;
}
```

If the function name and its parameters are too long to fit on one line,
break the line immediately after the opening parenthesis
that starts the parameter list.  Then, double-indent the parameters
to distinguish them from the body of the function.

```c++
void indeed_my_fellowbeings_this_function_name_is_unusually_long(
    std::uint32_t foo, // parameters are double-indented
    std::uint32_t const* bar,
    TypeA a,
    TypeB b,
    TypeC c) { // the ) and { go on the same line still
  auto d = body_of_the_function(a, b, c); // body is single-indented
  // ... more code ...
}
```

For a constructor with a long parameter list,
break the line after the parentheses, just as with other functions.
Align the colon that starts the constructor's initializer list
flush with the comma on the next line.

As with functions, double-indent the parameters
to distinguish them from the constructor body.
Here is an example.

```c++
class YesTheCommunityAgreesThatTheNameOfThisClassIsIndeedExtremelyLong {
public:
  CUTLASS_HOST_DEVICE
  YesTheCommunityAgreesThatTheNameOfThisClassIsIndeedExtremelyLong(
      int this_is_the_first_parameter_and_its_name_is_long,
      int this_is_the_second_parameter_and_its_name_is_also_long,
      int this_is_the_third_parameter_and_its_name_is_long_too)
  : x_(this_is_the_first_parameter_and_its_name_is_long)
  , y_(this_is_the_second_parameter_and_its_name_is_also_long)
  , z_(this_is_the_third_parameter_and_its_name_is_long_too) {
    // constructor body
    // more of the constructor body
  }

private:
  int x_ = 0;
  int y_ = 0;
  int z_ = 0;
};
```

#### Formatting function calls

When calling a function or function object with a long name,
break the line right after the invoking open parenthesis.
Here are some examples.

```c++
detail::very_long_function_object_name<TemplateArgument>{}(
  params.long_parameter_name, some_operator.another_long_function_name());

detail::an_even_longer_function_object_name<TemplateArgument1, TemplateArgument2>{}(
  params.long_parameter_name, some_operator.long_member_function_name(),
  another_operator.another_long_member_function_name(x, y, z));
```

#### If-else brackets and spacing

* Always use braces with conditionals such as `if`,
  even if the body is a single line.

* Use a space after control flow keywords
  such as `if`, `for`, and `while`.

* Use a space after the parenthesis closing a conditional
  such as `if`, and the curly bracket opening a scope.

* Use a new line between the closing brace
  of an `if` branch, and the `else` keyword.

```c++
if (condition) { // space after if, and between ) and {
  // ... code ...
} // newline after }
else {
  // ... other code ...
}

// space after keyword for
for (int k = 0; k < num_iters; ++k) {
  // ... still more code ...
}
```

#### East const

CUTLASS uses the
["East const"](http://slashslash.info/2018/02/a-foolish-consistency/)
convention.
That is, the `const` or `constexpr` keyword
goes after the type, not before.
The general rule is that `const` or `constexpr`
modifies the type to the left of it.
Here are some examples.

```c++
float constexpr compile_time_constant = 42.3f;

float const const_float = /* whatever */;
float const& reference_to_const_float = const_float;
float const* pointer_to_const_float = &const_float;
float const* const const_pointer_to_const_float = &const_float;

float nonconst_float;
float& reference_to_nonconst_float = nonconst_float;
float* pointer_to_nonconst_float = &nonconst_float;
float* const pointer_to_nonconst_float = &nonconst_float;
```

Contrast this with "West const" style, e.g.,

```c++
const float const_float = /* whatever */;
const float* pointer_to_const_float = &const_float;
```

#### Alignment of reference and pointer types

For reference and pointer types,
align the `&` resp. `*` flush against the type
that it modifies.  This is called "left alignment."

For example, do this:

```c++
int const& var;
int const* var;
```

and not this.

```c++
int const &var;
int const *var;
```
#### Avoid calling functions "fast" or "optimized"

Putting words like "fast" or "optimized"
in the name of a function
assumes that the "fast" path is actually faster.
That might be true now, but later changes
(in the code, compilers, or GPU hardware)
might make it false.  In that case,
your name could be unintentionally misleading.
Consider instead a name that briefly describes
the algorithm or feature that is relevant for optimization.
For example, `compute_on_host` is more meaningful
than `compute_slowly`, and computing on host
might be faster in some cases
(e.g., if the data are already on host
and the algorithm is not GPU-friendly).

CUTLASS code has not always followed this rule in the past.
Some functions and classes might have words like "fast" in their name.
New code should follow this rule, however.

#### Avoid creating unconstrained templated functions with common names

See [C++ Core Guidelines T.47](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#t47-avoid-highly-visible-unconstrained-templates-with-common-names):
"Avoid highly visible unconstrained templates
with common names."
Argument-dependent lookup (ADL) means that
if users call a function name without specifying the namespace,
the compiler can find overloads
of that function in any namespace.
This can lead to ambiguous overloads in users' code,
just because they happened to include one of your header files
that exposes an unconstrained function template.
The following illustrates this
with an unconstrained swap overload in the `cutlass` namespace.

```c++
#include <cassert>
#include <memory>
#include <utility>

// Uncomment the line below to observe unwarranted build errors.
//#define BAD_CUTLASS_SWAP 1

namespace cutlass {
struct Bar {
  float f;
};
} // namespace cutlass

#ifdef BAD_CUTLASS_SWAP
namespace cutlass {

// don't do this
template<class T>
void swap(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}

} // namespace cutlass
#endif // BAD_CUTLASS_SWAP

namespace other {

#ifdef BAD_CUTLASS_SWAP
using cutlass::swap;
#endif // BAD_CUTLASS_SWAP

// Imagine for the sake of this example
// that "foo" is a less common name,
// and that T is constrained via
// std::enable_if or a requires clause.
template<class T>
void foo(T& a, T& b) {
  // The usual idiom for using std::swap is the "swap two-step":
  //
  // 1. import std::swap into the current scope, then
  // 2. call swap without namespace qualification.
  //
  // That won't build if we have another swap
  // overload available in the scope already.

  using std::swap;
  swap(a, b); // OBSERVE UNWARRANTED BUILD ERROR HERE
}

} // namespace other

int main() {
  int x = 42;
  int y = 43;
  other::foo(x, y);
  assert(x == 43);
  assert(y == 42);

  cutlass::Bar a{42.0};
  cutlass::Bar b{43.0};
  other::foo(a, b);
  assert(a.f == 43.0);
  assert(b.f == 42.0);

  // GCC 7.5 std::unique_ptr::reset calls swap,
  // leading to the same issue as above.
  // GCC 12.2's implementation of std::unique_ptr
  // does not have this issue.  Nevertheless,
  // breaking the swap two-step will break users' code,
  // just by them happening to include your headers.
  auto ptr = std::make_unique<cutlass::Bar>(cutlass::Bar{666.0f});
  ptr.reset(new cutlass::Bar{777.0f}); // OBSERVE UNWARRANTED BUILD ERROR HERE

  return 0;
}
```

#### Function return values and in-out parameters

##### Prefer return values to output parameters

In general, avoid in-out mutable references to return a value.
If you need to return multiple values,
you can return them by `struct` or `tuple`,
rather than by output references.
This includes the special case of error reporting
by returning either a value or an error code.
Please see the next section for details.

```c++
// Instead of passing in-out mutable references ...
void not_preferred(float& input_and_output); // not preferred

// keep functions pure and return value types instead
float preferred(float input); // preferred
```

##### Return multiple values by struct or tuple

Sometimes a function needs to return multiple values.  In that case, consider the following, in decreasing order of preference.

1. Return a `struct`.  This lets you name the fields
   (for more self-documenting code),
   yet still permits use of structured binding.

2. Return a `tuple`.  If you need a tuple type
   that works on device, use `cute::tuple`.
   (Please note that `cute::tuple` does not work
   for all the types that work in `std::tuple`.
   CuTe's documentation explains.)

3. Resort to "returning" multiple values by output references
   only if performance requires it.

Here is an example of the struct approach for named values.
For a comparable example in the C++ Standard,
please see [`std::allocate_at_least`](https://en.cppreference.com/w/cpp/memory/allocate_at_least),
which returns `std::allocation_result`.

```c++
struct my_computation_result {
  float value = 0.0f;
  float relative_error = 0.0f;
  bool success = false;
};

my_computation_result my_computation(float tolerance);

void foo(float tolerance) {
  // Approach 1: Use structured binding.  The names
  // you choose on the left-hand side have nothing
  // to do with the struct, so it's up to you
  // to get the order right.  On the other hand,
  // this code works whether my_computation returns
  // a struct or a tuple.
  auto [val, rel_err, ok] = my_computation(tolerance);

  // Approach 2: Keep the struct and use its named fields.
  // This approach prevents errors like mixing the order of return types.
  // However, it only works for structs, not for tuples.

  auto result = my_computation(tolerance);
  if (not result.success) {
    // computation did not succeed
  }
  else if (result.relative_error > tolerance) {
    // successful but relative error too large
  }
  else {
    // successful and relative error is in bounds
  }
}
```

##### Reporting errors from a function that returns one or more values

We may want to return one or more values
from a function that could fail
or otherwise report errors.
That is, the function either

* returns one or more valid values, or

* does not return any values and reports an error,

but NOT BOTH.  We contrast this with cases
when it's meaningful to report both a result
and whether the result is satisfactory.
For example, when solving
a system of nonlinear equations iteratively,
users may want the approximate computed solution,
even if the iteration did not succeed
by converging to the desired tolerance
in the desired number of steps.
(Users may want to invest more steps,
or use the current approximation
to jump-start a different algorithm.)

We're talking here about the "either valid value(s),
or error, but not both" case.
For this case, C++ offers a few options.

1. Return the value(s), or throw an exception on error

2. `std::expected` (requiring C++23) or something like it

3. `std::optional` (for a Boolean error state)
   or something like it

4. `std::variant` (a C++17 fall-back for `std::expected`)
   or something like it

5. C-style interface: return an error code,
   and "return" the values as output parameters

We usually cannot or do not want to
throw exceptions on device.
Some code projects forbid exceptions entirely
(on host or device)
and tell the compiler to disable them.
If we exclude a C-style interface (the last option)
as not idiomatic C++, then for host-only code,
`std::expected`, `std::optional`, and `std::variant`
all work.
For code that needs to build and run on device,
we can fall back to libcu++ equivalents
in the `cuda::std::` namespace, when they exist.
Otherwise, we must resort to returning a struct or tuple
with the value and the error information,
and ask users not to use the value on error.
This is acceptable if the value can be constructed
cheaply with a reasonable default.

##### Performance of different value-or-error reporting methods

[P1886R0](https://wg21.link/P1886R0)
(Ben Craig, "Error speed benchmarking")
surveys different ways in Standard C++
to report errors from a function
that returns one or more values,
and compares their (host-only) performance
with different compilers.

##### Use aggregate initialization when returning a struct or tuple

Use aggregate initialization when returning a struct or tuple.
This avoids duplication of the return type name.

```c++
struct foo_result {
  float value = 0.0f;
  float error = 0.0f;
  bool success = false;
};

foo_result foo(std::span<const float> input) {
  // ... code  ...

  // Prefer this.  We know what type the function returns.
  return {val, err, ok}; // prefer this

  // Naming foo_result again here is unnecessary.
  // return foo_result{val, err, ok};
}
```

However, note that this won't work if the function returns `auto`.
The general rule is to avoid code duplication.

```c++
auto foo(std::span<const float> input) {
  // ... code  ...

  if constexpr (some_condition) {
    return foo_result{val, err, ok};
  }
  else {
    return bar_result{val, err, ok};
  }
}
```

##### Prefer using the actual return type to auto, if you know the type

C++ lets you use `auto` to deduce the type returned from a function.

* If you know the actual type, prefer using the type instead of `auto`.

* Use [Constructor Type Argument Deduction](https://en.cppreference.com/w/cpp/language/class_template_argument_deduction)
  (CTAD) if you know that a function returns some type
  (e.g., `Tensor`), but don't know the type's template arguments.

* Use `auto` in structured bindings (where you have to use it anyway).  This also makes your code agnostic of whether the return type is a `struct`, `tuple`, `pair`, or other tuple-like type.

* Be careful using `auto` with types that provide expression templates.

Contrast this with "Almost Always Auto" (AAA) style.
We deliberately choose not to follow AAA style,
for the following reasons.

* Using the actual type when we know it can help prevent common loss-of-precision errors in mixed-precision computations, an important use case for CUTLASS.

* CTAD gives us much of the brevity of AAA, with more clarity.

* Using the actual type instead of `auto` can prevent common dangling errors with expression templates.

#### Classes and structs

Type names use `CamelCase`.
That is, words start with capital letters.
The remaining letters in the word are lower case,
and words are joined with no intervening underscores.
The only exception is when implementations are
a drop-in replacement for C++ Standard Library components.

Follow the
[C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-struct)
to decide whether to use `class` or `struct`.

* Use `class` when the object must maintain an invariant.
  Data members related to the invariant should be `private`.

* Use `struct` when the class has no invariant to maintain,
  and data members may vary arbitrarily with respect to each other.

Prefer nonmember functions and statelessness where possible.
Member functions imply invariants.
More invariants make code maintenance and testing harder.

#### Class members

Methods and members are written using `snake_case`.

Private data and function members have suffix `_`.

#### Class Member Order

Members within classes and structures should be organized as follows:

1. Type and constant definitions

2. Data members

3. Constructors

4. Other methods

This convention follows the
[CUB library](https://nvlabs.github.io/cub/)
and is also described by
[Howard Hinnant](https://howardhinnant.github.io/classdecl.html).
It also approximates the usual ordering of chapters
in a typical Systems and Controls textbook.
That is, it

1. identifies relevant constants,

2. defines a state-space representation
   of the dynamical system under study
   (the class's data members), and then

3. devotes the remaining "chapters" to defining
   the system's dynamical behavior
   (the class's methods).

Here is an example class.

```c++
class A {
public:
  // type definitions
protected:
  // protected type definitions
private:
  // private type definitions

public:
  // data members
protected:
  // protected data members
  // STRONGLY TO BE AVOIDED;
  // please see C++ Core Guidelines
private:
  // private data members

public:
  // methods
protected:
  // protected methods
private:
  // private methods
};
```

#### For code reuse, prefer composition over inheritance

* [C++ Core Guidelines C.129](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c129-when-designing-a-class-hierarchy-distinguish-between-implementation-inheritance-and-interface-inheritance): "When designing a class hierarchy, distinguish between implementation inheritance and interface inheritance"
* [C++ Core Guidelines ES.63](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-slice): "Don't slice"

Suppose that a class hierarchy exists entirely for implementation convenience, so that implementers can reuse code and "program by difference" (changing or adding only what's different from the base class).  In the example below, both `PipelineA` and `PipelineB` are used by themselves.  `PipelineB` inherits from `PipelineA` just to avoid duplicating code.  There are no virtual member functions, and users don't expect to rely on run-time polymorphism.

```c++
class PipelineA {
public:
  PipelineA(Arg0 arg0, Arg1 arg1)
    : arg0_(arg0), arg1_(arg1)
  {}

  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    // ... implementation ... 
  }

  void consumer_release(uint32_t stage, uint32_t skip) {
    // ... implementation ...
  }

private:
  Arg0 arg0_;
  Arg1 arg1_;
};

class PipelineB : public PipelineA {
public:
  PipelineB(Arg0 arg0, Arg1 arg1, Arg2 arg2) :
    PipelineA(arg0, arg1), arg2_(arg2)
  {}

  // Reuse PipelineA::producer_acquire via inheritance

  // Override PipelineA::consumer_release
  void consumer_release(uint32_t stage, uint32_t skip) {
    // ... some other implementation, not invoking parent ...
  }

private:
  Arg2 arg2_;
};
```

The problem with public inheritance here is that `PipelineB` is NOT a (versus "is-a," i.e., substitutable-as) `PipelineA`. In particular, the following code would be incorrect.

```c++
void consume_and_release_pipeline(PipelineA* parent) {
  // ... code ...
  parent->consumer_release(stage, skip);
  // ... code ...
}

void use_pipeline( /* other args */ ) {
  // ... code ...
  PipelineB child{arg0, arg1, arg2};
  // ... code ...

  // WRONG!!! SLICES CHILD TO PARENT!!!
  consume_and_release_pipeline(&child); // BAD

  // ... code ...
}
```

`PipelineA::consumer_release` is not a virtual member function, so `consume_and_release_pipeline` would not actually be polymorphic, as callers might have expected from an interface that takes a base class pointer. What's worse is that the resulting slicing could violate `PipelineB`'s invariants, thus putting it in an incorrect state.

The most straightforward way to reuse code would be by changing from inheritance (is-a) to composition (has-a).

```c++
namespace detail {

// Implementation class; not for users
class PipelineImpl {
public:
  PipelineImpl(Arg0 arg0, Arg1 arg1)
    : arg0_(arg0), arg1_(arg1)
  {}

  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    // ... implementation ...
  }

  void consumer_release(uint32_t stage, uint32_t skip) {
    // ... implementation ...
  }

private:
  Arg0 arg0_;
  Arg1 arg1_;
};

} // namespace detail

class PipelineA {
public:
  PipelineA(Arg0 arg0, Arg1 arg1) :
    impl_(arg0, arg1)
  {}

  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    impl_.producer_acquire(stage, phase, skip_wait);
  }

  void consumer_release(uint32_t stage, uint32_t skip) {
    impl_.consumer_release(stage, skip);
  }

private:
  detail::PipelineImpl impl_;
};

// A second kind of pipeline.
// Note that this does NOT inherit from PipelineB!
// The two pipeline classes have the same compile-time interface
// (for compile-time polymorphism), but do not belong in an 
// inheritance hierarchy (as would imply run-time polymorphism).
class PipelineB {
public:
  PipelineB(Arg0 arg0, Arg1 arg1, Arg2 arg2) :
    impl_(arg0, arg1), otherTwo_(arg2)
  {}

  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    impl_.producer_acquire(stage, phase, skip_wait);
  }

  void consumer_release(uint32_t stage, uint32_t skip) {
    // this class doesn't actually use impl_ here
    otherTwo_.other_action(stage, skip);
    // ... some other code not using impl_ ...
  }

private:
  detail::PipelineImpl impl_;
  OtherTwo otherTwo_;
  // ... other member data ...
};
```

This design prevents users at compile time from incorrectly assuming that `PipelineB` is a `PipelineA`.  Implementers continue to get compile-time polymorphism, as long as `PipelineA` and `PipelineB` implement the same compile-time interface.

##### Behavioral subtyping

Another reason to avoid public inheritance would be if the public member functions of `PipelineA` and `PipelineB` have different behavior, such that the invariants satisfied by the member functions of the base class `PipelineA` are not satisfied by the correspondingly named member functions of the subclass `PipelineB`.  For example, suppose that both classes have a public `producer_arrive` member function.  However, for `PipelineA`, this issues a producer arrival only for its own block, whereas for `PipelineB`, this issues a producer arrival for all blocks in the cluster.  Again, PipelineB "is-not-a" PipelineA.  The child class doesn't just add behavior onto the parent class; it has completely different behavior. Thus, it fails to satisfy behavioral subtyping: invariants of the parent class's member functions are not satisfied by the child class.  Behavioral subtyping is especially important when reasoning about already difficult things like parallel synchronization.  The inheritance design would give developers the false impression that `PipelineB` just adds behavior atop `PipelineA`, whereas in fact, developers would need to understand both pipeline classes completely to build a correct mental model about their behavior.

The fix is the same: Use composition, not inheritance.  As [C++ Core Guidelines C.120](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c120-use-class-hierarchies-to-represent-concepts-with-inherent-hierarchical-structure-only) explains: "Use class hierarchies to represent concepts with inherent hierarchical structure (only)."

1. "Make sure the idea represented in the base class exactly matches all derived types and there is not a better way to express it than using the tight coupling of inheritance."
2. "Do not use inheritance when simply having a data member will do."

#### Use scoped enums

Use scoped enums (a C++11 feature) for enumerated types.
Use capital letters for the enumerated type name
and prefix `k` for enumerators like other constants.

```c++
enum class MatrixOperation {
  kNone,
  kTranspose,
  kConjugate,
  kHermitian
};
```

#### Namespaces

Namespaces are all lower case.
The top-level namespace is `cutlass::`.
The second nested namespace refers to
the general category of operation
performed by its members: e.g., `gemm::`.
The third nested namespace refers to
the operations' position in the conceptual hierarchy:
e.g., `device::`, `kernel::`, or `collective::`.

The bodies of namespace definitions should not be indented.
Comments on the closing brace to indicate
the namespace being closed are welcome.

```c++
namespace cutlass {
namespace gemm {
namespace kernel {

struct AnotherGemmKernel {
  // ... contents ...
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
```

#### File Names

New files should be named using `snake_case`
with extension `.hpp` for header files,
`.cu` for CUDA sources,
and `.cpp` for C++ host-only source files.

Header files with extension `.h`
are CUTLASS 2.x legacy headers.

#### Macros

Only use macros when the preprocessor
is the only way to accomplish the task.
Do not use macros for literal constants.
Instead, if inside the body of a function,
use `constexpr` values,
and if at namespace scope, use
[`inline constexpr` variables](https://en.cppreference.com/w/cpp/language/inline)
(a C++17 feature).

"Namespace" macros by starting them with the module name, e.g., `CUTLASS_`.
Macros and ONLY MACROS use all capital letters with underscores between words.
For example:

```c++
#define CUTLASS_MACROS_USE_ALL_CAPS inline __host__ __device__
```

Header files such as
[cutlass/cutlass.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/cutlass.h)
and
[cute/config.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/cutlass.h)
offer macros for expressing compiler-dependent behavior.
These include

* replacements for `__device__` and/or `__host__`
  annotations:

  * `CUTLASS_HOST_DEVICE` or `CUTE_HOST_DEVICE`
    for functions that run on the host and the device,

  * `CUTLASS_DEVICE` or `CUTE_DEVICE`
    for functions that run on the device only,

  * `CUTE_HOST`
    for functions that run on the host only, and

  * `CUTE_HOST_RTC`
    for functions that run on the host only,
    but occur as unevaluated operands (of e.g., `decltype` or `sizeof`;
    see C++ Standard, `[expr.context]` 1) in device code; and

* annotations to loop unrolling:

  * `CUTLASS_PRAGMA_UNROLL` or `CUTE_UNROLL`
    for full unrolling of loops with constant trip counts, and

  * `CUTLASS_PRAGMA_NO_UNROLL` or `CUTE_NO_UNROLL` to prevent unrolling.

#### Guard all headers with `#pragma once`

Use `#pragma once` to guard all headers.

### CuTe Layout Comments

* Right-align tensor shape layout comments at column 120. 
* If layout comment is too long do your best to align it.
* If layout comment is too long and there are many related tensors
  that the reader should read together,
  try to align the layout comments of related tensors.

Here are a couple examples.

```c++
Tensor mC = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N), params.dC);                              // (M,N)
Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N), params.dD);                              // (M,N)
Tensor mAux = make_tensor(make_gmem_ptr(params.ptr_Aux), make_shape(M,N), params.dAux);                        // (M,N)

auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
Tensor tCgD = thr_mma.partition_C(gD);                                                             // (VEC,THR_M,THR_N)
Tensor tCgC = thr_mma.partition_C(gC);                                                             // (VEC,THR_M,THR_N)
Tensor tCgAux = thr_mma.partition_C(gAux);                                                         // (VEC,THR_M,THR_N)
```

```c++
Tensor my_tensor = make_tensor<Type>(Layout<Shape<_2,_2>{}, Stride<_1,_2>>{});                           // (2,2):(1,2)
    
// Related tensors
Tensor my_tensor1 = make_tensor<Type>(ThisIsAVeryComplicatedLayoutWithAVeryLongName);         // ((Mode0_0,Mode0_1,Mode0_2),Mode1,Mode2,Mode3)
Tensor my_tensor2_related = make_tensor<Type>(ThisIsAVeryComplicatedLayoutWithAVeryLongName); // ((Mode0_0,Mode0_1,Mode0_2),Mode1,Mode2,Mode3)
```

### Warnings

CUTLASS code aims to build free of warnings.

#### Spurious warnings

Some compilers, or some versions of a compiler, emit spurious warnings, that is, "false positives" for perfectly fine code.  While such code is correct, the warnings can obscure errors.  Users also may report warnings as bugs, and processing those bugs takes developer time away from other tasks.  Thus, it's good to try to "fix" the warnings, if doing so wouldn't make the code worse.

#### Missing return statement

GCC 10 (but not 7.5, 9.4.0, or 11) has trouble deducing that a function with `auto` return type and all of its returns in an `if constexpr` ... `else` statement must actually return.  As a result, GCC emits spurious "missing return statement" build warnings.  Such functions have one of two forms: `if constexpr` ... `else` where `else` returns, and `if constexpr` ... `else` where `else` is meant to fail at compile time.  Here is an example of the first form.

```c++
template<class T>
constexpr auto first_form(T t) {
  if constexpr (some_condition_v<T>) {
    return some_function(t);
  }
  else if constexpr (another_condition_v<T>) {
    return another_function(t);
  }
  else {
    return yet_another_function(t);
  }
}
```

In this form, the `if constexpr` ... `else` sequence of branches covers all possibilities.  Here is an example of the second form.

```c++
template<class T>
constexpr auto second_form(T t) {
  if constexpr (some_condition_v<T>) {
    return some_function(t);
  }
  else if constexpr (another_condition_v<T>) {
    return another_function(t);
  }
  else {
    static_assert(sizeof(T) < 0, "This branch always fails");
  }
}
```

In this form, the `else` branch had a `static_assert` that was meant always to fail if the `else` branch were taken, such as `static_assert(sizeof(T) < 0)`.  (Note that we cannot use `static_assert(false)` here, because it will ALWAYS fail at compile time, even if the `else` branch is not taken.  C++23 fixes this behavior, but CUTLASS currently requires that its code be compatible with C++17.  As a result, CUTLASS includes a `dependent_false<T>` library function that you can use in place of the always-`false` test `sizeof(T) < 0`.)

One can suppress "missing return statement" warnings for both forms by invoking CUTLASS' function-like macro `CUTE_GCC_UNREACHABLE`.  When building with GCC, this invokes the GCC-specific built-in function `__builtin_unreachable()`.  Actually calling this function is undefined behavior, so using this lets the programmer declare that the code path calling that function will never be taken.  (C++23 introduces the `std::unreachable()` function, which achieves the same goal.  Again, though, CUTLASS cannot currently use C++23 library functions.)  Here is an example of how to use `CUTE_GCC_UNREACHABLE`.

```c++
template<class T>
constexpr auto second_form(T t) {
  if constexpr (some_condition_v<T>) {
    return some_function(t);
  }
  else if constexpr (another_condition_v<T>) {
    return another_function(t);
  }
  else {
    static_assert(sizeof(T) < 0, "This branch always fails");
  }
  CUTE_GCC_UNREACHABLE;
}
```

This macro should only be used if it is needed to suppress spurious warnings.  Also, this function should not be used if the developer is not sure whether the code exhaustively tests all possibilities.  For example, some functions may look like this.

```c++
template<class T>
constexpr auto possibly_nonexhaustive(T t) {
  if constexpr (some_condition_v<T>) {
    return some_function(t);
  }
  else if constexpr (another_condition_v<T>) {
    return another_function(t);
  }
 
  // NOTE lack of unadorned "else" here
}
```

This is a good opportunity to review the function.  If the branches are obviously meant to be exhaustive, you can add an `else` branch with a `static_assert` (see above for how to express this).  If you're not sure, leave it alone and let the compiler issue warnings.

#### Unused variable

Some compilers may emit spurious unused warnings for some variable declarations, where the variable was only being used inside a `decltype` in an `if constexpr` test. Marking the variables as `[[maybe_unused]]` (a standard C++17 attribute) suppresses these warnings.  Again, please only do this if you're sure that the code is right.

### CUDA C++ style

#### CUDA Built-in Variables

Avoid direct access to CUDA built-in variables `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` within
CUTLASS components except in special circumstances.

Using built-in global variables directly within resuable components necessitates that all components
use them consistently which may not be possible if CUTLASS components are used in other contexts.

Instead, components should accept a linear ID identifying threads, warps, and threadblocks from calling
code. The top-level kernel may then decide how to map threads, warps, and blocks to the problem it is
solving.

#### Use CUTLASS's and CuTe's fundamental types and operations

Use the
[fundamental types and operations](fundamental_types.md)
defined in CUTLASS consistently.
This contributes to a framework of interoperable, consistent components.
It reduces code duplication, which reduces build and test times.
It also saves developer effort.

CUTLASS's fundamental types and operations include

* [Numeric types](fundamental_types.md#numeric-types) to represent numeric data in host and device code, and

* [functional.h](fundamental_types.md#functional) to perform numeric operations in generic code.

CUTLASS 3.0 uses CuTe components to represent data layouts and multidimensional arrays.
Please refer to the [CuTe Tutorial](./cute/00_quickstart.md) for details.
CuTe has replaced CUTLASS 2.x components such as
[Containers](fundamental_types.md#containers),
[Layouts](layout.md), and
[`TensorRef` and `TensorView`](layout.md#tensorref).

## CUTLASS idioms

### Detecting major mode

Developers sometimes need to detect whether a tensor is MN-major or K-major.
(For definitions, see the [CuTe GEMM tutorial](./cute/0x_gemm_tutorial.md).)

* _Correct_: `cutlass::detail::is_major<0, Stride>()` or
`cutlass::detail::is_k_major()` from `include/cutlass/gemm/gemm.h`

* _Incorrect_: `get<0>(stride) == 1`

The second point is incorrect because it assumes that the mode
is a single integer, not a multimode.
This means that the code will fail to compile for tensor contractions.
For example, suppose that a tensor A
has shape `((X, Y), K)` and stride `((1, X), X*Y)`.
`get<0>(stride)` is the tuple `(1, X)`, not a single integer.
However, A is certainly M major if interpreted as a matrix.

### Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
