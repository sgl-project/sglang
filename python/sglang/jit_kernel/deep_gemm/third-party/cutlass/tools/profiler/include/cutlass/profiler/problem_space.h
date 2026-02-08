/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file
   \brief 

    "Any sufficiently complicated C or Fortran program contains an ad-hoc, informally-specified, 
     bug-ridden, slow implementation of half of Common Lisp."

      - Greenspun's Tenth Rule of Programming

 
  cutlass::profiler::ProblemSpace defines a set of data structures which represent the Cartesian
  product of sequences defined by integer ranges, lists of scalars, and sets of enumerated types.

  These permit a single invocation of the CUTLASS Profiler to iterate over a large set of problems,
  verify and profile various operations when they are compatible with the command line, and
  construct data tables of results that are convenient inputs to post processing in Excel or Pandas. 

  By executing multiple problems per invocation, startup overheads may be amortized across many
  kernel launches. 
*/

#pragma once

// Standard Library includes
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdlib>

// CUTLASS Utility includes
#include "cutlass/util/command_line.h"

// CUTLASS Library includes
#include "cutlass/library/library.h"

// Profiler includes
#include "enumerated_types.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the argument schema
struct ArgumentDescription {

  /// Type of argument
  ArgumentTypeID type;

  /// Prioritized array of aliases used in command line parsing
  std::vector<std::string> aliases;

  /// Description of argument
  std::string description;

  //
  // Methods
  //

  /// Default ctor
  ArgumentDescription(): 
    type(ArgumentTypeID::kInvalid) { }

  /// Constructor with aliases
  ArgumentDescription(
    ArgumentTypeID type_,
    std::vector<std::string> const &aliases_,
    std::string const &description_
  ):
    type(type_), aliases(aliases_), description(description_) { }
};

/// Vector of arguments
using ArgumentDescriptionVector = std::vector<ArgumentDescription>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Base class for kernel arguments
struct KernelArgument {

  //
  // Type definitions
  //

  /// Value base class
  struct Value {

    KernelArgument const *argument;
    bool not_null;

    //
    // Methods
    //

    Value(
      KernelArgument const *argument_ = nullptr, 
      bool not_null_ = true
    ): argument(argument_), not_null(not_null_) { }

    virtual ~Value() { }

    virtual std::ostream &print(std::ostream &out) const =0;
  };

  /// Abstract base class to iterate over values within arguments
  struct ValueIterator {

    /// Indicates type of kernel argument
    KernelArgument const *argument;
    
    /// If the iterator points to an argument that is null, it needs to be distinguished
    /// from end.
    bool null_argument;

    //
    // Methods
    //

    /// Constructs a value iterator - no methods are valid if argument_ == nullptr
    ValueIterator(
      KernelArgument const *argument_ = nullptr, 
      bool null_argument_ = false): 
      argument(argument_), null_argument(null_argument_) {

      if (!argument_->not_null()) {
        null_argument = true;
      }
    }

    virtual ~ValueIterator() { }

    /// Advances to next point in range
    virtual void operator++() = 0;

    /// Compares against another value iterator - must be of the same KernelArgument type
    virtual bool operator==(ValueIterator const &it) const = 0;

    /// Returns a unique_ptr<Value> object pointing to a newly created value object
    virtual std::unique_ptr<Value> at() const = 0;

    /// Gets the type of the iterator
    ArgumentTypeID type() const {
      return argument->description->type;
    }

    /// Helper to compute inequality
    bool operator!=(ValueIterator const &it) const {
      return !(*this == it); 
    }

    std::ostream &print(std::ostream &out) const;
  };

  //
  // Data members
  //

  /// Describes the argument
  ArgumentDescription const *description;

  /// Parent node
  KernelArgument *parent;

  /// Sequence in which the kernel argument is to be iterated over. 
  /// Smaller means faster changing. -1 is don't  care
  int ordinal;

  //
  // Methods
  //

  /// Default ctor
  KernelArgument(
    ArgumentDescription const *description_ = nullptr,
    KernelArgument *parent_ = nullptr,
    int ordinal_ = -1
  ): description(description_), parent(parent_), ordinal(ordinal_) { }

  virtual ~KernelArgument();

  /// Returns true if the kernel argument iself is empty
  virtual bool not_null() const =0;

  /// Returns a string name for debugging
  std::string qualified_name() const {
    if (description) {
      if (description->aliases.empty()) {
        return "<description_not_null_no_aliases>";
      }
      return description->aliases.front();
    }
    return "<description_null>";
  }

  virtual std::unique_ptr<ValueIterator> begin() const =0;
  virtual std::unique_ptr<ValueIterator> end() const =0;
};

using KernelArgumentVector = std::vector<std::unique_ptr<KernelArgument>>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a scalar argument type as a string that is lexically cast to the appropriate kernel
/// type.
struct ScalarArgument : public KernelArgument {

  //
  // Type definitions
  //
  
  /// Value type
  struct ScalarValue : public KernelArgument::Value {

    std::string value;

    //
    // Methods
    //

    ScalarValue(
      std::string const &value_ = "",
      ScalarArgument const *argument = nullptr,
      bool not_null_ = true
    );

    virtual std::ostream &print(std::ostream &out) const;
  };

  using ValueCollection = std::vector<decltype(ScalarValue::value)>;

  /// Abstract base class to iterate over values within arguments
  struct ScalarValueIterator : public KernelArgument::ValueIterator {

    //
    // Data members
    //

    ValueCollection::const_iterator value_it;

    //
    // Methods
    //

    explicit ScalarValueIterator(ScalarArgument const *argument = nullptr);

    virtual void operator++();
    virtual bool operator==(ValueIterator const &it) const;

    /// Gets the value pointed to
    virtual std::unique_ptr<KernelArgument::Value> at() const;
  };

  //
  // Data members
  //

  /// Set of possible values
  ValueCollection values;

  //
  // Methods
  //

  /// Default ctor
  explicit ScalarArgument(
    ArgumentDescription const *description
  ): 
    KernelArgument(description) { }

  virtual bool not_null() const {
    return !values.empty();
  }

  virtual std::unique_ptr<KernelArgument::ValueIterator> begin() const;
  virtual std::unique_ptr<KernelArgument::ValueIterator> end() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Closed range supporting additive increment
struct Range {
  
  //
  // Type definitions
  //

  enum class Mode {
    kSequence,
    kRandom,
    kRandomLog2,
    kInvalid
  };

  struct Iterator {

    int64_t value;
    int64_t increment;
    Range const *range;

    //
    // Methods
    //
    
    Iterator(
      int64_t value_ = 0, 
      int64_t increment_ = 1,
      Range const *range_ = nullptr
    ): 
      value(value_), increment(increment_), range(range_) { }

    Iterator & operator++() {
      value += increment;
      return *this;
    }

    Iterator operator++(int) {
      Iterator self(*this);
      ++(*this);
      return self;
    }

    bool operator==(Iterator const &it) const {
      return value == it.value;
    }

    bool operator!=(Iterator const &it) const {
      return !(*this == it);
    }

    static int64_t round(int64_t value, int64_t divisible) {
      int64_t rem = (value % divisible);

      // Round either up or down
      if (rem > divisible / 2) {
        value += (divisible - rem);
      }
      else {
        value -= rem;
      }

      return value;
    }

    int64_t at() const {
      if (!range) {
        return value;
      }

      switch (range->mode) {
        case Mode::kSequence: return value;

        case Mode::kRandom: {
          double rnd = double(range->minimum) + 
            double(std::rand()) / double(RAND_MAX) * (double(range->maximum) - double(range->minimum));

          int64_t value = int64_t(rnd);

          return round(value, range->divisible);      
        }
        break;

        case Mode::kRandomLog2: {
          double lg2_minimum = std::log(double(range->minimum)) / std::log(2.0);
          double lg2_maximum = std::log(double(range->maximum)) / std::log(2.0);
          double rnd = lg2_minimum + double(std::rand()) / double(RAND_MAX) * (lg2_maximum - lg2_minimum);      

          int64_t value = int64_t(std::pow(2.0, rnd));

          return round(value, range->divisible);
        }
        break;
        default: break;
      }
      return value;
    }

    int64_t operator*() const {
      return at();
    }
  };

  //
  // Data members
  //

  int64_t first;        ///< first element in range
  int64_t last;         ///< last element in range
  int64_t increment;    ///< additive increment between values
  
  Mode mode;            ///< mode selection enables alternative values 
  int64_t minimum;      ///< minimum value to return
  int64_t maximum;      ///< maximum value to return
  int64_t divisible;    ///< rounds value down to an integer multiple of this value 

  //
  // Methods
  //

  /// Default constructor - range acts as a scalar
  Range(int64_t first_ = 0): first(first_), last(first_), increment(1), mode(Mode::kSequence), minimum(0), maximum(0), divisible(1) { }

  /// Range acts as a range
  Range(
    int64_t first_, 
    int64_t last_, 
    int64_t increment_ = 1,
    Mode mode_ = Mode::kSequence,
    int64_t minimum_ = 0,
    int64_t maximum_ = 0,
    int64_t divisible_ = 1
  ): first(first_), last(last_), increment(increment_), mode(mode_), minimum(minimum_), maximum(maximum_), divisible(divisible_) {

    // Helpers to avoid constructing invalid ranges
    if (increment > 0) {
      if (last < first) {
        std::swap(last, first);
      }
    }
    else if (increment < 0) {
      if (first < last) {
        std::swap(last, first);
      }
    }
    else if (last != first) {
      last = first;
      increment = 1;
    }
  }

  /// Helper to construct a sequence range
  static Range Sequence(int64_t first_, int64_t last_, int64_t increment_ = 1) {
    return Range(first_, last_, increment_, Mode::kSequence);
  }

  /// Helper to construct a range that is a random distribution 
  static Range Random(int64_t minimum_, int64_t maximum_, int64_t count_, int64_t divisible_ = 1) {
    return Range(1, count_, 1, Mode::kRandom, minimum_, maximum_, divisible_);
  }

  /// Helper to construct a range that is a random distribution over a log scale
  static Range RandomLog2(int64_t minimum_, int64_t maximum_, int64_t count_, int64_t divisible_ = 1) {
    return Range(1, count_, 1, Mode::kRandomLog2, minimum_, maximum_, divisible_);
  }

  /// Returns an iterator to the first element within the range
  Iterator begin() const {
    return Iterator(first, increment, this);
  }

  /// Returns an iterator to the first element *after* the range
  Iterator end() const {
    return Iterator(first + ((last - first)/increment + 1) * increment, increment, this);
  }
};

/// Integer-valued argument - represented as a list of integer-valued ranges
struct IntegerArgument : public KernelArgument {

  //
  // Type definitions
  //

  /// Value type
  struct IntegerValue : public KernelArgument::Value {

    int64_t value;

    //
    // Methods
    //

    IntegerValue(
      int64_t value_ = 0, 
      IntegerArgument const *argument_ = nullptr, 
      bool not_null_ = true
    );

    /// Pretty printer for debugging
    virtual std::ostream &print(std::ostream &out) const;
  };
  
  /// Collection of ranges represent the IntegerArgument's state
  using RangeCollection = std::vector<Range>;

  /// Abstract base class to iterate over values within arguments
  struct IntegerValueIterator : public KernelArgument::ValueIterator {

    //
    // Data members
    //

    RangeCollection::const_iterator range_it;
    Range::Iterator value_it;

    //
    // Methods
    //

    IntegerValueIterator();
    IntegerValueIterator(IntegerArgument const *argument);

    virtual void operator++();
    virtual bool operator==(ValueIterator const &it) const;

    /// Gets the value pointed to
    virtual std::unique_ptr<KernelArgument::Value> at() const;
  };

  //
  // Data members
  //

  /// Set of possible values
  RangeCollection ranges;

  //
  // Methods
  //

  /// Default ctor
  IntegerArgument(
    ArgumentDescription const *description
  ): 
    KernelArgument(description) { }

  virtual bool not_null() const {
    bool _not_null = !ranges.empty();
    return _not_null;
  }

  virtual std::unique_ptr<KernelArgument::ValueIterator> begin() const;
  virtual std::unique_ptr<KernelArgument::ValueIterator> end() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure defining the data type of tensors
struct TensorArgument : public KernelArgument {

  //
  // Type definitions
  //

  struct TensorDescription {

    /// Data type of elements
    library::NumericTypeID element;

    /// Layout definition
    library::LayoutTypeID layout;

    /// Computed extent
    std::vector<int> extent;

    /// Enables directly specifying stride value used to size tensor
    std::vector<int> stride;

    //
    // Methods
    //

    TensorDescription(
      library::NumericTypeID element_ = library::NumericTypeID::kUnknown,
      library::LayoutTypeID layout_ = library::LayoutTypeID::kUnknown,
      std::vector<int> extent_ = std::vector<int>(),
      std::vector<int> stride_ = std::vector<int>()
    ): 
      element(element_), layout(layout_), extent(extent_), stride(stride_) {}
  };

  using ValueCollection = std::vector<TensorDescription>;

  /// Value structure
  struct TensorValue : public KernelArgument::Value {

    TensorDescription desc;

    //
    // Methods
    //

    TensorValue(
      TensorDescription const &desc_ = TensorDescription(),
      TensorArgument const *argument_ = nullptr, 
      bool not_null_ = true
    );
    
    /// Pretty printer for debugging
    virtual std::ostream &print(std::ostream &out) const;
  };

  /// Abstract base class to iterate over values within arguments
  struct TensorValueIterator : public KernelArgument::ValueIterator {

    //
    // Data members
    //

    ValueCollection::const_iterator value_it;

    //
    // Methods
    //

    explicit TensorValueIterator(TensorArgument const *argument_);

    virtual void operator++();
    virtual bool operator==(ValueIterator const &it) const;

    /// Gets the value pointed to
    virtual std::unique_ptr<KernelArgument::Value> at() const;
  };

  /// Set of possible values
  ValueCollection values;

  //
  // Methods
  //

  /// Default ctor
  explicit TensorArgument(
    ArgumentDescription const *description
  ): 
    KernelArgument(description) { }

  virtual bool not_null() const {
    return !values.empty();
  }

  virtual std::unique_ptr<KernelArgument::ValueIterator> begin() const;
  virtual std::unique_ptr<KernelArgument::ValueIterator> end() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Numeric data type
struct EnumeratedTypeArgument : public KernelArgument {

  //
  // Type definitions
  //

  struct EnumeratedTypeValue : public KernelArgument::Value {

    /// Data type of element
    std::string element;

    //
    // Methods
    //

    EnumeratedTypeValue(
      std::string const &element_ = std::string(),
      EnumeratedTypeArgument const *argument_ = nullptr, 
      bool not_null_ = true
    );
    
    /// Pretty printer for debugging
    virtual std::ostream &print(std::ostream &out) const;
  };

  using ValueCollection = std::vector<decltype(EnumeratedTypeValue::element)>;

  /// Abstract base class to iterate over values within arguments
  struct EnumeratedTypeValueIterator : public KernelArgument::ValueIterator {

    //
    // Data members
    //

    ValueCollection::const_iterator value_it;

    //
    // Methods
    //

    explicit EnumeratedTypeValueIterator(EnumeratedTypeArgument const *argument_ = nullptr);

    virtual void operator++();
    virtual bool operator==(ValueIterator const &it) const;

    /// Gets the value pointed to
    virtual std::unique_ptr<KernelArgument::Value> at() const;
  };

  //
  // Data members
  //

  ValueCollection values;

  //
  // Members
  //

  /// Default ctor
  explicit EnumeratedTypeArgument(ArgumentDescription const *description):
    KernelArgument(description) {}

  virtual bool not_null() const {
    return !values.empty();
  }

  virtual std::unique_ptr<KernelArgument::ValueIterator> begin() const;
  virtual std::unique_ptr<KernelArgument::ValueIterator> end() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object storing the space argument values
class ProblemSpace {
public:

  /// Tuple of arguments
  using Problem = std::vector<std::unique_ptr<KernelArgument::Value>>;

  /// Type used to iterator over things
  using IteratorVector = std::vector<std::unique_ptr<KernelArgument::ValueIterator>>;

  /// Iterates over points in the design space
  class Iterator {
  private:

    /// One iterator per argument
    IteratorVector iterators;

  public:

    //
    // Methods
    //

    explicit Iterator();
    Iterator(ProblemSpace const &problem_space);
    Iterator(Iterator &&it);

    // Rule of three
    Iterator(Iterator const &) = delete;
    Iterator &operator=(Iterator const &it) = delete;
    ~Iterator() = default;

    /// Pre-increment - advances to next point in argument range
    void operator++();

    /// Gets the current argument value
    Problem at() const;

    /// Moves iterator to end
    void move_to_end();

    /// Equality operator
    bool operator==(Iterator const &it) const;

    /// Inequality operator
    bool operator!=(Iterator const &it) const {
      return !(*this == it);
    }

    /// Helper to call at() method
    Problem operator*() const {
      return at();
    }

    /// Helper to print iterator state
    std::ostream & print(std::ostream &out) const;

  private:

    /// Helper for recursively constructing iterators
    void construct_(KernelArgument const *argument);
  };

public:

  //
  // Data members
  //

  KernelArgumentVector arguments;

  /// Map of argument names to their position within the argument vector
  std::unordered_map<std::string, size_t> argument_index_map;

public:
  
  //
  // Methods
  //

  /// Default ctor
  ProblemSpace() = default;

  /// Constructs a problem space from a vector of arguments. This vector must outlive
  /// the ProblemSpace object, which stores pointers to objects within the
  /// ArgumentDescriptionVector.
  ProblemSpace(ArgumentDescriptionVector const &schema, CommandLine const &cmdline);

  Iterator begin() const;   // returns an iterator to the first point in the range
  Iterator end() const;     // returns an iterator to the first point after the range

  /// Returns the index of an argument by name
  size_t argument_index(char const *name) const;

  /// Gets all argument names as an ordered vector
  std::vector<std::string> argument_names() const;

  /// Returns the number of dimensions of the problem space
  size_t rank() const { return arguments.size(); }
 
private:

  /// Helper for recursively cloning
  void clone_(
    KernelArgumentVector &kernel_args,
    ArgumentDescription const *arg_desc);

  /// Parses command line argument
  void parse_(
    KernelArgument *arg,
    CommandLine const &cmdline);
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexically casts an argument to an int if it is defined. Returns true if not null.
bool arg_as_int(int &int_value, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_int(int64_t &int_value, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_int(
  int &int_value,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_int(
  int64_t &int_value,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

bool arg_as_bool(bool &bool_value, KernelArgument::Value const *value_ptr);

bool arg_as_bool(bool &bool_value,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_NumericTypeID(library::NumericTypeID &numeric_type, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_NumericTypeID(
  library::NumericTypeID &numeric_type,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_LayoutTypeID(library::LayoutTypeID &layout_type, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_LayoutTypeID(
  library::LayoutTypeID &layout_type,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_OpcodeClassID(library::OpcodeClassID &opcode_class, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_OpcodeClassID(
  library::OpcodeClassID &opcode_class,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_SplitKModeID(library::SplitKMode &split_k_mode, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_SplitKModeID(
  library::SplitKMode &split_k_mode,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_ConvModeID(library::ConvModeID &conv_mode, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_ConvModeID(
  library::ConvModeID &conv_mode,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_IteratorAlgorithmID(library::IteratorAlgorithmID &iterator_algorithm, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_IteratorAlgorithmID(
  library::IteratorAlgorithmID &iterator_algorithm,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_RuntimeDatatype(library::RuntimeDatatype &runtime_datatype, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_RuntimeDatatype(
  library::RuntimeDatatype &runtime_datatype,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_RasterOrder(library::RasterOrder &raster_order, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_RasterOrder(
  library::RasterOrder &raster_order,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_ProviderID(library::Provider &provider, KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_ProviderID(
  library::Provider &provider,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/// Lexically casts an argument to a given type stored in a byte array. Returns true if not null.
bool arg_as_scalar(
  std::vector<uint8_t> &bytes,
  library::NumericTypeID numeric_type, 
  KernelArgument::Value const *value_ptr);

/// Lexically casts an argument to a given type stored in a byte array. Returns true if not null.
bool arg_as_scalar(
  std::vector<uint8_t> &bytes,
  library::NumericTypeID numeric_type, 
  char const *name, 
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

bool arg_as_string(
  std::string& arg,
  char const* name,
  ProblemSpace const& problem_space,
  ProblemSpace::Problem const& problem);

/// Returns true if a tensor description satisfies a `tensor` value
bool tensor_description_satisfies(
  library::TensorDescription const &tensor_desc,
  TensorArgument::TensorValue const *value_ptr);

/// Returns true if a tensor description satisfies a `tensor` value
bool tensor_description_satisfies(
  library::TensorDescription const &tensor_desc,
  char const *name, 
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);


/// Returns true if a conv kind satisfies the value
bool conv_kind_satisfies(
  library::ConvKind const &conv_kind,
  EnumeratedTypeArgument::EnumeratedTypeValue const *value_ptr);

/// Returns true if a conv kind satisfies the value
bool conv_kind_satisfies(
  library::ConvKind const &conv_kind,
  char const *name, 
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/// Returns true if a iterator algorithm satisfies the value
bool iterator_algorithm_satisfies(
  library::IteratorAlgorithmID const &iterator_algorithm,
  EnumeratedTypeArgument::EnumeratedTypeValue const *value_ptr);

/// Returns true if a iterator algorithm satisfies the value
bool iterator_algorithm_satisfies(
  library::IteratorAlgorithmID const &iterator_algorithm,
  char const *name, 
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem);

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////
