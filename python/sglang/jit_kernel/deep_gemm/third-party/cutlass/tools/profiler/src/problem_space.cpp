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
*/

#include <string>
#include <stdexcept>
#include <sstream>

#include "cutlass/library/util.h"

#include "cutlass/profiler/problem_space.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static T lexical_cast(std::string const &str) {
  std::stringstream ss;
  T value;
  
  ss << str;
  ss >> value;

  return value;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream & KernelArgument::ValueIterator::print(std::ostream &out) const {
  out << "[" << (void *)this << "  " <<  argument->qualified_name() << "] ";
  if (this->null_argument) {
    out << "<null>";
  }
  else {
    out << "<not null>";
  }
  return out;
}

KernelArgument::~KernelArgument() {

}

//////////////////////////////////////////////////////////////////////////////////////////////////

ScalarArgument::ScalarValue::ScalarValue(
  std::string const &value_,
  ScalarArgument const *argument_,
  bool not_null_
):
  KernelArgument::Value(argument_, not_null_),
  value(value_) {

}

std::ostream &ScalarArgument::ScalarValue::print(std::ostream &out) const {
  out << argument->qualified_name() << ": ";
  if (not_null) {
    out << value;
  }
  else {
    out << "<null>";
  }
  return out;
}

ScalarArgument::ScalarValueIterator::ScalarValueIterator(
  ScalarArgument const *argument_
): 
  KernelArgument::ValueIterator(argument_) {

  if (argument_) {
    value_it = argument_->values.begin(); 
  }
}

void ScalarArgument::ScalarValueIterator::operator++() {
  if (this->null_argument) {
    this->null_argument = false;
  }
  else {
    ++value_it; 
  }
}

bool ScalarArgument::ScalarValueIterator::operator==(ValueIterator const &it) const {
  if (it.type() != ArgumentTypeID::kScalar) {
    throw std::runtime_error("Cannot compare ScalarValueIterator with iterator of different type");
  }
  auto const & scalar_it = static_cast<ScalarValueIterator const &>(it);
  return value_it == scalar_it.value_it;
}

/// Gets the value pointed to
std::unique_ptr<KernelArgument::Value> ScalarArgument::ScalarValueIterator::at() const {
  if (this->null_argument) {
    return std::unique_ptr<KernelArgument::Value>(
      new ScalarArgument::ScalarValue(
        std::string(), 
        static_cast<ScalarArgument const *>(argument),
        false)); 
  }
  else {
    return std::unique_ptr<KernelArgument::Value>(
      new ScalarArgument::ScalarValue(
        *value_it, 
        static_cast<ScalarArgument const *>(argument))); 
  }
}

std::unique_ptr<KernelArgument::ValueIterator> ScalarArgument::begin() const {
  return std::unique_ptr<KernelArgument::ValueIterator>(new ScalarValueIterator(this));
}

std::unique_ptr<KernelArgument::ValueIterator> ScalarArgument::end() const {
  ScalarValueIterator *it = new ScalarValueIterator(this);
  it->value_it = this->values.end();
  it->null_argument = false;
  return std::unique_ptr<ValueIterator>(it);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

IntegerArgument::IntegerValue::IntegerValue(
  int64_t value_, 
  IntegerArgument const *argument_, 
  bool not_null_
): KernelArgument::Value(argument_, not_null_), value(value_) {

}


/// Pretty printer for debugging
std::ostream &IntegerArgument::IntegerValue::print(std::ostream &out) const {
  out << argument->qualified_name() << ": ";
  if (not_null) {
    out << value;
  }
  else {
    out << "<null>";
  }
  return out;
}

IntegerArgument::IntegerValueIterator::IntegerValueIterator(IntegerArgument const *argument_): 
  KernelArgument::ValueIterator(argument_) {

  if (argument_) {
    range_it = argument_->ranges.begin();
    if (range_it != argument_->ranges.end()) {
      value_it = range_it->begin();
    }
  }
}

void IntegerArgument::IntegerValueIterator::operator++() {

  if (this->null_argument) {
    this->null_argument = false;
  }
  else {
    ++value_it;
    if (value_it == range_it->end()) {
      ++range_it;
      if (range_it != static_cast<IntegerArgument const *>(argument)->ranges.end()) {
        value_it = range_it->begin();
      }
    }
  }
}

bool IntegerArgument::IntegerValueIterator::operator==(ValueIterator const &it) const {
  if (it.type() != ArgumentTypeID::kInteger) {
    throw std::runtime_error("Cannot compare IntegerValueIterator with iterator of different type");
  }
  
  auto const & integer_iterator = static_cast<IntegerValueIterator const &>(it);

  if (this->null_argument) {
    return it.null_argument;
  }
  else {
    if (range_it != integer_iterator.range_it) {
      return false;
    }
    if (range_it == static_cast<IntegerArgument const *>(argument)->ranges.end() &&
      range_it == integer_iterator.range_it) {
      return true;
    }
    return value_it == integer_iterator.value_it;
  }
}

std::unique_ptr<KernelArgument::Value> IntegerArgument::IntegerValueIterator::at() const {
  if (this->null_argument) {
    return std::unique_ptr<KernelArgument::Value>(
      new IntegerArgument::IntegerValue(
        0, static_cast<IntegerArgument const *>(argument), false));  
  }
  else {
    return std::unique_ptr<KernelArgument::Value>(
      new IntegerArgument::IntegerValue(
        *value_it, static_cast<IntegerArgument const *>(argument)));  
  }
}

std::unique_ptr<KernelArgument::ValueIterator> IntegerArgument::begin() const {
  return std::unique_ptr<KernelArgument::ValueIterator>(new IntegerValueIterator(this));
}

std::unique_ptr<KernelArgument::ValueIterator> IntegerArgument::end() const {
  IntegerValueIterator *it = new IntegerValueIterator(this);
  it->range_it = this->ranges.end();
  it->null_argument = false;
  return std::unique_ptr<ValueIterator>(it);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

TensorArgument::TensorValue::TensorValue(
  TensorDescription const &desc_,
  TensorArgument const *argument_, 
  bool not_null_ 
):
  KernelArgument::Value(argument_, not_null_),
  desc(desc_) {

}

/// Pretty printer for debugging
std::ostream &TensorArgument::TensorValue::print(std::ostream &out) const {
  out << argument->qualified_name() << ": " << to_string(desc.element) << ": " << to_string(desc.layout);
  return out;
}

TensorArgument::TensorValueIterator::TensorValueIterator(
  TensorArgument const *argument_
): 
  KernelArgument::ValueIterator(argument_) {

  if (argument_) {
    value_it = argument_->values.begin();
  }
}

void TensorArgument::TensorValueIterator::operator++() {
  if (this->null_argument) {
    this->null_argument = false;
  }
  else {
    ++value_it;
  }
}

bool TensorArgument::TensorValueIterator::operator==(ValueIterator const &it) const {
  if (it.type() != ArgumentTypeID::kTensor) {
    throw std::runtime_error("Cannot compare TensorValueIterator with iterator of different type");
  }
  auto const & tensor_it = static_cast<TensorValueIterator const &>(it);
  return value_it == tensor_it.value_it;
}

/// Gets the value pointed to
std::unique_ptr<KernelArgument::Value> TensorArgument::TensorValueIterator::at() const {

  if (this->null_argument) {
    return std::unique_ptr<KernelArgument::Value>(
      new TensorArgument::TensorValue(
        TensorDescription(), static_cast<TensorArgument const *>(argument), false)); 
  }
  else {
    return std::unique_ptr<KernelArgument::Value>(
      new TensorArgument::TensorValue(
        *value_it, static_cast<TensorArgument const *>(argument)));  
  }
}

std::unique_ptr<KernelArgument::ValueIterator> TensorArgument::begin() const {
  return std::unique_ptr<KernelArgument::ValueIterator>(new TensorValueIterator(this));
}

std::unique_ptr<KernelArgument::ValueIterator> TensorArgument::end() const {
  TensorValueIterator *it = new TensorValueIterator(this);
  it->value_it = this->values.end();
  it->null_argument = false;
  return std::unique_ptr<ValueIterator>(it);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

EnumeratedTypeArgument::EnumeratedTypeValue::EnumeratedTypeValue(
  std::string const & element_,
  EnumeratedTypeArgument const *argument_, 
  bool not_null_
):
  KernelArgument::Value(argument_, not_null_),
  element(element_) {

}

/// Pretty printer for debugging
std::ostream &EnumeratedTypeArgument::EnumeratedTypeValue::print(std::ostream &out) const {
  out << argument->qualified_name() << ": " << element;
  return out;
}

EnumeratedTypeArgument::EnumeratedTypeValueIterator::EnumeratedTypeValueIterator(
  EnumeratedTypeArgument const *argument_
):
  KernelArgument::ValueIterator(argument_) {

  if (argument_) {
    value_it = argument_->values.begin();
  }
}

void EnumeratedTypeArgument::EnumeratedTypeValueIterator::operator++() {
  if (this->null_argument) {
    this->null_argument = false;
  }
  else {
    ++value_it;
  }
}

bool EnumeratedTypeArgument::EnumeratedTypeValueIterator::operator==(ValueIterator const &it) const {

  if (it.type() != ArgumentTypeID::kEnumerated) {
    throw std::runtime_error("Cannot compare EnumeratedTypeValueIterator with iterator of different type");
  }

  auto const & enumerated_type_it = static_cast<EnumeratedTypeValueIterator const &>(it);
  return value_it == enumerated_type_it.value_it;
}

/// Gets the value pointed to
std::unique_ptr<KernelArgument::Value> EnumeratedTypeArgument::EnumeratedTypeValueIterator::at() const {

  if (this->null_argument) {
    return std::unique_ptr<KernelArgument::Value>(
      new EnumeratedTypeValue(
        std::string(), static_cast<EnumeratedTypeArgument const *>(argument), false));
  }
  else {
    return std::unique_ptr<KernelArgument::Value>(
      new EnumeratedTypeValue(
        *value_it, static_cast<EnumeratedTypeArgument const *>(argument)));  
  }
}

std::unique_ptr<KernelArgument::ValueIterator> EnumeratedTypeArgument::begin() const {
  return std::unique_ptr<KernelArgument::ValueIterator>(new EnumeratedTypeValueIterator(this));
}

std::unique_ptr<KernelArgument::ValueIterator> EnumeratedTypeArgument::end() const {
  EnumeratedTypeValueIterator *it = new EnumeratedTypeValueIterator(this);
  it->value_it = this->values.end();
  it->null_argument = false;
  return std::unique_ptr<ValueIterator>(it);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

ProblemSpace::Iterator::Iterator(ProblemSpace const &problem_space) {
  for (auto const & arg_ptr : problem_space.arguments) {
    construct_(arg_ptr.get());
  }
}

ProblemSpace::Iterator::Iterator(Iterator && it) {
  iterators = std::move(it.iterators);
}

/// Helper for recursively constructing iterators
void ProblemSpace::Iterator::construct_(KernelArgument const *argument) {
  iterators.emplace_back(argument->begin());
}

/// Given a set of ranges, iterate over the points within their Cartesian product. No big deal.
void ProblemSpace::Iterator::operator++() {

  // Define a pair of iterator into the vector of iterators.
  IteratorVector::iterator iterator_it = iterators.begin(); 
  IteratorVector::iterator next_iterator = iterator_it;
  
  // Advance the first argument.
  ++(**iterator_it);

  // Maintain a pair of iterators over consecutive arguments.
  ++next_iterator;

  // Carry logic
  while (next_iterator != iterators.end() &&
    **iterator_it == *((*iterator_it)->argument->end())) {   // Did an iterator reach the end of its range?

    (*iterator_it) = (*iterator_it)->argument->begin();      // Reset that iterator,
  
    ++(**next_iterator);                                     // and increment the next argument's iterator.

    iterator_it = next_iterator;                             // Advance to the next argument
    ++next_iterator;
  }
}

/// Moves iterator to end
void ProblemSpace::Iterator::move_to_end() {
  if (!iterators.empty()) {
    std::unique_ptr<KernelArgument::ValueIterator> new_iter = iterators.back()->argument->end();
    std::swap(iterators.back(), new_iter);
  }
}

ProblemSpace::Problem ProblemSpace::Iterator::at() const {
  Problem problem;

  for (std::unique_ptr<KernelArgument::ValueIterator> const & it : iterators) {
    problem.emplace_back(it->at());
  }

  return problem;
}

/// Equality operator
bool ProblemSpace::Iterator::operator==(Iterator const &it) const {

  // This would be an opportunity for auto, but explicitly denoting references to 
  // owning smart pointers to dynamic polymorphic objects seems like a kindness to the reader.
  IteratorVector::const_iterator first_it = iterators.begin();
  IteratorVector::const_iterator second_it = it.iterators.begin();

  int idx = 0;
  for (; first_it != iterators.end(); ++first_it, ++second_it, ++idx) {

    KernelArgument::ValueIterator const *my_it = first_it->get();
    KernelArgument::ValueIterator const *their_it = second_it->get();

    if (*my_it != *their_it) {
      return false;
    }
  }

  return true;
}

std::ostream &ProblemSpace::Iterator::print(std::ostream &out) const {

  for (std::unique_ptr<KernelArgument::ValueIterator> const & iter_ptr : iterators) {
    out << "  [iter " << (iter_ptr->null_argument ? "null" : "<not null>") 
      << ", type: " << to_string(iter_ptr->argument->description->type) << "]" << std::endl;
  }

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

ProblemSpace::ProblemSpace(ArgumentDescriptionVector const &schema, CommandLine const &cmdline) {

  // Clone the arguments
  for (ArgumentDescription const & arg_desc : schema) {
    clone_(arguments, &arg_desc);
  }

  // Parse values from the command line
  for (auto & arg : arguments) {
    parse_(arg.get(), cmdline);
  }
}


/// Returns the index of an argument by name
size_t ProblemSpace::argument_index(char const *name) const {
  return argument_index_map.at(name);
}

/// Helper for recursively cloning
void ProblemSpace::clone_(
  KernelArgumentVector &kernel_args,
  ArgumentDescription const *arg_desc) {

  KernelArgument *kernel_arg = nullptr;

  switch (arg_desc->type) {
    case ArgumentTypeID::kScalar:
      kernel_arg = new ScalarArgument(arg_desc);
      break;
    case ArgumentTypeID::kInteger:
      kernel_arg = new IntegerArgument(arg_desc);
      break;
    case ArgumentTypeID::kTensor:
      kernel_arg = new TensorArgument(arg_desc);
      break;
    case ArgumentTypeID::kStructure:
    {
      throw std::runtime_error("ArgumentTypeID::kStructure not supported");
    }
      break;
    case ArgumentTypeID::kEnumerated:
      kernel_arg = new EnumeratedTypeArgument(arg_desc);
      break;
      
    default: break;
  }

  if (kernel_arg) {
    size_t idx = kernel_args.size();
    for (auto const &alias : arg_desc->aliases) {
      argument_index_map.insert(std::make_pair(alias, idx));
    }
    kernel_args.emplace_back(kernel_arg);
  }
}

/// Parses a command line
void ProblemSpace::parse_(KernelArgument *arg, CommandLine const &cmdline) {

  switch (arg->description->type) {
  case ArgumentTypeID::kScalar:
  {
    auto * scalar = static_cast<ScalarArgument *>(arg);

    for (auto const &alias : arg->description->aliases) {
      if (cmdline.check_cmd_line_flag(alias.c_str())) {

        std::vector<std::vector<std::string>> tokens;
        cmdline.get_cmd_line_argument_ranges(alias.c_str(), tokens);

        for (auto const & vec : tokens) {
          if (!vec.empty()) {
            scalar->values.push_back(vec.front());
          }
        }
        break;
      }
    }
  }
    break;
  case ArgumentTypeID::kInteger:
  {
    auto *integer = static_cast<IntegerArgument *>(arg);

    for (auto const &alias : arg->description->aliases) {
      if (cmdline.check_cmd_line_flag(alias.c_str())) {
   
        std::vector<std::vector<std::string> > tokens;
        cmdline.get_cmd_line_argument_ranges(alias.c_str(), tokens);

        for (auto &range_tokens : tokens) {

          if (!range_tokens.empty()) {

            Range range;

            if (range_tokens.front() == "rand") {
              range.mode = Range::Mode::kRandom;
            }
            else if (range_tokens.front() == "randlg2") {
              range.mode = Range::Mode::kRandomLog2;
            }

            switch (range.mode) {
              case Range::Mode::kSequence:
              {
                range.first = lexical_cast<int64_t>(range_tokens.front());
            
                if (range_tokens.size() > 1) {
                  range.last = lexical_cast<int64_t>(range_tokens.at(1));
                }
                else {
                  range.last = range.first;
                }

                if (range_tokens.size() > 2) {
                  range.increment = lexical_cast<int64_t>(range_tokens.at(2));
                }
                else {
                  range.increment = 1;
                }
              }
              break;
              case Range::Mode::kRandom: // fall-through
              case Range::Mode::kRandomLog2:
              {
                if (range_tokens.size() < 4) {
                  throw std::runtime_error(
                    "Range of mode 'rand' must have four tokens showing "
                    "the minimum, maximum, and number of iterations. For example, "
                    "rand:16:128:1000");
                }

                range.minimum = lexical_cast<int64_t>(range_tokens.at(1));
                range.maximum = lexical_cast<int64_t>(range_tokens.at(2));
                range.first = 1;
                range.last = lexical_cast<int64_t>(range_tokens.at(3));
                range.increment = 1;
                
                if (range_tokens.size() > 4) {
                  range.divisible = lexical_cast<int64_t>(range_tokens.at(4));
                }
              }
              break;
              default:
                throw std::runtime_error("Unsupported range mode.");
                break;
            }
          
            integer->ranges.push_back(range);
          }
        } 
        break;
      }
    } 
  }
    break;
  case ArgumentTypeID::kTensor:
  {
    auto *tensor = static_cast<TensorArgument *>(arg);
    
    for (auto const &alias : arg->description->aliases) {
      if (cmdline.check_cmd_line_flag(alias.c_str())) {

        std::vector<std::vector<std::string>> tokens;

        cmdline.get_cmd_line_argument_ranges(alias.c_str(), tokens);

        for (auto const & tensor_tokens : tokens) {
          if (!tensor_tokens.empty()) {
            TensorArgument::TensorDescription tensor_desc;

            tensor_desc.element = cutlass::library::from_string<library::NumericTypeID>(tensor_tokens.front());

            // Layout
            if (tensor_tokens.size() > 1) {
              tensor_desc.layout = cutlass::library::from_string<library::LayoutTypeID>(tensor_tokens.at(1));
            }

            // Stride
            for (size_t i = 2; i < tensor_tokens.size(); ++i) {
              tensor_desc.stride.push_back(lexical_cast<int>(tensor_tokens.at(i)));
            }

            tensor->values.push_back(tensor_desc);
          }
        }
        break;
      }
    }
  }
    break;
  case ArgumentTypeID::kStructure:
  {
    throw std::runtime_error("Structure arguments not supported");
  }
    break;
  case ArgumentTypeID::kEnumerated:
  {
    auto *enumerated_type = static_cast<EnumeratedTypeArgument *>(arg);

    for (auto const &alias : arg->description->aliases) {
      if (cmdline.check_cmd_line_flag(alias.c_str())) {
      
        std::vector<std::string> tokens;
        cmdline.get_cmd_line_arguments(alias.c_str(), tokens);

        for (auto const & token : tokens) {
          enumerated_type->values.push_back(token); 
        }

        break;
      }
    }    
  }
    break;
  default:
    break;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

ProblemSpace::Iterator ProblemSpace::begin() const {
  return ProblemSpace::Iterator(*this);
}

ProblemSpace::Iterator ProblemSpace::end() const {
  ProblemSpace::Iterator it(*this);
  it.move_to_end();
  return it;
}

/// Gets all argument names as an ordered vector
std::vector<std::string> ProblemSpace::argument_names() const {

  Problem problem = this->begin().at();

  std::vector<std::string> names;
  names.reserve(problem.size());
  
  for (auto const & arg : problem) {
    names.push_back(arg->argument->description->aliases.front());
  }

  return names;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_int(int64_t &int_value, KernelArgument::Value const *value_ptr) {
  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kInteger) {
      int_value = static_cast<IntegerArgument::IntegerValue const *>(value_ptr)->value; 
    }
    else if (value_ptr->argument->description->type == ArgumentTypeID::kScalar) {
      std::stringstream ss;
      ss << static_cast<ScalarArgument::ScalarValue const *>(value_ptr)->value;
      ss >> int_value; 
    }
    else {
      throw std::runtime_error(
        "arg_as_int64_t() - illegal cast. Problem space argument must be integer or scalar");
    }

    return true;
  }

  return false;
}

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_int(int &int_value, KernelArgument::Value const *value_ptr) {
  int64_t value64;
  bool obtained = arg_as_int(value64, value_ptr);
  if (obtained) {
    int_value = int(value64);
    return true;
  }
  return false;
}

/// Lexically casts an argument to an int
bool arg_as_int(
  int &int_value,
  char const *name,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_int(int_value, value_ptr);
}

/// Lexically casts an argument to an int64
bool arg_as_int(
  int64_t &int_value,
  char const *name,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_int(int_value, value_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexically casts an argument to an bool if it is defined. Returns true if not null.
bool arg_as_bool(bool &bool_value, KernelArgument::Value const *value_ptr) {
  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kInteger) {
      int64_t value64;
      arg_as_int(value64, value_ptr);
      bool_value = static_cast<bool>(value64); 
    }
    else if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {
      bool_value = library::from_string<bool>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);
    }
    else {
      throw std::runtime_error(
        "arg_as_bool() - illegal cast. Problem space argument must be integer or enumerated");
    }

    return true;
  }

  return false;
}

/// Lexically casts an argument to a bool
bool arg_as_bool(
  bool &bool_value,
  char const *name,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_bool(bool_value, value_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_NumericTypeID(
  library::NumericTypeID &numeric_type, 
  KernelArgument::Value const *value_ptr) {
  
  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {

      numeric_type = library::from_string<library::NumericTypeID>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);

      if (numeric_type == library::NumericTypeID::kInvalid) {
        throw std::runtime_error(
          "arg_as_NumericTypeID() - illegal cast.");
      }
    }
    else {

      throw std::runtime_error(
        "arg_as_NumericTypeID() - illegal cast.");
    }
    return true;
  }
  return false;
}

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_NumericTypeID(
  library::NumericTypeID &numeric_type,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_NumericTypeID(numeric_type, value_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_RuntimeDatatype(
  library::RuntimeDatatype &runtime_datatype, 
  KernelArgument::Value const *value_ptr) {
  
  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {
      
      runtime_datatype = library::from_string<library::RuntimeDatatype>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);
      if (runtime_datatype == library::RuntimeDatatype::kInvalid) {
        throw std::runtime_error(
          "arg_as_RuntimeDatatype() - illegal cast.");
      }
    }
    else {
      throw std::runtime_error(
        "arg_as_RuntimeDatatype() - illegal cast.");
    }
    return true;
  }
  return false;
}


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_RasterOrder(
  library::RasterOrder &raster_order, 
  KernelArgument::Value const *value_ptr) {
  
  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {

      raster_order = library::from_string<library::RasterOrder>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);

      if (raster_order == library::RasterOrder::kInvalid) {
        throw std::runtime_error(
          "arg_as_RasterOrder() - illegal cast.");
      }
    }
    else {
      throw std::runtime_error(
        "arg_as_RasterOrder() - illegal cast.");
    }
    return true;
  }
  return false;
}

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_RasterOrder(
  library::RasterOrder &raster_order,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_RasterOrder(raster_order, value_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_LayoutTypeID(
  library::LayoutTypeID &layout_type, 
  KernelArgument::Value const *value_ptr) {

  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {

      layout_type = library::from_string<library::LayoutTypeID>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);

      if (layout_type == library::LayoutTypeID::kInvalid) {
        throw std::runtime_error(
          "arg_as_LayoutTypeID() - illegal cast.");
      }
    }
    else {

      throw std::runtime_error(
        "arg_as_LayoutTypeID() - illegal cast.");
    }
    return true;
  }
  return false;
}


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_RuntimeDatatype(
  library::RuntimeDatatype &runtime_datatype,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_RuntimeDatatype(runtime_datatype, value_ptr);
}


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_LayoutTypeID(
  library::LayoutTypeID &layout_type,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_LayoutTypeID(layout_type, value_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_OpcodeClassID(
  library::OpcodeClassID &opcode_class,
  KernelArgument::Value const *value_ptr) {

  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {

      opcode_class = library::from_string<library::OpcodeClassID>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);

      if (opcode_class == library::OpcodeClassID::kInvalid) {
        throw std::runtime_error(
          "arg_as_OpcodeClassID() - illegal cast.");
      }
    }
    else {

      throw std::runtime_error(
        "arg_as_OpcodeClassID() - illegal cast.");
    }
    return true;
  }
  return false;
}

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_OpcodeClassID(
  library::OpcodeClassID &opcode_class,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_OpcodeClassID(opcode_class, value_ptr);
}


/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_SplitKModeID(
  library::SplitKMode &split_k_mode,
  KernelArgument::Value const *value_ptr) {

  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {

      split_k_mode = library::from_string<library::SplitKMode>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);

      if (split_k_mode == library::SplitKMode::kInvalid) {
        throw std::runtime_error(
          "arg_as_SplitKModeID() - illegal cast.");
      }
    }
    else {

      throw std::runtime_error(
        "arg_as_SplitKModeID() - illegal cast.");
    }
    return true;
  }
  return false;
}

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_SplitKModeID(
  library::SplitKMode &split_k_mode,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_SplitKModeID(split_k_mode, value_ptr);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_ConvModeID(
  library::ConvModeID &conv_mode,
  KernelArgument::Value const *value_ptr) {

  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {

      conv_mode = library::from_string<library::ConvModeID>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);

      if (conv_mode == library::ConvModeID::kInvalid) {
        throw std::runtime_error(
          "arg_as_ConvModeID() - illegal cast.");
      }
    }
    else {

      throw std::runtime_error(
        "arg_as_ConvModeID() - illegal cast.");
    }
    return true;
  }
  return false;
}

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_ConvModeID(
  library::ConvModeID &conv_mode,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_ConvModeID(conv_mode, value_ptr);
}

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_ProviderID(
  library::Provider &provider,
  KernelArgument::Value const *value_ptr) {

  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {

      provider = library::from_string<library::Provider>(
        static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr)->element);

      if (provider == library::Provider::kInvalid) {
        throw std::runtime_error(
          "arg_as_ProviderID() - illegal cast.");
      }
    }
    else {

      throw std::runtime_error(
        "arg_as_ProviderID() - illegal cast.");
    }
    return true;
  }
  return false;
}

/// Lexically casts an argument to an int64 if it is defined. Returns true if not null.
bool arg_as_ProviderID(
  library::Provider &provider,
  char const *name,
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_ProviderID(provider, value_ptr);
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexically casts an argument to a given type stored in a byte array. Returns true if not null.
bool arg_as_scalar(
  std::vector<uint8_t> &bytes,
  library::NumericTypeID numeric_type,
  KernelArgument::Value const *value_ptr) {

  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kInteger) {
      int64_t int_value = static_cast<IntegerArgument::IntegerValue const *>(value_ptr)->value;
      
      // TODO - convert int64_t => destination type
    }
    else if (value_ptr->argument->description->type == ArgumentTypeID::kScalar) {
      std::string const &str_value = static_cast<ScalarArgument::ScalarValue const *>(value_ptr)->value;

      return lexical_cast(bytes, numeric_type, str_value);
    }
    else {
      throw std::runtime_error(
        "arg_as_int() - illegal cast. Problem space argument must be integer or scalar");
    }

    return true;
  }

  return false;
}

/// Lexically casts an argument to a given type and returns a byte array
bool arg_as_scalar(
  std::vector<uint8_t> &bytes,
  library::NumericTypeID numeric_type,
  char const *name,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {
  
  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  return arg_as_scalar(bytes, numeric_type, value_ptr);
}

/// Returns a copy of the string passed to the argument.
/// (kScalar arguments are stored as strings).
bool arg_as_string(
  std::string& arg,
  char const* name,
  ProblemSpace const& problem_space,
  ProblemSpace::Problem const& problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const* value_ptr = problem.at(idx).get();

  if (value_ptr->not_null) {
    if (value_ptr->argument->description->type == ArgumentTypeID::kScalar) {
      std::string const& str_value =
        static_cast<ScalarArgument::ScalarValue const*>(value_ptr)->value;
      arg = std::string(str_value);
    }
    else {
      throw std::runtime_error(
        "arg_as_string() - illegal cast. Problem space argument must be scalar");
    }

    return true;
  }

  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if a tensor description satisfies a `tensor` value
bool tensor_description_satisfies(
  library::TensorDescription const &tensor_desc,
  TensorArgument::TensorValue const *value_ptr) {

  if (value_ptr->not_null) {
    if (value_ptr->desc.element != library::NumericTypeID::kUnknown && 
      value_ptr->desc.element != tensor_desc.element) {

      return false;
    }

    if (value_ptr->desc.layout != library::LayoutTypeID::kUnknown &&
      value_ptr->desc.layout != tensor_desc.layout) {

      return false;
    }
  }

  return true;
}

/// Returns true if a tensor description satisfies a `tensor` value
bool tensor_description_satisfies(
  library::TensorDescription const &tensor_desc,
  char const *name, 
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  if (value_ptr->argument->description->type == ArgumentTypeID::kTensor) {
    return tensor_description_satisfies(
      tensor_desc, 
      static_cast<TensorArgument::TensorValue const *>(value_ptr));
  }
  else {
    throw std::runtime_error("Kernel argument mismatch");
  }

  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if conv_kind satisfies the value
bool conv_kind_satisfies(
  library::ConvKind const &conv_kind,
  EnumeratedTypeArgument::EnumeratedTypeValue const *value_ptr) {

  if (value_ptr->not_null) {
    library::ConvKind conv_kind_cmd_line = 
      library::from_string<library::ConvKind>(value_ptr->element);

    if (conv_kind_cmd_line != library::ConvKind::kUnknown && 
      conv_kind_cmd_line != conv_kind) {

      return false;
    }
  }

  return true;
}

/// Returns true if conv_kind satisfies the value
bool conv_kind_satisfies(
  library::ConvKind const &conv_kind,
  char const *name, 
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {
    return conv_kind_satisfies(
      conv_kind, 
      static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr));
  }
  else {
    throw std::runtime_error("Kernel argument mismatch");
  }

  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if a iterator algorithm satisfies the value
bool iterator_algorithm_satisfies(
  library::IteratorAlgorithmID const &iterator_algorithm,
  EnumeratedTypeArgument::EnumeratedTypeValue const *value_ptr) {

  if (value_ptr->not_null) {
    library::IteratorAlgorithmID iterator_algorithm_cmd_line = 
      library::from_string<library::IteratorAlgorithmID>(value_ptr->element);

    if (iterator_algorithm_cmd_line != library::IteratorAlgorithmID::kNone && 
      iterator_algorithm_cmd_line != iterator_algorithm) {

      return false;
    }
  }

  return true;
}

/// Returns true if a iterator algorithm satisfies the value
bool iterator_algorithm_satisfies(
  library::IteratorAlgorithmID const &iterator_algorithm,
  char const *name, 
  ProblemSpace const &problem_space, 
  ProblemSpace::Problem const &problem) {

  size_t idx = problem_space.argument_index(name);
  KernelArgument::Value const *value_ptr = problem.at(idx).get();

  if (value_ptr->argument->description->type == ArgumentTypeID::kEnumerated) {
    return iterator_algorithm_satisfies(
      iterator_algorithm, 
      static_cast<EnumeratedTypeArgument::EnumeratedTypeValue const *>(value_ptr));
  }
  else {
    throw std::runtime_error("Kernel argument mismatch");
  }

  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
