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
   \brief Execution environment
*/

#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cstring>

#include "cutlass/library/util.h"

#include "cutlass/library/util.h"

#include "cutlass/profiler/performance_report.h"
#include "cutlass/profiler/debug.h"
namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__unix__)

#define SHELL_COLOR_BRIGHT()  "\033[1;37m"
#define SHELL_COLOR_GREEN()   "\033[1;32m"
#define SHELL_COLOR_RED()     "\033[1;31m"
#define SHELL_COLOR_END()     "\033[0m"

#else

#define SHELL_COLOR_BRIGHT()  ""
#define SHELL_COLOR_GREEN()   ""
#define SHELL_COLOR_RED()     ""
#define SHELL_COLOR_END()     ""

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

PerformanceReport::PerformanceReport(
  Options const &options,
  std::vector<std::string> const &argument_names,
  library::OperationKind const &op_kind
):
  options_(options), argument_names_(argument_names), problem_index_(0), good_(true), op_kind_(op_kind) {

  // Strip '.csv' if present
  std::string base_path = options_.report.output_path;
  base_path = base_path.substr(0, base_path.rfind(".csv"));
  op_file_name_ = base_path + "." + to_string(op_kind_) + ".csv";

  base_path = options_.report.junit_output_path;
  base_path = base_path.substr(0, base_path.rfind(".xml"));
  base_path = base_path.substr(0, base_path.rfind(".junit"));
  op_junit_file_name_ = base_path + "." + to_string(op_kind_) + ".junit.xml";

  //
  // Open output file for operation of PerformanceReport::op_kind
  //
  if (!options_.report.output_path.empty()) {

    bool print_header = true;

    if (options_.report.append) {

      std::ifstream test_output_file(op_file_name_);

      if (test_output_file.is_open()) {
        print_header = false;
        test_output_file.close();
      }

      output_file_.open(op_file_name_, std::ios::app);
    }
    else {
      output_file_.open(op_file_name_);
    }

    if (!output_file_.good()) {

      std::cerr << "Could not open output file at path '"
         << options_.report.output_path << "'" << std::endl;

      good_ = false;
    }

    if (print_header) {
      print_csv_header_(output_file_) << std::endl;
    }
  }

  if (!options_.report.junit_output_path.empty()) {

    junit_output_file_.open(op_junit_file_name_);

    if (!junit_output_file_.good()) {

      std::cerr << "Could not open junit output file at path '"
         << options_.report.junit_output_path << "'" << std::endl;

      good_ = false;
    }

    print_junit_header_(junit_output_file_);
  }
}

void PerformanceReport::next_problem() {
  ++problem_index_;
}

void PerformanceReport::append_result(PerformanceResult result) {

  result.problem_index = problem_index_;

  if (options_.report.verbose) {
    std::cout << "\n";
    print_result_pretty_(std::cout, result) << std::flush;
  }

  if (junit_output_file_.is_open()) {
    print_junit_result_(junit_output_file_, result);
  }

  if (output_file_.is_open()) {
    print_result_csv_(output_file_, result) << std::endl;
  }
  else {
    concatenated_results_.push_back(result);
  }
}

void PerformanceReport::sort_flops_per_byte(PerformanceResultVector &results) {

  struct FlopsPerByteCompare
  {
    bool operator()(const PerformanceResult &a, const PerformanceResult &b)
    {
      double a_flops_per_byte = double(a.flops) / double(a.bytes);
      double b_flops_per_byte = double(b.flops) / double(b.bytes);

      return (a_flops_per_byte < b_flops_per_byte);
    }
  };

  std::stable_sort(results.begin(), results.end(), FlopsPerByteCompare());
}

void PerformanceReport::sort_flops_per_sec(PerformanceResultVector &results) {

  struct FlopsPerSecondCompare
  {
    bool operator()(const PerformanceResult &a, const PerformanceResult &b)
    {
      return a.gflops_per_sec() > b.gflops_per_sec();
    }
  };

  std::stable_sort(results.begin(), results.end(), FlopsPerSecondCompare());
}

void PerformanceReport::append_results(PerformanceResultVector const &results) {

  if (options_.report.verbose) {
    std::cout << "\n\n";
  }

  // For each result
  for (auto const & result : results) {
    append_result(result);
  }
}

PerformanceReport::~PerformanceReport() {

  //
  // Output results to stdout if they were not written to a file already.
  //
  if (options_.report.verbose && !concatenated_results_.empty()) {

    if (options_.report.sort_flops_per_byte) {
      sort_flops_per_byte(concatenated_results_);
    }

    if (options_.report.sort_flops_per_sec) {
      sort_flops_per_sec(concatenated_results_);
    }

    std::cout << "\n\n";
    std::cout << "=============================\n\n";
    std::cout << "CSV Results:\n\n";

    print_csv_header_(std::cout) << std::endl;

    for (auto const &result : concatenated_results_) {
      print_result_csv_(std::cout, result) << "\n";
    }
  }
  else if (output_file_.is_open() && options_.report.verbose) {
    std::cout << "\nWrote results to '" << op_file_name_ << "'" << std::endl;
  }

  if (output_file_.is_open()) {
    output_file_.close();
  }

  if (junit_output_file_.is_open()) {
    print_junit_footer_(junit_output_file_);
    junit_output_file_.close();
    std::cout << "\nWrote jUnit results to '" << op_junit_file_name_ << "'" << std::endl;
  }
}

static const char *disposition_status_color(Disposition disposition) {
  switch (disposition) {
    case Disposition::kPassed: return SHELL_COLOR_GREEN();
    case Disposition::kIncorrect: return SHELL_COLOR_RED();
    case Disposition::kFailed: return SHELL_COLOR_RED();
    default:
    break;
  }
  return SHELL_COLOR_END();
}

/// Prints the result in human readable form
std::ostream & PerformanceReport::print_result_pretty_(
  std::ostream &out,
  PerformanceResult const &result,
  bool use_shell_coloring) {

  out << "=============================\n"
    << "  Problem ID: " << result.problem_index << "\n";

  if (!options_.report.pivot_tags.empty()) {

    out << "        Tags: ";

    int column_idx = 0;
    for (auto const & tag : options_.report.pivot_tags) {
      out << (column_idx++ ? "," : "") << tag.first << ":" << tag.second;
    }

    out << "\n";
  }

  std::string shell_color_bright = use_shell_coloring ? SHELL_COLOR_BRIGHT() : "";
  std::string shell_color_end = use_shell_coloring ? SHELL_COLOR_END() : "";
  auto _disposition_status_color = [&](Disposition d) -> const char * {
    return use_shell_coloring ? disposition_status_color(d) : "";
  };

  out
    << "\n"
    << "        Provider: " << shell_color_bright << library::to_string(result.provider, true) << shell_color_end << "\n"
    << "   OperationKind: " << shell_color_bright << library::to_string(result.op_kind) << shell_color_end << "\n"
    << "       Operation: " << result.operation_name << "\n\n"
    << "          Status: " << shell_color_bright << library::to_string(result.status, true) << shell_color_end << "\n"
    << "    Verification: " << shell_color_bright << (options_.verification.enabled ? "ON":"OFF") << shell_color_end << "\n"
    << "     Disposition: " << _disposition_status_color(result.disposition) << to_string(result.disposition, true) << shell_color_end << "\n\n";

  // Display individual verification results for each verification-provider
  if (options_.verification.enabled) {

    static int const indent_spaces = 16;

    for(auto & m : result.verification_map) {
      out  << std::right << std::setw(indent_spaces) << library::to_string(m.first, true) << ": " << to_string(m.second, true) << "\n";
    }
  }

  out
    << "\n       Arguments:";

  int column_idx = 0;
  for (auto const &arg : result.arguments) {
    if (!arg.second.empty()) {
      out << " --" << arg.first << "=" << arg.second;
      column_idx += int(4 + arg.first.size() + arg.second.size());
      if (column_idx > 98) {
        out << "  \\\n                 ";
        column_idx = 0;
      }
    }
  }
  out << "\n\n";

  out
    << "           Bytes: " << result.bytes << "  bytes\n"
    << "           FLOPs: " << result.flops << "  flops\n"
    << "           FLOPs/Byte: " << (result.flops / result.bytes) << "\n\n";

  if (result.good()) {

    out
      << "         Runtime: " << result.runtime << "  ms\n"
      << "          Memory: " << result.gbytes_per_sec() << " GiB/s\n"
      << "\n            Math: " << result.gflops_per_sec() << " GFLOP/s\n";

  }

  return out;
}

/// Prints the CSV header
std::ostream & PerformanceReport::print_csv_header_(
  std::ostream &out) {

  int column_idx = 0;

  // Pivot tags
  for (auto const & tag : options_.report.pivot_tags) {
    out << (column_idx++ ? "," : "") << tag.first;
  }

  out
    << (column_idx ? "," : "") << "Problem,Provider"
    << ",OperationKind,Operation,Disposition,Status";

  for (auto const &arg_name : argument_names_) {
    out << "," << arg_name;
  }

  out
    << ",Bytes"
    << ",Flops"
    << ",Flops/Byte"
    << ",Runtime";

  if (options_.device.devices.size() > 1) {
    for (size_t i = 0; i < options_.device.devices.size(); i++) {
      out << ",Runtime_" << i;
    }
  }

  out
    << ",GB/s"
    << ",GFLOPs"
    ;

  return out;
}

/// Print the result in CSV output
std::ostream & PerformanceReport::print_result_csv_(
  std::ostream &out,
  PerformanceResult const &result) {

  int column_idx = 0;

  // Pivot tags
  for (auto const & tag : options_.report.pivot_tags) {
    out << (column_idx++ ? "," : "") << tag.second;
  }

  out
    << (column_idx ? "," : "")
    << result.problem_index
    << "," << to_string(result.provider, true)
    << "," << to_string(result.op_kind)
    << "," << result.operation_name
    << "," << to_string(result.disposition)
    << "," << library::to_string(result.status);

  for (auto const & arg : result.arguments) {
    out << "," << arg.second;
  }

  out
    << "," << result.bytes
    << "," << result.flops
    << "," << result.flops / result.bytes
    << "," << result.runtime;

  if (options_.device.devices.size() > 1) {
    if (result.runtime_vector.size() != options_.device.devices.size()) {
      throw std::runtime_error("Runtime vector size mismatch");
    }

    for (const auto runtime : result.runtime_vector) {
      out << "," << runtime;
    }
  }

  if (result.good()) {

    out
      << "," << result.gbytes_per_sec()
      << "," << result.gflops_per_sec()
      ;

  }
  else {
    out << std::string(2
      , ','
    );
  }

  return out;
}

std::ostream & PerformanceReport::print_junit_header_(std::ostream &out) {

  out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
  out << "<testsuite name=\"cutlass_profiler\">" << std::endl;
  return out;

}

namespace {

  std::string escape_xml_special_chars(const std::string& src) {
    std::stringstream dst;
    for (char ch : src) {
      switch (ch) {
      case '&': dst << "&amp;"; break;
      case '\'': dst << "&apos;"; break;
      case '"': dst << "&quot;"; break;
      case '<': dst << "&lt;"; break;
      case '>': dst << "&gt;"; break;
      default: dst << ch; break;
      }
    }
    return dst.str();
  }

  template<typename T>
  std::ostream & print_junit_result_property_(std::ostream & os, const std::string & name, const T & property) {
    return os << "    <property name=\"" << name << "\" value=\"" << property << "\" />" << std::endl;
  }
}

std::ostream & PerformanceReport::print_junit_result_(std::ostream &out, PerformanceResult const &result) {

  out << "  " << "<testcase name=\"";

  std::string delim = "";

  // Pivot tags
  for (auto const & tag : options_.report.pivot_tags) {
    out << delim << tag.second; delim = "_";
  }

  out << delim << to_string(result.op_kind); delim = "_";
  out << delim << result.operation_name;

  for (auto const & arg : result.arguments) {
    out << delim << arg.second;
  }

  out << "\" ";

  bool skipped = false, failed = false, error = false;

  switch (result.disposition) {
  case Disposition::kNotRun:
  case Disposition::kNotSupported:
    skipped = true;
    break;
  case Disposition::kPassed:
  case Disposition::kNotVerified:
    break;
  case Disposition::kFailed:
  case Disposition::kIncorrect:
    failed = true;
    break;
  case Disposition::kInvalidProblem:
  case Disposition::kInvalid:
    error = true;
    break;
  };

  if (skipped) {
    out << "status=\"notrun\"";
  } else {
    out << "status=\"run\"";
  }

  out << ">" << std::endl;

  if (failed) {
    out << "    <failure message=\"" << to_string(result.disposition) << "\" />" << std::endl;
  }

  if (error) {
    out << "    <error message=\"" << to_string(result.disposition) << "\" />" << std::endl;
  }

  out << "    <system-out><![CDATA[" << std::endl;
  std::stringstream ss;
  print_result_pretty_(ss, result, false);
  out << escape_xml_special_chars(ss.str()) << std::endl;
  out << "    ]]></system-out>" << std::endl;

  out << "  </testcase>" << std::endl;

  return out;

}

std::ostream & PerformanceReport::print_junit_footer_(std::ostream &out) {

  out << "</testsuite>" << std::endl;
  return out;

}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
