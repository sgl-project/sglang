#ifndef __SPECTRE_PROTOCOL_HPP_INCLUDED_
#define __SPECTRE_PROTOCOL_HPP_INCLUDED_

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <msgpack.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zmq.hpp>

namespace py = pybind11;

namespace spectre {
enum class SpectreAction { DRAFT = 0, FINISH = 1, ABORT = 2, REJECT = 3 };

enum class SpecType { NORMAL = 0, DRAFT_REQUEST = 1, DRAFT_RESPONSE = 2 };

} // namespace spectre

MSGPACK_ADD_ENUM(spectre::SpectreAction);
MSGPACK_ADD_ENUM(spectre::SpecType);

namespace spectre {
inline std::string to_string(SpecType t) {
  static const std::map<SpecType, std::string> m = {
      {SpecType::NORMAL, "normal"},
      {SpecType::DRAFT_REQUEST, "draft_request"},
      {SpecType::DRAFT_RESPONSE, "draft_response"}};
  return m.at(t);
}

inline std::string to_string(SpectreAction t) {
  static const std::map<SpectreAction, std::string> m = {
      {SpectreAction::DRAFT, "draft"},
      {SpectreAction::FINISH, "finish"},
      {SpectreAction::ABORT, "abort"},
      {SpectreAction::REJECT, "reject"}};
  return m.at(t);
}

inline SpecType str_to_spec_type(const std::string &s) {
  if (s == "normal")
    return SpecType::NORMAL;
  if (s == "draft_request")
    return SpecType::DRAFT_REQUEST;
  if (s == "draft_response")
    return SpecType::DRAFT_RESPONSE;
  throw std::invalid_argument("Invalid SpecType: " + s);
}

inline SpectreAction str_to_remote_action(const std::string &s) {
  if (s == "draft")
    return SpectreAction::DRAFT;
  if (s == "finish")
    return SpectreAction::FINISH;
  if (s == "abort")
    return SpectreAction::ABORT;
  if (s == "reject")
    return SpectreAction::REJECT;
  throw std::invalid_argument("Invalid SpectreAction: " + s);
}

struct SamplingParams {
  int max_new_tokens = 128;
  std::optional<std::vector<std::string>> stop_strs;
  std::optional<std::vector<int>> stop_token_ids;
  std::optional<std::vector<std::string>> stop_regex_strs;
  float temperature = 1.0f;
  float top_p = 1.0f;
  int top_k = -1;
  float min_p = 0.0f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  float repetition_penalty = 1.0f;
  int min_new_tokens = 0;
  int n = 1;
  std::optional<std::string> json_schema;
  std::optional<std::string> regex;
  std::optional<std::string> ebnf;
  std::optional<std::string> structural_tag;
  bool ignore_eos = false;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;
  bool no_stop_trim = false;
  std::optional<std::map<std::string, std::string>> custom_params;
  std::optional<int> stream_interval;
  std::optional<std::map<std::string, float>> logit_bias;
  std::optional<int> sampling_seed;

  MSGPACK_DEFINE(max_new_tokens, stop_strs, stop_token_ids, stop_regex_strs,
                 temperature, top_p, top_k, min_p, frequency_penalty,
                 presence_penalty, repetition_penalty, min_new_tokens, n,
                 json_schema, regex, ebnf, structural_tag, ignore_eos,
                 skip_special_tokens, spaces_between_special_tokens,
                 no_stop_trim, custom_params, stream_interval, logit_bias,
                 sampling_seed);
};

struct SpectreRequest {
  std::optional<std::string> request_id;
  std::optional<int> spec_cnt;
  SpectreAction action = SpectreAction::FINISH;
  SpecType spec_type = SpecType::NORMAL;
  std::optional<std::vector<int>> draft_token_ids;
  std::optional<std::vector<int>> input_ids;
  std::optional<std::vector<int>> output_ids;
  std::optional<int> num_draft_tokens;
  std::optional<SamplingParams> sampling_params;
  std::optional<std::string> grammar;
  double target_send_time = -1.0;
  double target_recv_time = -1.0;
  std::optional<std::vector<float>> draft_logprobs;
  double draft_recv_time = -1.0;
  double draft_send_time = -1.0;

  MSGPACK_DEFINE(request_id, spec_cnt, action, spec_type, draft_token_ids,
                 input_ids, output_ids, num_draft_tokens, sampling_params,
                 grammar, target_send_time, target_recv_time, draft_logprobs,
                 draft_recv_time, draft_send_time);
};

inline SamplingParams sampling_params_from_py_dict(const py::dict &d) {
  SamplingParams p;
  auto assign_if_exists = [&](const char *key, auto &target) {
    if (d.contains(key) && !d[key].is_none())
      target = py::cast<std::decay_t<decltype(target)>>(d[key]);
  };

  assign_if_exists("max_new_tokens", p.max_new_tokens);
  assign_if_exists("stop_strs", p.stop_strs);
  assign_if_exists("stop_token_ids", p.stop_token_ids);
  assign_if_exists("stop_regex_strs", p.stop_regex_strs);
  assign_if_exists("temperature", p.temperature);
  assign_if_exists("top_p", p.top_p);
  assign_if_exists("top_k", p.top_k);
  assign_if_exists("min_p", p.min_p);
  assign_if_exists("frequency_penalty", p.frequency_penalty);
  assign_if_exists("presence_penalty", p.presence_penalty);
  assign_if_exists("repetition_penalty", p.repetition_penalty);
  assign_if_exists("min_new_tokens", p.min_new_tokens);
  assign_if_exists("n", p.n);
  assign_if_exists("json_schema", p.json_schema);
  assign_if_exists("regex", p.regex);
  assign_if_exists("ebnf", p.ebnf);
  assign_if_exists("structural_tag", p.structural_tag);
  assign_if_exists("ignore_eos", p.ignore_eos);
  assign_if_exists("skip_special_tokens", p.skip_special_tokens);
  assign_if_exists("spaces_between_special_tokens",
                   p.spaces_between_special_tokens);
  assign_if_exists("no_stop_trim", p.no_stop_trim);
  assign_if_exists("custom_params", p.custom_params);
  assign_if_exists("stream_interval", p.stream_interval);
  assign_if_exists("logit_bias", p.logit_bias);
  assign_if_exists("sampling_seed", p.sampling_seed);

  return p;
}

template <typename T>
inline void assign_optional_from_py_dict(const py::dict &d, const char *key,
                                         std::optional<T> &target) {
  if (d.contains(key) && !d[key].is_none()) {
    target = py::cast<T>(d[key]);
  }
}

inline SpectreRequest from_py_dict(const py::dict &d) {
  SpectreRequest r;

  assign_optional_from_py_dict(d, "request_id", r.request_id);
  assign_optional_from_py_dict(d, "spec_cnt", r.spec_cnt);
  if (d.contains("action"))
    r.action = str_to_remote_action(py::cast<std::string>(d["action"]));
  if (d.contains("spec_type"))
    r.spec_type = str_to_spec_type(py::cast<std::string>(d["spec_type"]));
  assign_optional_from_py_dict(d, "draft_token_ids", r.draft_token_ids);
  assign_optional_from_py_dict(d, "input_ids", r.input_ids);
  assign_optional_from_py_dict(d, "output_ids", r.output_ids);
  assign_optional_from_py_dict(d, "num_draft_tokens", r.num_draft_tokens);

  if (d.contains("sampling_params") && !d["sampling_params"].is_none()) {
    r.sampling_params =
        sampling_params_from_py_dict(d["sampling_params"].cast<py::dict>());
  }

  if (d.contains("grammar") && !d["grammar"].is_none())
    r.grammar = py::cast<std::string>(d["grammar"]);
  if (d.contains("target_send_time") && !d["target_send_time"].is_none())
    r.target_send_time = py::cast<double>(d["target_send_time"]);
  if (d.contains("target_recv_time") && !d["target_recv_time"].is_none())
    r.target_recv_time = py::cast<double>(d["target_recv_time"]);
  if (d.contains("draft_logprobs") && !d["draft_logprobs"].is_none())
    r.draft_logprobs = py::cast<std::vector<float>>(d["draft_logprobs"]);
  if (d.contains("draft_recv_time") && !d["draft_recv_time"].is_none())
    r.draft_recv_time = py::cast<double>(d["draft_recv_time"]);
  if (d.contains("draft_send_time") && !d["draft_send_time"].is_none())
    r.draft_send_time = py::cast<double>(d["draft_send_time"]);

  return r;
}

inline py::dict to_py_dict(const SpectreRequest &r) {
  py::dict d;

  auto set_if_present = [&](const char *key, const auto &optional_val) {
    if (optional_val)
      d[key] = *optional_val;
  };

  set_if_present("request_id", r.request_id);
  set_if_present("spec_cnt", r.spec_cnt);
  d["action"] = to_string(r.action);
  d["spec_type"] = to_string(r.spec_type);
  set_if_present("draft_token_ids", r.draft_token_ids);
  set_if_present("input_ids", r.input_ids);
  set_if_present("output_ids", r.output_ids);
  set_if_present("num_draft_tokens", r.num_draft_tokens);

  if (r.sampling_params) {
    py::dict p_dict;
    const auto &p = *r.sampling_params;
    p_dict["max_new_tokens"] = p.max_new_tokens;
    p_dict["temperature"] = p.temperature;
    p_dict["top_p"] = p.top_p;
    p_dict["top_k"] = p.top_k;
    p_dict["min_p"] = p.min_p;
    p_dict["frequency_penalty"] = p.frequency_penalty;
    p_dict["presence_penalty"] = p.presence_penalty;
    p_dict["repetition_penalty"] = p.repetition_penalty;
    p_dict["n"] = p.n;
    p_dict["min_new_tokens"] = p.min_new_tokens;
    p_dict["ignore_eos"] = p.ignore_eos;
    p_dict["skip_special_tokens"] = p.skip_special_tokens;
    p_dict["spaces_between_special_tokens"] = p.spaces_between_special_tokens;
    p_dict["no_stop_trim"] = p.no_stop_trim;

    auto set_p_if_present = [&](const char *key, const auto &optional_val) {
      if (optional_val)
        p_dict[key] = *optional_val;
    };

    set_p_if_present("stop_strs", p.stop_strs);
    set_p_if_present("stop_token_ids", p.stop_token_ids);
    set_p_if_present("stop_regex_strs", p.stop_regex_strs);
    set_p_if_present("json_schema", p.json_schema);
    set_p_if_present("regex", p.regex);
    set_p_if_present("ebnf", p.ebnf);
    set_p_if_present("structural_tag", p.structural_tag);
    set_p_if_present("custom_params", p.custom_params);
    set_p_if_present("stream_interval", p.stream_interval);
    set_p_if_present("logit_bias", p.logit_bias);
    set_p_if_present("sampling_seed", p.sampling_seed);

    d["sampling_params"] = std::move(p_dict);
  }

  set_if_present("grammar", r.grammar);
  d["target_send_time"] = r.target_send_time;
  d["target_recv_time"] = r.target_recv_time;
  set_if_present("draft_logprobs", r.draft_logprobs);
  d["draft_recv_time"] = r.draft_recv_time;
  d["draft_send_time"] = r.draft_send_time;

  return d;
}

} // namespace spectre

#endif
