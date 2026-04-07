#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <msgpack.hpp>
#include <pybind11/pybind11.h>

#include "spectre_protocol.hpp"

std::string
pack_spectre_batch_payload(const std::vector<spectre::SpectreRequest> &objs);

bool unpack_spectre_batch_payload(msgpack::unpacker &unpacker, const void *data,
                                  size_t len,
                                  std::vector<spectre::SpectreRequest> &objs);

std::vector<spectre::SpectreRequest> from_py_list(const pybind11::list &objs);
