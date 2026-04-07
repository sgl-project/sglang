#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "spectre_protocol.hpp"
#include "spectre_zmq_endpoints.hpp"
#include "spectre_zmq_logging.hpp"
#include "spectre_zmq_serialization.hpp"

namespace py = pybind11;

PYBIND11_MODULE(spectre_zmq, m) {
  m.def("set_spectre_log_level", &spectre_set_log_level);

  py::class_<DealerEndpoint>(m, "DealerEndpoint")
      .def(py::init<const std::string &, const std::string &, bool>())
      .def("start", &DealerEndpoint::start)
      .def("stop", &DealerEndpoint::stop)
      .def("send_objs",
           [](DealerEndpoint &self, py::list objs) {
             auto reqs = from_py_list(objs);
             {
               py::gil_scoped_release release;
               self.send_objs(std::move(reqs));
             }
           })
      .def("get_received_objs", [](DealerEndpoint &self) {
        py::list out;
        auto data = [&self]() {
          py::gil_scoped_release release;
          return self.get_received_objs();
        }();
        for (auto &obj : data) {
          out.append(spectre::to_py_dict(obj));
        }
        return out;
      });

  py::class_<RouterEndpoint>(m, "RouterEndpoint")
      .def(py::init<const std::string &, bool>())
      .def("start", &RouterEndpoint::start)
      .def("stop", &RouterEndpoint::stop)
      .def("get_all_dealers", &RouterEndpoint::get_all_dealers)
      .def("send_objs",
           [](RouterEndpoint &self, const std::string &id, py::list objs) {
             auto reqs = from_py_list(objs);
             {
               py::gil_scoped_release release;
               self.send_objs(id, std::move(reqs));
             }
           })
      .def("get_received_objs", [](RouterEndpoint &self) {
        py::list out;
        auto data = [&self]() {
          py::gil_scoped_release release;
          return self.get_received_objs();
        }();
        for (auto &p : data) {
          out.append(py::make_tuple(p.first, spectre::to_py_dict(p.second)));
        }
        return out;
      });
}
