#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
py::bytes run_canny(const py::bytes& imageBytes, int t_high_percent);

PYBIND11_MODULE(cv_backend, m) {
    m.def("run_canny", &run_canny,
          py::arg("image_bytes"),
          py::arg("t_high_percent") = 10);
}