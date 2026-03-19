#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations from Canny.cpp
py::bytes run_canny(const py::bytes& imageBytes, int t_high_percent);

py::bytes detect_lines(const py::bytes& drawBytes, const py::bytes& edgeBytes);
py::bytes detect_circles(const py::bytes& drawBytes, const py::bytes& edgeBytes);
py::bytes detect_ellipses(const py::bytes& drawBytes, const py::bytes& edgeBytes);

PYBIND11_MODULE(cv_backend, m) {
    m.def("run_canny", &run_canny,
          py::arg("image_bytes"),
          py::arg("t_high_percent") = 10);
          
    m.def("detect_lines", &detect_lines,
          py::arg("draw_bytes"),
          py::arg("edge_bytes"));

    m.def("detect_circles", &detect_circles,
          py::arg("draw_bytes"),
          py::arg("edge_bytes"));

    m.def("detect_ellipses", &detect_ellipses,
          py::arg("draw_bytes"),
          py::arg("edge_bytes"));
}