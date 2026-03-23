#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "image_utils.hpp"
#include "Snake.hpp"

namespace py = pybind11;

// ── Forward declarations ──────────────────────────────────────────────────────
py::bytes run_canny(const py::bytes& imageBytes, int t_high_percent);

py::bytes detect_lines(const py::bytes& drawBytes, const py::bytes& edgeBytes);
py::bytes detect_circles(const py::bytes& drawBytes, const py::bytes& edgeBytes);
py::bytes detect_ellipses(const py::bytes& drawBytes, const py::bytes& edgeBytes);

// ── Helper: convert Python list[(x,y)] ↔ std::vector<Point> ─────────────────
static std::vector<Point> pyToPoints(const std::vector<std::pair<int,int>> &v) {
    std::vector<Point> out;
    out.reserve(v.size());
    for (auto &p : v) out.push_back({p.first, p.second});
    return out;
}
static std::vector<std::pair<int,int>> pointsToPy(const std::vector<Point> &v) {
    std::vector<std::pair<int,int>> out;
    out.reserve(v.size());
    for (auto &p : v) out.push_back({p.x, p.y});
    return out;
}

// ── Module ────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "CV backend — Canny, Hough transforms, Snake active contour + utilities";

    // ── Canny edge detection ──────────────────────────────────────────────────
    m.def("run_canny", &run_canny,
          py::arg("image_bytes"),
          py::arg("t_high_percent") = 10,
          "Run Canny edge detection. Returns result as PNG bytes.");

    // ── Hough transforms ──────────────────────────────────────────────────────
    m.def("detect_lines", &detect_lines,
          py::arg("draw_bytes"),
          py::arg("edge_bytes"),
          "Detect lines via Hough transform. Returns annotated image as PNG bytes.");

    m.def("detect_circles", &detect_circles,
          py::arg("draw_bytes"),
          py::arg("edge_bytes"),
          "Detect circles via Hough transform. Returns annotated image as PNG bytes.");

    m.def("detect_ellipses", &detect_ellipses,
          py::arg("draw_bytes"),
          py::arg("edge_bytes"),
          "Detect ellipses. Returns annotated image as PNG bytes.");

    // ── Snake API (stateless — pure functions, easy to call from Python) ──────

    // Single evolution step
    // Args: image bytes (PNG), points [(x,y)], alpha, beta, gamma, window_half
    // Returns: new points [(x,y)]
    m.def("snake_evolve_once",
        [](const py::bytes &img_bytes,
           const std::vector<std::pair<int,int>> &pts_in,
           double alpha, double beta, double gamma, int window_half) {

            cv::Mat bgr  = decode_image(img_bytes);
            cv::Mat gray;
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            gray.convertTo(gray, CV_64F);

            Snake s;
            auto pts    = pyToPoints(pts_in);
            auto newPts = s.evolveOnce(gray, pts, alpha, beta, gamma, window_half);
            return pointsToPy(newPts);
        },
        py::arg("image"), py::arg("points"),
        py::arg("alpha")=1.0, py::arg("beta")=1.0, py::arg("gamma")=1.0,
        py::arg("window_half")=2,
        "Run one greedy Snake iteration. Returns updated (x,y) list.");

    // Chain code
    m.def("snake_chain_code",
        [](const std::vector<std::pair<int,int>> &pts_in) {
            Snake s;
            return s.generateChainCode(pyToPoints(pts_in));
        },
        py::arg("points"),
        "Generate 8-direction chain code string from contour points.");

    // Perimeter from chain code
    m.def("snake_perimeter",
        [](const std::string &code) {
            Snake s;
            return s.perimeterFromChainCode(code);
        },
        py::arg("chain_code"),
        "Compute perimeter (in pixels) from chain code.");

    // Area from points + chain code
    m.def("snake_area",
        [](const std::vector<std::pair<int,int>> &pts_in, const std::string &code) {
            Snake s;
            return s.areaFromChainCode(pyToPoints(pts_in), code);
        },
        py::arg("points"), py::arg("chain_code"),
        "Compute enclosed area (pixels²) using Shoelace formula.");

    // Formatted chain code for display
    m.def("snake_format_chain_code",
        [](const std::string &raw, int group_size) {
            Snake s;
            return s.formatChainCode(raw, group_size);
        },
        py::arg("raw_code"), py::arg("group_size")=6,
        "Format chain code with spaces every group_size digits.");
}