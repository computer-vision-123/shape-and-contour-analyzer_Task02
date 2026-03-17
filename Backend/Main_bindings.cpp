#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "image_utils.hpp"
#include "Snake.hpp"

namespace py = pybind11;

// ── forward declarations (defined in Contour.cpp) ────────────────────────────
int add_numbers(int a, int b);

// ── helper: convert Python list[(x,y)] ↔ std::vector<Point> ─────────────────
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

// ── module ────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "CV backend — Snake active contour + utilities";

    // legacy
    m.def("add_numbers",      &add_numbers,      "Adds two numbers");

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