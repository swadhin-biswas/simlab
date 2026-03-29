#include "hypercube4d.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hd {

int Dataset4D::numClasses() const {
    if (labels.empty()) return 0;
    return *std::max_element(labels.begin(), labels.end()) + 1;
}

void Dataset4D::normalize() {
    if (points.empty()) return;
    Vec4 centroid = {0,0,0,0};
    for (auto& p : points)
        centroid = centroid + p * (1.f / points.size());
    float maxR = 0.f;
    for (auto& p : points) {
        p.x -= centroid.x; p.y -= centroid.y;
        p.z -= centroid.z; p.w -= centroid.w;
        float r = p.x*p.x + p.y*p.y + p.z*p.z + p.w*p.w;
        maxR = std::max(maxR, r);
    }
    float inv = maxR > 0 ? 1.f / std::sqrt(maxR) : 1.f;
    for (auto& p : points) { p.x*=inv; p.y*=inv; p.z*=inv; p.w*=inv; }
}

Dataset4D Dataset4D::tesseract() {
    Dataset4D d;
    for (int i = 0; i < 16; i++) {
        d.points.push_back({
            (i&1) ? 1.f : -1.f,
            (i&2) ? 1.f : -1.f,
            (i&4) ? 1.f : -1.f,
            (i&8) ? 1.f : -1.f
        });
        d.labels.push_back((i & 8) ? 1 : 0);
        d.weights.push_back(1.f);
    }
    return d;
}

Dataset4D Dataset4D::kleinBottle() {
    Dataset4D d;
    constexpr int U = 60, V = 40;
    constexpr float PI = 3.14159265f;
    for (int ui = 0; ui < U; ui++) {
        for (int vi = 0; vi < V; vi++) {
            float u = 2*PI * ui / U;
            float v = 2*PI * vi / V;
            d.points.push_back({
                (2.f + std::cos(u/2)*std::sin(v) - std::sin(u/2)*std::sin(2*v)) * std::cos(u),
                (2.f + std::cos(u/2)*std::sin(v) - std::sin(u/2)*std::sin(2*v)) * std::sin(u),
                std::sin(u/2)*std::sin(v) + std::cos(u/2)*std::sin(2*v),
                std::cos(v) + std::sin(v)*std::cos(u/2)
            });
            d.labels.push_back(ui % 4);
            d.weights.push_back(1.f);
        }
    }
    d.normalize();
    return d;
}

Dataset4D Dataset4D::hopfFibration(int n) {
    Dataset4D d;
    constexpr float PI = 3.14159265f;
    for (int i = 0; i < n; i++) {
        float t    = (float)i / n;
        float phi  = std::acos(1.f - 2.f*t);
        float theta= 2*PI * i * 1.6180339887f;
        float fib_t = 2*PI * (i % 40) / 40.f;
        float sphi = std::sin(phi/2), cphi = std::cos(phi/2);
        d.points.push_back({
            sphi * std::cos(theta + fib_t),
            sphi * std::sin(theta + fib_t),
            cphi * std::cos(fib_t),
            cphi * std::sin(fib_t)
        });
        d.labels.push_back(i % 8);
        d.weights.push_back(0.5f + 0.5f * cphi);
    }
    d.normalize();
    return d;
}

Dataset4D Dataset4D::loadCSV(
    const std::string& path,
    int xi, int yi, int zi, int wi, int label_col)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open: " + path);
    Dataset4D d;
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::vector<float> cols;
        std::stringstream  ss(line);
        std::string        tok;
        while (std::getline(ss, tok, ','))
            cols.push_back(std::stof(tok));
        int maxIdx = std::max({xi,yi,zi,wi, label_col});
        if ((int)cols.size() <= maxIdx) continue;
        d.points.push_back({cols[xi], cols[yi], cols[zi], cols[wi]});
        d.labels.push_back(label_col >= 0 ? (int)cols[label_col] : 0);
        d.weights.push_back(1.f);
    }
    d.normalize();
    return d;
}

Dataset4D Dataset4D::load4D(const std::string& path) {
    return loadCSV(path, 0, 1, 2, 3, 4);
}

} // namespace hd

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
