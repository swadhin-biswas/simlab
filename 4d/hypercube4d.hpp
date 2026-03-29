#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace hd {

// ---------------------------------------------------------------------------
// 4D vector (separate from glm::vec4 to avoid confusion with homogeneous coords)
// ---------------------------------------------------------------------------
struct Vec4 {
    float x, y, z, w;
    Vec4 operator+(const Vec4& o) const { return {x+o.x, y+o.y, z+o.z, w+o.w}; }
    Vec4 operator*(float s)       const { return {x*s,   y*s,   z*s,   w*s};   }
};

// ---------------------------------------------------------------------------
// 4×4 matrix for 4D linear transforms (NOT the same as an OpenGL MVP matrix)
// ---------------------------------------------------------------------------
struct Mat4 {
    float m[4][4] = {};

    static Mat4 identity() {
        Mat4 r; r.m[0][0]=r.m[1][1]=r.m[2][2]=r.m[3][3]=1.f; return r;
    }

    // Build a rotation matrix in the ij-plane (0=X,1=Y,2=Z,3=W)
    static Mat4 rotation(int i, int j, float angle) {
        Mat4 r = identity();
        float c = std::cos(angle), s = std::sin(angle);
        r.m[i][i] =  c;  r.m[i][j] = -s;
        r.m[j][i] =  s;  r.m[j][j] =  c;
        return r;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) {
            r.m[i][j] = 0;
            for (int k=0;k<4;k++) r.m[i][j] += m[i][k]*o.m[k][j];
        }
        return r;
    }

    Vec4 operator*(const Vec4& v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
            m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w,
        };
    }
};

// ---------------------------------------------------------------------------
// 6 rotation planes in 4D: every pair of axes
// ---------------------------------------------------------------------------
enum class Plane : int { XY=0, XZ=1, XW=2, YZ=3, YW=4, ZW=5 };
constexpr int NUM_PLANES = 6;
// axis pairs for each plane
constexpr int PLANE_I[6] = { 0,0,0,1,1,2 };
constexpr int PLANE_J[6] = { 1,2,3,2,3,3 };
constexpr const char* PLANE_NAME[6] = { "XY","XZ","XW","YZ","YW","ZW" };

// ---------------------------------------------------------------------------
// Stateful 4D rotation — accumulates angles per plane, exports combined matrix
// ---------------------------------------------------------------------------
struct Rotation4D {
    float angles[NUM_PLANES] = {};
    float speeds[NUM_PLANES] = { 0.0f, 0.0f, 0.41f, 0.0f, 0.29f, 0.23f };
    bool  active[NUM_PLANES] = { false, false, true, false, true, true };

    void update(float dt) {
        for (int p=0;p<NUM_PLANES;p++)
            if (active[p]) angles[p] += dt * speeds[p];
    }

    Mat4 matrix() const {
        Mat4 r = Mat4::identity();
        for (int p=0;p<NUM_PLANES;p++)
            if (active[p])
                r = Mat4::rotation(PLANE_I[p], PLANE_J[p], angles[p]) * r;
        return r;
    }
};

// ---------------------------------------------------------------------------
// Projection: 4D → 3D via perspective along w-axis
//   scale = 1 / (w_dist - w)   (just like standard 3D perspective but for w)
// ---------------------------------------------------------------------------
inline glm::vec4 project4Dto3D(const Vec4& v, float w_dist = 2.0f) {
    float s = 1.0f / (w_dist - v.w);
    return { v.x * s, v.y * s, v.z * s, v.w }; // .w preserved for color mapping
}

// ---------------------------------------------------------------------------
// Dataset: N points in 4D with class labels and per-point importance weight
// ---------------------------------------------------------------------------
struct Dataset4D {
    std::vector<Vec4>  points;
    std::vector<int>   labels;   // class index for color per category
    std::vector<float> weights;  // importance → point size hint

    int numClasses() const;
    void normalize(); // scale all points to fit inside unit 4D hypersphere

    // --- Built-in generators ---
    static Dataset4D tesseract();       // 16 vertices of a unit hypercube
    static Dataset4D kleinBottle();     // parametric Klein bottle in 4D
    static Dataset4D hopfFibration(int n = 800); // beautiful S3 structure

    // --- Load from file ---
    // CSV: specify which column indices map to x,y,z,w,label
    static Dataset4D loadCSV(
        const std::string& path,
        int xi, int yi, int zi, int wi,
        int label_col = -1  // -1 = no labels
    );

    // Pre-reduced: load a file where each row is already 4 floats [+ label]
    static Dataset4D load4D(const std::string& path);
};

} // namespace hd

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
