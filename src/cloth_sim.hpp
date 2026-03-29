#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace cloth {

struct ClothSim {
    // Grid
    int gridW = 30, gridH = 30;
    float restDist = 0.25f;

    struct Point {
        float x, y, z;       // current position
        float ox, oy, oz;    // old position (Verlet)
        float nx, ny, nz;    // normal
        bool pinned;
    };

    std::vector<Point> points;
    std::vector<unsigned int> indices; // triangle indices

    // Physics
    float gravity     = -6.0f;
    float damping     = 0.995f;
    float windStrX    = 0.f;
    float windStrZ    = 2.5f;
    float windTurb    = 0.4f;
    int   iterations  = 8;  // constraint solver iterations
    float time        = 0.f;

    // Camera
    float camYaw   = 0.5f;
    float camPitch = 0.3f;
    float camDist  = 8.f;
    bool  autoRotate = false;

    // Display
    int   colorMode = 0; // 0=stress, 1=normals, 2=solid
    bool  wireframe = false;

    std::mt19937 rng{42};

    void init() {
        points.clear();
        indices.clear();
        time = 0.f;

        // Create grid of points
        float startX = -(gridW - 1) * restDist * 0.5f;
        float startY =  (gridH - 1) * restDist * 0.5f + 1.f;

        for (int j = 0; j < gridH; j++) {
            for (int i = 0; i < gridW; i++) {
                Point p;
                p.x = startX + i * restDist;
                p.y = startY - j * restDist;
                p.z = 0.f;
                p.ox = p.x; p.oy = p.y; p.oz = p.z;
                p.nx = 0; p.ny = 0; p.nz = 1;
                // Pin the top row
                p.pinned = (j == 0);
                points.push_back(p);
            }
        }

        // Build triangle indices
        for (int j = 0; j < gridH - 1; j++) {
            for (int i = 0; i < gridW - 1; i++) {
                int tl = j * gridW + i;
                int tr = tl + 1;
                int bl = tl + gridW;
                int br = bl + 1;
                // Two triangles per quad
                indices.push_back(tl); indices.push_back(bl); indices.push_back(tr);
                indices.push_back(tr); indices.push_back(bl); indices.push_back(br);
            }
        }
    }

    void solveConstraint(int a, int b, float restLen) {
        Point& pa = points[a];
        Point& pb = points[b];
        float dx = pb.x - pa.x, dy = pb.y - pa.y, dz = pb.z - pa.z;
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < 1e-6f) return;

        float diff = (dist - restLen) / dist * 0.5f;
        if (!pa.pinned) { pa.x += dx * diff; pa.y += dy * diff; pa.z += dz * diff; }
        if (!pb.pinned) { pb.x -= dx * diff; pb.y -= dy * diff; pb.z -= dz * diff; }
    }

    void computeNormals() {
        // Reset normals
        for (auto& p : points) { p.nx = p.ny = p.nz = 0.f; }

        // Accumulate face normals
        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            auto& a = points[indices[i]];
            auto& b = points[indices[i+1]];
            auto& c = points[indices[i+2]];

            float e1x = b.x-a.x, e1y = b.y-a.y, e1z = b.z-a.z;
            float e2x = c.x-a.x, e2y = c.y-a.y, e2z = c.z-a.z;
            float nx = e1y*e2z - e1z*e2y;
            float ny = e1z*e2x - e1x*e2z;
            float nz = e1x*e2y - e1y*e2x;

            a.nx += nx; a.ny += ny; a.nz += nz;
            b.nx += nx; b.ny += ny; b.nz += nz;
            c.nx += nx; c.ny += ny; c.nz += nz;
        }

        // Normalize
        for (auto& p : points) {
            float len = std::sqrt(p.nx*p.nx + p.ny*p.ny + p.nz*p.nz);
            if (len > 0.001f) { p.nx /= len; p.ny /= len; p.nz /= len; }
        }
    }

    float getStress(int a, int b) const {
        auto& pa = points[a];
        auto& pb = points[b];
        float dx = pb.x-pa.x, dy = pb.y-pa.y, dz = pb.z-pa.z;
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        return std::abs(dist - restDist) / restDist;
    }

    void update(float dt) {
        time += dt;
        dt = std::min(dt, 0.016f); // cap timestep

        if (autoRotate) camYaw += dt * 0.15f;

        std::uniform_real_distribution<float> turb(-windTurb, windTurb);

        // Verlet integration
        for (auto& p : points) {
            if (p.pinned) continue;

            float vx = (p.x - p.ox) * damping;
            float vy = (p.y - p.oy) * damping;
            float vz = (p.z - p.oz) * damping;

            p.ox = p.x; p.oy = p.y; p.oz = p.z;

            // Gravity
            vy += gravity * dt * dt;

            // Wind (with turbulence)
            float wt = std::sin(time * 2.f + p.x * 3.f) * 0.5f + 0.5f;
            vx += (windStrX + turb(rng)) * dt * dt * wt;
            vz += (windStrZ + turb(rng)) * dt * dt * wt;

            p.x += vx; p.y += vy; p.z += vz;
        }

        // Constraint solving (distance constraints along grid edges)
        for (int iter = 0; iter < iterations; iter++) {
            for (int j = 0; j < gridH; j++) {
                for (int i = 0; i < gridW; i++) {
                    int idx = j * gridW + i;
                    // Structural: right neighbor
                    if (i < gridW - 1)
                        solveConstraint(idx, idx + 1, restDist);
                    // Structural: bottom neighbor
                    if (j < gridH - 1)
                        solveConstraint(idx, idx + gridW, restDist);
                    // Shear: diagonal
                    if (i < gridW - 1 && j < gridH - 1)
                        solveConstraint(idx, idx + gridW + 1, restDist * 1.414f);
                    if (i > 0 && j < gridH - 1)
                        solveConstraint(idx, idx + gridW - 1, restDist * 1.414f);
                    // Bend: skip-one
                    if (i < gridW - 2)
                        solveConstraint(idx, idx + 2, restDist * 2.f);
                    if (j < gridH - 2)
                        solveConstraint(idx, idx + gridW * 2, restDist * 2.f);
                }
            }
        }

        computeNormals();
    }

    void reset() {
        init();
    }

    void dropCorner() {
        // Unpin one of the top corners
        if (gridW > 0) {
            points[gridW - 1].pinned = false;
        }
    }

    int totalTris() const { return (int)indices.size() / 3; }
    int totalVerts() const { return (int)points.size(); }
};

} // namespace cloth

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
