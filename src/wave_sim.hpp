#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace wave {

struct WaveSim {
    int gridSize = 80;
    std::vector<float> height;    // current
    std::vector<float> prevH;     // previous (for Verlet)
    std::vector<float> velocity;

    float damping   = 0.995f;
    float speed     = 1.8f;       // wave propagation speed
    float time      = 0.f;

    // Drop parameters
    int   dropMode  = 0; // 0=single drop, 1=rain, 2=dual slit, 3=ripple
    float dropForce = 3.f;
    float rainRate  = 3.f; // drops per second
    float rainAccum = 0.f;

    // Camera
    float camYaw    = 0.6f;
    float camPitch  = 0.5f;
    float camDist   = 14.f;
    bool  autoRotate = false;

    // Display
    int   colorMode = 0; // 0=height, 1=velocity, 2=gradient
    bool  wireframe = false;
    float heightScale = 1.0f;

    std::vector<unsigned int> indices;

    std::mt19937 rng{99};

    void init() {
        int n = gridSize;
        height.assign(n * n, 0.f);
        prevH.assign(n * n, 0.f);
        velocity.assign(n * n, 0.f);
        time = 0.f;

        // Build mesh indices
        indices.clear();
        for (int j = 0; j < n-1; j++) {
            for (int i = 0; i < n-1; i++) {
                int tl = j*n + i, tr = tl+1;
                int bl = tl+n,    br = bl+1;
                indices.push_back(tl); indices.push_back(bl); indices.push_back(tr);
                indices.push_back(tr); indices.push_back(bl); indices.push_back(br);
            }
        }
    }

    void drop(int cx, int cy, float force) {
        int r = 2;
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                int px = cx + dx, py = cy + dy;
                if (px >= 0 && px < gridSize && py >= 0 && py < gridSize) {
                    float d = std::sqrt((float)(dx*dx + dy*dy));
                    if (d <= r) {
                        float falloff = 1.f - d / (r + 0.5f);
                        height[py * gridSize + px] += force * falloff;
                    }
                }
            }
        }
    }

    void update(float dt) {
        time += dt;
        dt = std::min(dt, 0.02f);
        if (autoRotate) camYaw += dt * 0.1f;

        int n = gridSize;
        float c = speed * speed;

        // Source modes
        std::uniform_int_distribution<int> rpos(5, n-6);
        switch (dropMode) {
            case 1: { // Rain
                rainAccum += rainRate * dt;
                int drops = (int)rainAccum;
                rainAccum -= drops;
                for (int i = 0; i < drops; i++)
                    drop(rpos(rng), rpos(rng), dropForce * 0.5f);
                break;
            }
            case 2: { // Dual slit - oscillating source on left edge
                float freq = 4.f;
                int slit1 = n/3, slit2 = 2*n/3;
                float val = std::sin(time * freq * 6.28f) * dropForce * 0.3f;
                for (int dy = -1; dy <= 1; dy++) {
                    height[(slit1+dy)*n + 2] = val;
                    height[(slit2+dy)*n + 2] = val;
                }
                break;
            }
            case 3: { // Center ripple
                float freq = 3.f;
                float val = std::sin(time * freq * 6.28f) * dropForce * 0.2f;
                height[(n/2)*n + n/2] = val;
                break;
            }
        }

        // Wave equation: d²h/dt² = c² * ∇²h
        std::vector<float> newH(n * n);
        for (int j = 1; j < n-1; j++) {
            for (int i = 1; i < n-1; i++) {
                int idx = j * n + i;
                float laplacian = height[idx-1] + height[idx+1]
                                + height[idx-n] + height[idx+n]
                                - 4.f * height[idx];
                velocity[idx] += c * laplacian * dt;
                velocity[idx] *= damping;
                newH[idx] = height[idx] + velocity[idx] * dt;
            }
        }

        prevH = height;
        height = newH;
    }

    void getNormal(int i, int j, float& nx, float& ny, float& nz) const {
        int n = gridSize;
        float hL = (i > 0)   ? height[j*n + i-1] : height[j*n + i];
        float hR = (i < n-1) ? height[j*n + i+1] : height[j*n + i];
        float hD = (j > 0)   ? height[(j-1)*n + i] : height[j*n + i];
        float hU = (j < n-1) ? height[(j+1)*n + i] : height[j*n + i];
        float scale = heightScale * 2.f;
        nx = (hL - hR) * scale;
        ny = 2.f;
        nz = (hD - hU) * scale;
        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len > 0.001f) { nx /= len; ny /= len; nz /= len; }
    }

    void reset() {
        height.assign(gridSize * gridSize, 0.f);
        prevH.assign(gridSize * gridSize, 0.f);
        velocity.assign(gridSize * gridSize, 0.f);
        time = 0.f;
    }

    void singleDrop() {
        std::uniform_int_distribution<int> rp(10, gridSize-11);
        drop(rp(rng), rp(rng), dropForce);
    }

    int totalTris() const { return (int)indices.size() / 3; }
};

} // namespace wave

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
