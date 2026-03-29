#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace lorenz {

struct LorenzSim {
    // Lorenz parameters
    float sigma = 10.f;
    float rho   = 28.f;
    float beta  = 8.f / 3.f;

    // State
    float x = 1.f, y = 1.f, z = 1.f;
    float time = 0.f;

    // Trail
    struct TrailPoint { float x, y, z; float age; };
    std::vector<TrailPoint> trail;
    int maxTrail = 8000;
    float trailSpeed = 1.0f;

    // Camera
    float camYaw   = 0.5f;
    float camPitch = 0.3f;
    float camDist  = 55.f;
    bool  autoRotate = true;

    // Display
    float lineWidth = 2.f;
    int   colorMode = 0; // 0=speed, 1=age, 2=position
    float stepSize  = 0.005f;
    int   stepsPerFrame = 8;
    bool  paused = false;

    // Second attractor for comparison
    float x2 = 1.001f, y2 = 1.f, z2 = 1.f; // slightly different IC
    std::vector<TrailPoint> trail2;
    bool showSecond = false;

    void init() {
        x = 1.f; y = 1.f; z = 1.f;
        x2 = 1.001f; y2 = 1.f; z2 = 1.f;
        trail.clear();
        trail2.clear();
        time = 0.f;
    }

    void step(float& px, float& py, float& pz, float dt) {
        float dx = sigma * (py - px);
        float dy = px * (rho - pz) - py;
        float dz = px * py - beta * pz;
        px += dx * dt;
        py += dy * dt;
        pz += dz * dt;
    }

    void update(float dt) {
        if (paused) {
            if (autoRotate) camYaw += dt * 0.1f;
            return;
        }

        time += dt;
        if (autoRotate) camYaw += dt * 0.1f;

        for (int i = 0; i < stepsPerFrame; i++) {
            step(x, y, z, stepSize * trailSpeed);
            trail.push_back({x, y, z, 0.f});

            if (showSecond) {
                step(x2, y2, z2, stepSize * trailSpeed);
                trail2.push_back({x2, y2, z2, 0.f});
            }
        }

        // Age trail points
        for (auto& p : trail) p.age += dt;
        for (auto& p : trail2) p.age += dt;

        // Trim trails
        while ((int)trail.size() > maxTrail)
            trail.erase(trail.begin());
        while ((int)trail2.size() > maxTrail)
            trail2.erase(trail2.begin());
    }

    // Get speed at a point for coloring
    float getSpeed(float px, float py, float pz) const {
        float dx = sigma * (py - px);
        float dy = px * (rho - pz) - py;
        float dz = px * py - beta * pz;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    void getColor(const TrailPoint& p, int mode, float maxAge,
                  float& r, float& g, float& b, float& a) const {
        float t;
        switch (mode) {
            case 0: { // Speed
                float spd = getSpeed(p.x, p.y, p.z);
                t = std::clamp(spd / 180.f, 0.f, 1.f);
                r = 0.1f + t * 0.9f;
                g = 0.5f - t * 0.3f + (1.f-t) * 0.5f;
                b = 1.f - t * 0.7f;
                break;
            }
            case 1: { // Age (new=bright, old=dim)
                t = std::clamp(p.age / maxAge, 0.f, 1.f);
                r = 0.f; g = 1.f - t * 0.7f; b = 0.3f + t * 0.3f;
                break;
            }
            case 2: { // Z-position
                t = std::clamp((p.z - 5.f) / 40.f, 0.f, 1.f);
                r = t; g = 0.2f + (1.f-t) * 0.6f; b = 1.f - t;
                break;
            }
            default: r = g = b = 0.8f;
        }
        // Fade oldest points
        a = std::clamp(1.f - p.age / (maxAge * 1.2f), 0.1f, 1.f);
    }

    int trailSize() const { return (int)trail.size(); }
};

} // namespace lorenz

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
