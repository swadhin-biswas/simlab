#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace pendulum {

struct DoublePendulumSim {
    // Pendulum parameters
    float L1 = 1.5f, L2 = 1.5f;   // arm lengths
    float m1 = 2.0f, m2 = 2.0f;   // masses
    float g  = 9.81f;

    // State: angles and angular velocities
    float a1 = 2.5f, a2 = 2.0f;   // angles from vertical
    float w1 = 0.f,  w2 = 0.f;    // angular velocities

    float time = 0.f;

    // Trail
    struct TrailPt { float x, y; float age; };
    std::vector<TrailPt> trail;
    int maxTrail = 4000;
    bool showTrail = true;

    // Multiple pendulums for chaos comparison
    int numPendulums = 1; // 1, 3, or 5 for comparison
    struct PendState { float a1, a2, w1, w2; };
    std::vector<PendState> extras;
    std::vector<std::vector<TrailPt>> extraTrails;

    // Display
    float trailFade = 0.95f;
    bool  paused = false;
    int   colorMode = 0; // 0=energy, 1=velocity, 2=solid
    float zoom = 1.0f;
    int   subSteps = 8;

    void init() {
        a1 = 2.5f; a2 = 2.0f;
        w1 = 0.f; w2 = 0.f;
        time = 0.f;
        trail.clear();

        extras.clear();
        extraTrails.clear();
        if (numPendulums > 1) {
            for (int i = 1; i < numPendulums; i++) {
                float offset = i * 0.001f;
                extras.push_back({a1 + offset, a2, 0.f, 0.f});
                extraTrails.push_back({});
            }
        }
    }

    // RK4 step for the equations of motion
    struct Derivs { float da1, da2, dw1, dw2; };

    Derivs computeDerivs(float pa1, float pa2, float pw1, float pw2) const {
        float delta = pa1 - pa2;
        float sd = std::sin(delta), cd = std::cos(delta);
        float s1 = std::sin(pa1), s2 = std::sin(pa2);

        float denom = 2*m1 + m2 - m2*cd*cd;

        float dw1 = (-g*(2*m1+m2)*s1 - m2*g*std::sin(pa1-2*pa2)
                     - 2*sd*m2*(pw2*pw2*L2 + pw1*pw1*L1*cd))
                    / (L1 * denom);

        float dw2 = (2*sd*(pw1*pw1*L1*(m1+m2) + g*(m1+m2)*std::cos(pa1)
                     + pw2*pw2*L2*m2*cd))
                    / (L2 * denom);

        return {pw1, pw2, dw1, dw2};
    }

    void stepRK4(float& pa1, float& pa2, float& pw1, float& pw2, float dt) {
        auto k1 = computeDerivs(pa1, pa2, pw1, pw2);
        auto k2 = computeDerivs(pa1+k1.da1*dt/2, pa2+k1.da2*dt/2,
                                pw1+k1.dw1*dt/2, pw2+k1.dw2*dt/2);
        auto k3 = computeDerivs(pa1+k2.da1*dt/2, pa2+k2.da2*dt/2,
                                pw1+k2.dw1*dt/2, pw2+k2.dw2*dt/2);
        auto k4 = computeDerivs(pa1+k3.da1*dt, pa2+k3.da2*dt,
                                pw1+k3.dw1*dt, pw2+k3.dw2*dt);

        pa1 += (k1.da1 + 2*k2.da1 + 2*k3.da1 + k4.da1) * dt / 6;
        pa2 += (k1.da2 + 2*k2.da2 + 2*k3.da2 + k4.da2) * dt / 6;
        pw1 += (k1.dw1 + 2*k2.dw1 + 2*k3.dw1 + k4.dw1) * dt / 6;
        pw2 += (k1.dw2 + 2*k2.dw2 + 2*k3.dw2 + k4.dw2) * dt / 6;
    }

    void update(float dt) {
        if (paused) return;
        time += dt;
        dt = std::min(dt, 0.02f);
        float subDt = dt / subSteps;

        for (int s = 0; s < subSteps; s++) {
            stepRK4(a1, a2, w1, w2, subDt);

            for (auto& e : extras)
                stepRK4(e.a1, e.a2, e.w1, e.w2, subDt);
        }

        // Second bob position for trail
        float x1 = L1 * std::sin(a1);
        float y1 = -L1 * std::cos(a1);
        float x2 = x1 + L2 * std::sin(a2);
        float y2 = y1 - L2 * std::cos(a2);
        trail.push_back({x2, y2, 0.f});

        for (size_t i = 0; i < extras.size(); i++) {
            auto& e = extras[i];
            float ex1 = L1 * std::sin(e.a1);
            float ey1 = -L1 * std::cos(e.a1);
            float ex2 = ex1 + L2 * std::sin(e.a2);
            float ey2 = ey1 - L2 * std::cos(e.a2);
            extraTrails[i].push_back({ex2, ey2, 0.f});
        }

        // Age and trim
        for (auto& p : trail) p.age += dt;
        while ((int)trail.size() > maxTrail) trail.erase(trail.begin());

        for (auto& et : extraTrails) {
            for (auto& p : et) p.age += dt;
            while ((int)et.size() > maxTrail) et.erase(et.begin());
        }
    }

    // Kinetic + potential energy
    float totalEnergy() const {
        float c12 = std::cos(a1 - a2);
        float KE = 0.5f*m1*L1*L1*w1*w1
                 + 0.5f*m2*(L1*L1*w1*w1 + L2*L2*w2*w2 + 2*L1*L2*w1*w2*c12);
        float PE = -(m1+m2)*g*L1*std::cos(a1) - m2*g*L2*std::cos(a2);
        return KE + PE;
    }

    void reset() { init(); }
};

} // namespace pendulum

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
