#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace boids {

struct Boid {
    float x, y, z;
    float vx, vy, vz;
    int   group; // for coloring
};

struct BoidsSim {
    std::vector<Boid> boids;
    int   numBoids     = 300;
    float separation   = 1.5f;
    float alignment    = 1.0f;
    float cohesion     = 1.0f;
    float perception   = 2.0f;   // neighbor radius
    float maxSpeed     = 3.0f;
    float maxForce     = 5.0f;
    float boundSize    = 5.0f;   // soft boundary
    float boidSize     = 5.f;
    float time         = 0.f;

    // Predator (optional obstacle)
    bool  predatorOn   = false;
    float predX = 0, predY = 0, predZ = 0;
    float predRadius   = 3.0f;

    // Camera
    float camYaw   = 0.4f;
    float camPitch = 0.25f;
    float camDist  = 14.f;
    bool  autoRotate = true;

    // Display
    int groups = 3;

    std::mt19937 rng{77};

    void init() {
        boids.clear();
        std::uniform_real_distribution<float> pos(-boundSize * 0.5f, boundSize * 0.5f);
        std::uniform_real_distribution<float> vel(-1.f, 1.f);

        for (int i = 0; i < numBoids; i++) {
            Boid b;
            b.x = pos(rng); b.y = pos(rng); b.z = pos(rng);
            b.vx = vel(rng); b.vy = vel(rng); b.vz = vel(rng);
            b.group = i % groups;
            boids.push_back(b);
        }
        time = 0.f;
    }

    static float dist3(const Boid& a, const Boid& b) {
        float dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    static float len3(float x, float y, float z) {
        return std::sqrt(x*x + y*y + z*z);
    }

    void limit(float& vx, float& vy, float& vz, float maxVal) {
        float l = len3(vx, vy, vz);
        if (l > maxVal && l > 0.001f) {
            float s = maxVal / l;
            vx *= s; vy *= s; vz *= s;
        }
    }

    void update(float dt) {
        time += dt;
        if (autoRotate) camYaw += dt * 0.12f;

        // Animate predator
        if (predatorOn) {
            predX = std::sin(time * 0.5f) * boundSize * 0.4f;
            predY = std::cos(time * 0.7f) * boundSize * 0.3f;
            predZ = std::sin(time * 0.3f + 1.f) * boundSize * 0.4f;
        }

        // For each boid, compute steering forces
        struct Force { float fx, fy, fz; };
        std::vector<Force> forces(boids.size(), {0,0,0});

        for (size_t i = 0; i < boids.size(); i++) {
            auto& bi = boids[i];
            float sepX=0, sepY=0, sepZ=0;
            float aliX=0, aliY=0, aliZ=0;
            float cohX=0, cohY=0, cohZ=0;
            int neighbors = 0;

            for (size_t j = 0; j < boids.size(); j++) {
                if (i == j) continue;
                float d = dist3(bi, boids[j]);
                if (d < perception && d > 0.001f) {
                    neighbors++;

                    // Separation: steer away from nearby boids
                    float dx = bi.x - boids[j].x;
                    float dy = bi.y - boids[j].y;
                    float dz = bi.z - boids[j].z;
                    float invD = 1.f / (d * d);
                    sepX += dx * invD;
                    sepY += dy * invD;
                    sepZ += dz * invD;

                    // Alignment: match velocity of neighbors
                    aliX += boids[j].vx;
                    aliY += boids[j].vy;
                    aliZ += boids[j].vz;

                    // Cohesion: steer towards center of mass
                    cohX += boids[j].x;
                    cohY += boids[j].y;
                    cohZ += boids[j].z;
                }
            }

            float fx = 0, fy = 0, fz = 0;

            if (neighbors > 0) {
                // Separation
                fx += sepX * separation;
                fy += sepY * separation;
                fz += sepZ * separation;

                // Alignment
                aliX /= neighbors; aliY /= neighbors; aliZ /= neighbors;
                fx += (aliX - bi.vx) * alignment;
                fy += (aliY - bi.vy) * alignment;
                fz += (aliZ - bi.vz) * alignment;

                // Cohesion
                cohX /= neighbors; cohY /= neighbors; cohZ /= neighbors;
                fx += (cohX - bi.x) * cohesion;
                fy += (cohY - bi.y) * cohesion;
                fz += (cohZ - bi.z) * cohesion;
            }

            // Boundary avoidance (soft walls)
            float margin = 1.0f;
            float turnForce = 3.0f;
            if (bi.x > boundSize)  fx -= turnForce * (bi.x - boundSize);
            if (bi.x < -boundSize) fx -= turnForce * (bi.x + boundSize);
            if (bi.y > boundSize)  fy -= turnForce * (bi.y - boundSize);
            if (bi.y < -boundSize) fy -= turnForce * (bi.y + boundSize);
            if (bi.z > boundSize)  fz -= turnForce * (bi.z - boundSize);
            if (bi.z < -boundSize) fz -= turnForce * (bi.z + boundSize);

            // Predator avoidance
            if (predatorOn) {
                float dx = bi.x - predX, dy = bi.y - predY, dz = bi.z - predZ;
                float d = len3(dx, dy, dz);
                if (d < predRadius && d > 0.01f) {
                    float force = 15.f / (d * d);
                    fx += dx / d * force;
                    fy += dy / d * force;
                    fz += dz / d * force;
                }
            }

            // Limit force
            limit(fx, fy, fz, maxForce);
            forces[i] = {fx, fy, fz};
        }

        // Apply forces and integrate
        for (size_t i = 0; i < boids.size(); i++) {
            auto& b = boids[i];
            b.vx += forces[i].fx * dt;
            b.vy += forces[i].fy * dt;
            b.vz += forces[i].fz * dt;
            limit(b.vx, b.vy, b.vz, maxSpeed);

            b.x += b.vx * dt;
            b.y += b.vy * dt;
            b.z += b.vz * dt;
        }
    }

    int liveCount() const { return (int)boids.size(); }
};

} // namespace boids

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
