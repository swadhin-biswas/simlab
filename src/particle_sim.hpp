#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace particle {

struct Particle {
    float x, y, z;       // position
    float vx, vy, vz;    // velocity
    float life;          // remaining life [0,1]
    float maxLife;       // total life
    float size;
    float r, g, b;       // base color
};

struct ParticleSim {
    std::vector<Particle> particles;
    int maxParticles     = 3000;
    float emitRate       = 400.f;  // particles per second
    float emitAccum      = 0.f;
    float gravity        = -2.8f;
    float windX          = 0.f;
    float windZ          = 0.f;
    float spread         = 0.6f;
    float initialSpeed   = 3.5f;
    float particleSize   = 6.f;
    float time           = 0.f;
    int   emitterMode    = 0; // 0=fountain, 1=fire, 2=explosion, 3=vortex

    // Camera orbit
    float camYaw   = 0.4f;
    float camPitch = 0.3f;
    float camDist  = 8.f;
    bool  autoRotate = true;

    std::mt19937 rng{123};

    void init() {
        particles.clear();
        particles.reserve(maxParticles);
        emitAccum = 0.f;
        time = 0.f;
    }

    void emit(int count) {
        std::uniform_real_distribution<float> d01(0.f, 1.f);
        std::uniform_real_distribution<float> ds(-1.f, 1.f);

        for (int i = 0; i < count && (int)particles.size() < maxParticles; i++) {
            Particle p;

            switch (emitterMode) {
                case 0: { // Fountain
                    float angle = d01(rng) * 6.2831853f;
                    float r = d01(rng) * spread * 0.3f;
                    p.x = r * std::cos(angle);
                    p.y = 0.f;
                    p.z = r * std::sin(angle);
                    float spd = initialSpeed * (0.8f + d01(rng) * 0.4f);
                    p.vx = ds(rng) * spread;
                    p.vy = spd;
                    p.vz = ds(rng) * spread;
                    p.r = 0.2f; p.g = 0.5f; p.b = 1.0f;
                    p.maxLife = 2.0f + d01(rng) * 1.5f;
                    break;
                }
                case 1: { // Fire
                    float angle = d01(rng) * 6.2831853f;
                    float r = d01(rng) * 0.4f;
                    p.x = r * std::cos(angle);
                    p.y = -0.5f;
                    p.z = r * std::sin(angle);
                    p.vx = ds(rng) * 0.3f;
                    p.vy = initialSpeed * 0.6f * (0.7f + d01(rng) * 0.6f);
                    p.vz = ds(rng) * 0.3f;
                    p.r = 1.0f; p.g = 0.6f; p.b = 0.1f;
                    p.maxLife = 1.0f + d01(rng) * 0.8f;
                    break;
                }
                case 2: { // Explosion
                    float phi = d01(rng) * 6.2831853f;
                    float theta = std::acos(ds(rng));
                    float spd = initialSpeed * (0.5f + d01(rng) * 1.5f);
                    p.x = 0; p.y = 0; p.z = 0;
                    p.vx = spd * std::sin(theta) * std::cos(phi);
                    p.vy = spd * std::sin(theta) * std::sin(phi);
                    p.vz = spd * std::cos(theta);
                    p.r = 1.0f; p.g = 0.3f + d01(rng)*0.4f; p.b = 0.1f;
                    p.maxLife = 1.5f + d01(rng) * 1.0f;
                    break;
                }
                case 3: { // Vortex
                    float angle = time * 2.f + d01(rng) * 1.5f;
                    float r = 0.5f + d01(rng) * 0.3f;
                    p.x = r * std::cos(angle);
                    p.y = -1.f;
                    p.z = r * std::sin(angle);
                    float spd = initialSpeed * 0.7f;
                    p.vx = -std::sin(angle) * spd + ds(rng) * 0.2f;
                    p.vy = spd * 0.9f;
                    p.vz = std::cos(angle) * spd + ds(rng) * 0.2f;
                    p.r = 0.f; p.g = 1.f; p.b = 0.5f;
                    p.maxLife = 2.5f + d01(rng) * 1.0f;
                    break;
                }
            }

            p.life = p.maxLife;
            p.size = particleSize * (0.5f + d01(rng) * 0.5f);
            particles.push_back(p);
        }
    }

    void update(float dt) {
        time += dt;

        if (autoRotate) camYaw += dt * 0.15f;

        // Emit new particles
        emitAccum += emitRate * dt;
        int toEmit = (int)emitAccum;
        if (toEmit > 0) {
            emit(toEmit);
            emitAccum -= toEmit;
        }

        // Update existing particles
        for (auto& p : particles) {
            p.life -= dt;

            // Apply forces
            p.vy += gravity * dt;
            p.vx += windX * dt;
            p.vz += windZ * dt;

            // Vortex mode: swirl force
            if (emitterMode == 3) {
                float dist = std::sqrt(p.x*p.x + p.z*p.z);
                if (dist > 0.01f) {
                    float force = 1.5f / (dist + 0.5f);
                    p.vx += (-p.z / dist) * force * dt;
                    p.vz += ( p.x / dist) * force * dt;
                }
            }

            // Integrate
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.z += p.vz * dt;
        }

        // Remove dead particles
        particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                [](const Particle& p) { return p.life <= 0.f; }),
            particles.end()
        );
    }

    // Get color with lifecycle fade
    void getColor(const Particle& p, float& r, float& g, float& b, float& a) const {
        float t = 1.f - (p.life / p.maxLife); // 0=born, 1=dead

        if (emitterMode == 1) { // Fire: yellow → orange → red → dark
            r = p.r * (1.f - t * 0.3f);
            g = p.g * (1.f - t * 0.8f);
            b = p.b * (1.f - t);
        } else if (emitterMode == 2) { // Explosion: bright → dark
            r = p.r * (1.f - t * 0.5f);
            g = p.g * (1.f - t * 0.6f);
            b = p.b + t * 0.3f;
        } else { // Fountain/Vortex
            r = p.r + t * 0.3f;
            g = p.g + t * 0.2f;
            b = p.b * (1.f - t * 0.5f);
        }
        a = std::clamp(p.life / p.maxLife, 0.f, 1.f);
        // Fade in at birth
        float birth = 1.f - t;
        if (birth > 0.9f) a *= (1.f - birth) * 10.f;
    }

    int liveCount() const { return (int)particles.size(); }
};

} // namespace particle

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
