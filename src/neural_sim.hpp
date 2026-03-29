#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <random>

namespace neural {

// ---------------------------------------------------------------------------
// Neural network simulation state — ported from neural_viz.html
// ---------------------------------------------------------------------------
struct NeuralSim {
    // Architecture
    static constexpr int NUM_LAYERS = 4;
    int arch[NUM_LAYERS] = {16, 12, 8, 10};
    const char* layerNames[NUM_LAYERS] = {"INPUT", "HIDDEN_1", "HIDDEN_2", "OUTPUT"};

    // Network state
    std::vector<std::vector<float>> weights;   // weights[l][i*next + j]
    std::vector<std::vector<float>> biases;
    std::vector<std::vector<float>> activations;
    std::vector<std::vector<float>> animActivations;

    // 3D node positions for visualization
    struct Node3D {
        float x, y, z;
        int layer, index;
    };
    std::vector<Node3D> nodes;

    struct Edge {
        int fromNode, toNode;
        int layer, fromIdx, toIdx;
        float weight;
    };
    std::vector<Edge> edges;

    // Pulse animation
    struct Pulse {
        int edgeIdx;
        float progress;
        float brightness;
        float speed;
        int layer;
    };
    std::vector<Pulse> pulses;

    // Camera
    float camYaw   = -0.38f;
    float camPitch = 0.12f;
    float camDist  = 5.5f;

    // State
    bool  autoRotate = true;
    bool  autoInfer  = true;
    float autoTimer  = 0.f;
    bool  inferActive = false;
    int   inferPhase  = -1;
    float inferTimer  = 0.f;
    float wScale = 1.f;
    float noise  = 0.15f;
    float pulseSpeed = 0.9f;
    int   activationFn = 0; // 0=relu, 1=sigmoid, 2=tanh
    int   prediction = -1;
    float confidence = 0.f;
    float inferTime  = 0.f;
    float time = 0.f;

    std::mt19937 rng{42};

    void init() {
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::uniform_real_distribution<float> small(-0.15f, 0.15f);

        weights.clear();
        biases.clear();
        activations.resize(NUM_LAYERS);
        animActivations.resize(NUM_LAYERS);

        for (int l = 0; l < NUM_LAYERS; l++) {
            activations[l].resize(arch[l], 0.f);
            animActivations[l].resize(arch[l], 0.f);
            for (int i = 0; i < arch[l]; i++)
                activations[l][i] = small(rng) + 0.15f;

            if (l < NUM_LAYERS - 1) {
                int sz = arch[l] * arch[l+1];
                std::vector<float> W(sz);
                for (auto& w : W) w = dist(rng) * 1.8f;
                weights.push_back(W);

                std::vector<float> b(arch[l+1]);
                for (auto& bi : b) bi = small(rng);
                biases.push_back(b);
            }
        }

        buildNodes();
        buildEdges();
    }

    void buildNodes() {
        nodes.clear();
        float lz[4] = {-2.4f, -0.8f, 0.8f, 2.4f};
        for (int l = 0; l < NUM_LAYERS; l++) {
            int n = arch[l];
            int cols = (int)std::ceil(std::sqrt((float)n));
            int rows = (int)std::ceil((float)n / cols);
            for (int i = 0; i < n; i++) {
                int c = i % cols, r = i / cols;
                nodes.push_back({
                    (c - (cols-1)/2.f) * 0.62f,
                    ((rows-1)/2.f - r) * 0.62f,
                    lz[l],
                    l, i
                });
            }
        }
    }

    void buildEdges() {
        edges.clear();
        int off = 0;
        for (int l = 0; l < NUM_LAYERS - 1; l++) {
            for (int i = 0; i < arch[l]; i++) {
                for (int j = 0; j < arch[l+1]; j++) {
                    edges.push_back({
                        off + i,
                        off + arch[l] + j,
                        l, i, j,
                        weights[l][i * arch[l+1] + j]
                    });
                }
            }
            off += arch[l];
        }
    }

    float activate(float x) const {
        switch (activationFn) {
            case 0: return std::max(0.f, x);              // ReLU
            case 1: return 1.f / (1.f + std::exp(-x));    // Sigmoid
            case 2: return std::tanh(x);                   // Tanh
            default: return std::max(0.f, x);
        }
    }

    void forwardPass() {
        std::uniform_real_distribution<float> d01(0.f, 1.f);
        std::uniform_real_distribution<float> noise_d(-0.5f, 0.5f);

        // Random input
        for (int i = 0; i < arch[0]; i++) {
            activations[0][i] = d01(rng) > 0.45f
                ? d01(rng) * 0.7f + 0.3f
                : d01(rng) * 0.1f;
            activations[0][i] = std::clamp(
                activations[0][i] + noise_d(rng) * noise, 0.f, 1.f);
        }

        // Forward
        for (int l = 0; l < NUM_LAYERS - 1; l++) {
            for (int j = 0; j < arch[l+1]; j++) {
                float s = biases[l][j];
                for (int i = 0; i < arch[l]; i++)
                    s += activations[l][i] * weights[l][i * arch[l+1] + j] * wScale;
                activations[l+1][j] = (l == NUM_LAYERS - 2) ? s : activate(s);
            }
        }

        // Softmax on output
        auto& out = activations[NUM_LAYERS - 1];
        float mx = *std::max_element(out.begin(), out.end());
        float sum = 0.f;
        for (auto& v : out) { v = std::exp(v - mx); sum += v; }
        for (auto& v : out) v /= sum;

        // Find prediction
        prediction = 0;
        for (int i = 1; i < arch[NUM_LAYERS-1]; i++)
            if (out[i] > out[prediction]) prediction = i;
        confidence = out[prediction];
    }

    void triggerInference() {
        forwardPass();
        inferActive = true;
        inferPhase = 0;
        inferTimer = 0.f;
        spawnPulses(0);
    }

    void spawnPulses(int lyr) {
        int maxP = 50;
        int count = 0;
        for (size_t i = 0; i < edges.size(); i++) {
            if (edges[i].layer != lyr) continue;
            float br = std::min(1.f,
                (std::abs(activations[lyr][edges[i].fromIdx]) +
                 std::abs(activations[lyr+1][edges[i].toIdx])) * 0.6f);
            if (br > 0.08f && count < maxP) {
                std::uniform_real_distribution<float> d(0.f, 0.5f);
                pulses.push_back({(int)i, 0.f, br, 0.7f + d(rng), lyr});
                count++;
            }
        }
    }

    void update(float dt) {
        time += dt;

        // Smooth activations
        for (int l = 0; l < NUM_LAYERS; l++)
            for (int i = 0; i < arch[l]; i++)
                animActivations[l][i] += (activations[l][i] - animActivations[l][i])
                                         * std::min(1.f, dt * 4.5f);

        // Inference FSM
        if (inferActive) {
            inferTimer += dt * pulseSpeed;
            if (inferTimer > 0.7f) {
                inferTimer = 0.f;
                inferPhase++;
                if (inferPhase < NUM_LAYERS - 1)
                    spawnPulses(inferPhase);
                else {
                    inferActive = false;
                    inferPhase = -1;
                }
            }
        }

        // Auto inference
        autoTimer += dt;
        if (autoTimer > 3.2f && autoInfer) {
            autoTimer = 0.f;
            triggerInference();
        }

        // Auto rotate
        if (autoRotate) camYaw += dt * 0.18f;

        // Update pulses
        for (int i = (int)pulses.size() - 1; i >= 0; i--) {
            pulses[i].progress += dt * pulses[i].speed * pulseSpeed * 1.6f;
            if (pulses[i].progress >= 1.f)
                pulses.erase(pulses.begin() + i);
        }
    }

    // Project 3D to 2D screen coords
    struct Proj2D {
        float px, py, z2, sc;
    };

    Proj2D project(float x, float y, float z, float viewW, float viewH) const {
        float cy = std::cos(camYaw), sy = std::sin(camYaw);
        float x1 = x * cy - z * sy, z1 = x * sy + z * cy;
        float cx = std::cos(camPitch), sx = std::sin(camPitch);
        float y1 = y * cx - z1 * sx, z2 = y * sx + z1 * cx;
        float sc = camDist / (camDist + z2);
        float ds = std::min(viewW, viewH) * 0.14f;
        return {viewW/2 + x1*sc*ds, viewH/2 + y1*sc*ds, z2, sc};
    }

    int totalNodes() const {
        int s = 0;
        for (int l = 0; l < NUM_LAYERS; l++) s += arch[l];
        return s;
    }

    int totalEdges() const { return (int)edges.size(); }

    int totalParams() const {
        int s = totalEdges();
        for (int l = 1; l < NUM_LAYERS; l++) s += arch[l];
        return s;
    }
};

} // namespace neural

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
