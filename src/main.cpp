// ============================================================================
// SimLab TUI — Simulation Explorer
// A terminal-style C++/OpenGL application with ImGui that hosts multiple
// computer graphics simulations in a single window.
//
// Left panel:  File-tree explorer to select simulations
// Right panel: Full OpenGL viewport rendering the active simulation
//
// Developer: @swadhinbiswas
// Contact:   swadhinbiswas.cse@gmail.com
// ============================================================================

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "hypercube4d.hpp"
#include "neural_sim.hpp"
#include "particle_sim.hpp"
#include "mandelbrot_sim.hpp"
#include "cloth_sim.hpp"
#include "boids_sim.hpp"
#include "lorenz_sim.hpp"
#include "wave_sim.hpp"
#include "pendulum_sim.hpp"
#include "gol3d_sim.hpp"
#include "audio_sim.hpp"

#include <random>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <ctime>

// ============================================================================
// Shader helper — load from file, compile, link
// ============================================================================
static std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open shader: " << path << "\n"; return ""; }
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

static GLuint compileShader(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
    }
    return s;
}

static GLuint createProgram(const std::string& vertPath, const std::string& fragPath) {
    auto vs = readFile(vertPath), fs = readFile(fragPath);
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    int ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetProgramInfoLog(p, 512, nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ============================================================================
// Window config
// ============================================================================
static constexpr int WIN_W = 1440;
static constexpr int WIN_H = 900;

// ============================================================================
// Active simulation enum
// ============================================================================
enum class ActiveSim { NONE, HYPERCUBE_4D, NEURAL_VIZ, PARTICLES, MANDELBROT, CLOTH, BOIDS, LORENZ, WAVE, PENDULUM, GOL3D, AUDIO };
static ActiveSim activeSim = ActiveSim::NONE;

// ============================================================================
// 4D Hypercube state
// ============================================================================
static hd::Rotation4D rot4d;
static hd::Dataset4D  dataset;
static int currentDataset = 0; // 0=tesseract, 1=klein, 2=hopf

static float wDist    = 2.0f;
static float ptSize   = 8.0f;
static float ptAlpha  = 0.85f;
static bool  showEdges = true;
static int   colorMode = 0;

// 4D camera
static glm::vec3 cam4dPos   = {0.f, 0.f, 4.f};
static glm::vec3 cam4dFront = {0.f, 0.f, -1.f};
static glm::vec3 cam4dUp    = {0.f, 1.f,  0.f};
static float yaw4d = -90.f, pitch4d = 0.f;

// 4D GPU buffers
static GLuint ptVao, ptVbo;
static GLuint edgeVao4d, edgeVbo4d, edgeEbo4d;
static GLuint ptShader, edgeShader;

// ============================================================================
// Neural viz state
// ============================================================================
static neural::NeuralSim neuralSim;
static GLuint neuralProgram;
static GLuint nLineVao, nLineVbo;
static GLuint nPointVao, nPointVbo;

// ============================================================================
// Mouse / interaction state
// ============================================================================
static bool  captureMouse = false;
static bool  firstMouse   = true;
static float lastMX = WIN_W/2.f, lastMY = WIN_H/2.f;
static bool  dragNeural = false;

// ============================================================================
// Particle sim state
// ============================================================================
static particle::ParticleSim particleSim;
static GLuint particleProgram;
static GLuint partVao, partVbo;

// ============================================================================
// Mandelbrot state
// ============================================================================
static mandelbrot::MandelbrotSim mandelbrotSim;
static GLuint mandelbrotTex;
static GLuint mandelbrotVao, mandelbrotVbo;
static GLuint mandelbrotProgram;
static bool mbDragging = false;
static double mbDragStartX = 0, mbDragStartY = 0;
static int mandelbrotRes = 400; // render resolution
static bool mandelbrotDirty = true;

// ============================================================================
// Cloth sim state
// ============================================================================
static cloth::ClothSim clothSim;
static GLuint clothProgram;
static GLuint clothVao, clothVbo, clothEbo;

// ============================================================================
// Boids sim state
// ============================================================================
static boids::BoidsSim boidsSim;
static GLuint boidsVao, boidsVbo;
// Reuse particleProgram for boids (same vertex layout)

// ============================================================================
// TUI file tree state
// ============================================================================
static bool folder4dOpen       = true;
static bool folderNuralOpen    = true;
static bool folderParticleOpen = true;
static bool folderFractalOpen  = true;
static bool folderPhysicsOpen  = true;
static bool folderBoidsOpen    = true;
static bool folderLorenzOpen   = true;
static bool folderWaveOpen     = true;
static bool folderPendOpen     = true;
static bool folderGolOpen      = true;

// ============================================================================
// Lorenz state
// ============================================================================
static lorenz::LorenzSim lorenzSim;
static GLuint lorenzVao, lorenzVbo;
// Reuses particleProgram for lines (same layout but used as GL_LINE_STRIP)

// ============================================================================
// Wave state
// ============================================================================
static wave::WaveSim waveSim;
static GLuint waveVao, waveVbo, waveEbo;
// Reuses clothProgram (pos3 + normal3 + color3)

// ============================================================================
// Pendulum state
// ============================================================================
static pendulum::DoublePendulumSim pendSim;
static GLuint pendVao, pendVbo;

// ============================================================================
// Game of Life 3D state
// ============================================================================
static gol3d::GameOfLife3D golSim;
static GLuint golVao, golVbo;

// ============================================================================
// Audio sim state
// ============================================================================
static audio::AudioSim audioSim;
static GLuint audioVao, audioVbo;
static bool folderAudioOpen = true;

// ============================================================================
// Build tesseract edge indices
// ============================================================================
static std::vector<unsigned int> buildTesseractEdges() {
    std::vector<unsigned int> idx;
    for (int i = 0; i < 16; i++)
        for (int j = i+1; j < 16; j++) {
            int d = i ^ j;
            if (d && !(d & (d-1))) { idx.push_back(i); idx.push_back(j); }
        }
    return idx;
}

// ============================================================================
// Initialize 4D simulation GPU resources
// ============================================================================
static void init4D() {
    dataset = hd::Dataset4D::tesseract();
    dataset.normalize();

    // Point cloud VBO
    glGenVertexArrays(1, &ptVao);
    glGenBuffers(1, &ptVbo);
    glBindVertexArray(ptVao);
    glBindBuffer(GL_ARRAY_BUFFER, ptVbo);
    glBufferData(GL_ARRAY_BUFFER,
        dataset.points.size() * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(4*sizeof(float)));
    glEnableVertexAttribArray(2);

    // Edge IBO
    auto edgeIdx = buildTesseractEdges();
    glGenVertexArrays(1, &edgeVao4d);
    glGenBuffers(1, &edgeVbo4d);
    glGenBuffers(1, &edgeEbo4d);
    glBindVertexArray(edgeVao4d);
    glBindBuffer(GL_ARRAY_BUFFER, ptVbo); // share VBO
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEbo4d);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        edgeIdx.size()*sizeof(unsigned int), edgeIdx.data(), GL_STATIC_DRAW);

    ptShader   = createProgram("shaders/point.vert", "shaders/point.frag");
    edgeShader = createProgram("shaders/edge.vert",  "shaders/edge.frag");
}

// ============================================================================
// Switch 4D dataset
// ============================================================================
static void switchDataset(int id) {
    currentDataset = id;
    switch(id) {
        case 0: dataset = hd::Dataset4D::tesseract(); break;
        case 1: dataset = hd::Dataset4D::kleinBottle(); break;
        case 2: dataset = hd::Dataset4D::hopfFibration(); break;
    }
    dataset.normalize();

    // Reallocate VBO
    glBindBuffer(GL_ARRAY_BUFFER, ptVbo);
    glBufferData(GL_ARRAY_BUFFER,
        dataset.points.size() * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // Rebuild edge indices if tesseract
    if (id == 0) {
        auto edgeIdx = buildTesseractEdges();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEbo4d);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
            edgeIdx.size()*sizeof(unsigned int), edgeIdx.data(), GL_STATIC_DRAW);
    }
}

// ============================================================================
// Initialize neural sim GPU resources
// ============================================================================
static void initNeural() {
    neuralSim.init();
    neuralSim.triggerInference();

    neuralProgram = createProgram("shaders/neural.vert", "shaders/neural.frag");

    // Line VAO — for edges and pulses
    glGenVertexArrays(1, &nLineVao);
    glGenBuffers(1, &nLineVbo);
    glBindVertexArray(nLineVao);
    glBindBuffer(GL_ARRAY_BUFFER, nLineVbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    // Point VAO — for nodes
    glGenVertexArrays(1, &nPointVao);
    glGenBuffers(1, &nPointVbo);
    glBindVertexArray(nPointVao);
    glBindBuffer(GL_ARRAY_BUFFER, nPointVbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
}

// ============================================================================
// Initialize Particle sim GPU resources
// ============================================================================
static void initParticles() {
    particleSim.init();
    particleProgram = createProgram("shaders/particle.vert", "shaders/particle.frag");

    // layout: vec3 pos + vec4 color + float size = 8 floats per vertex
    glGenVertexArrays(1, &partVao);
    glGenBuffers(1, &partVbo);
    glBindVertexArray(partVao);
    glBindBuffer(GL_ARRAY_BUFFER, partVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(7*sizeof(float)));
    glEnableVertexAttribArray(2);
}

// ============================================================================
// Initialize Mandelbrot sim GPU resources
// ============================================================================
static void initMandelbrot() {
    mandelbrotSim.init();

    // Simple textured fullscreen quad shader
    std::string vs = R"(
#version 460 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() { gl_Position = vec4(aPos, 0, 1); vUV = aUV; })";
    std::string fs = R"(
#version 460 core
in vec2 vUV;
uniform sampler2D uTex;
out vec4 FragColor;
void main() { FragColor = texture(uTex, vUV); })";
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    mandelbrotProgram = glCreateProgram();
    glAttachShader(mandelbrotProgram, v); glAttachShader(mandelbrotProgram, f);
    glLinkProgram(mandelbrotProgram);
    glDeleteShader(v); glDeleteShader(f);

    // Fullscreen quad
    float quad[] = {
        -1,-1, 0,0,  1,-1, 1,0,  -1,1, 0,1,
         1,-1, 1,0,  1, 1, 1,1,  -1,1, 0,1
    };
    glGenVertexArrays(1, &mandelbrotVao);
    glGenBuffers(1, &mandelbrotVbo);
    glBindVertexArray(mandelbrotVao);
    glBindBuffer(GL_ARRAY_BUFFER, mandelbrotVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Texture for CPU-rendered fractal
    glGenTextures(1, &mandelbrotTex);
    glBindTexture(GL_TEXTURE_2D, mandelbrotTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

// ============================================================================
// Initialize Cloth sim GPU resources
// ============================================================================
static void initCloth() {
    clothSim.init();
    clothProgram = createProgram("shaders/cloth.vert", "shaders/cloth.frag");

    // layout: vec3 pos + vec3 normal + vec3 color = 9 floats per vertex
    glGenVertexArrays(1, &clothVao);
    glGenBuffers(1, &clothVbo);
    glGenBuffers(1, &clothEbo);
    glBindVertexArray(clothVao);

    glBindBuffer(GL_ARRAY_BUFFER, clothVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, clothEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, clothSim.indices.size()*sizeof(unsigned int),
                 clothSim.indices.data(), GL_STATIC_DRAW);
}

// ============================================================================
// Initialize Boids sim GPU resources
// ============================================================================
static void initBoids() {
    boidsSim.init();
    // Reuse particleProgram (same vertex layout: pos3 + color4 + size1)
    glGenVertexArrays(1, &boidsVao);
    glGenBuffers(1, &boidsVbo);
    glBindVertexArray(boidsVao);
    glBindBuffer(GL_ARRAY_BUFFER, boidsVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(7*sizeof(float)));
    glEnableVertexAttribArray(2);
}

// ============================================================================
// Initialize Lorenz sim GPU resources
// ============================================================================
static void initLorenz() {
    lorenzSim.init();
    // Line strip VAO: pos3 + color4 + size1 = 8 floats (reuse particle layout)
    glGenVertexArrays(1, &lorenzVao);
    glGenBuffers(1, &lorenzVbo);
    glBindVertexArray(lorenzVao);
    glBindBuffer(GL_ARRAY_BUFFER, lorenzVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
}

// ============================================================================
// Initialize Wave sim GPU resources
// ============================================================================
static void initWave() {
    waveSim.init();
    // Reuse cloth shader (pos3 + normal3 + color3 = 9 floats)
    glGenVertexArrays(1, &waveVao);
    glGenBuffers(1, &waveVbo);
    glGenBuffers(1, &waveEbo);
    glBindVertexArray(waveVao);
    glBindBuffer(GL_ARRAY_BUFFER, waveVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, waveEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, waveSim.indices.size()*sizeof(unsigned int),
                 waveSim.indices.data(), GL_STATIC_DRAW);
}

// ============================================================================
// Initialize Pendulum sim GPU resources
// ============================================================================
static void initPendulum() {
    pendSim.init();
    // Simple line/point rendering VAO: pos2 as pos3(z=0) + color4 = 7 floats
    glGenVertexArrays(1, &pendVao);
    glGenBuffers(1, &pendVbo);
    glBindVertexArray(pendVao);
    glBindBuffer(GL_ARRAY_BUFFER, pendVbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
}

// ============================================================================
// Initialize Game of Life 3D GPU resources
// ============================================================================
static void initGol3d() {
    golSim.init();
    // Point rendering: pos3 + color4 + size1 = 8 floats (reuse particle layout)
    glGenVertexArrays(1, &golVao);
    glGenBuffers(1, &golVbo);
    glBindVertexArray(golVao);
    glBindBuffer(GL_ARRAY_BUFFER, golVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(7*sizeof(float)));
    glEnableVertexAttribArray(2);
}

// ============================================================================
// Initialize Audio sim GPU resources
// ============================================================================
static void initAudio() {
    audioSim.init();
    glGenVertexArrays(1, &audioVao);
    glGenBuffers(1, &audioVbo);
    glBindVertexArray(audioVao);
    glBindBuffer(GL_ARRAY_BUFFER, audioVbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
}

// ============================================================================
// Render Audio Visualizer
// ============================================================================
static void renderAudio(float dt, float vpX, float vpY, float vpW, float vpH) {
    audioSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);

    std::vector<float> data; // pos2 + color4 + size1 = 7 floats

    std::lock_guard<std::mutex> lock(audioSim.mtx);

    if (audioSim.vizMode == 0) {
        // === MODE 0: Waveform (top half) + Spectrum bars (bottom half) ===

        // Waveform
        int waveN = (int)audioSim.waveform.size();
        if (waveN > 0) {
            for (int i = 0; i < waveN; i += 4) {
                float x = -1.f + 2.f * (float)i / waveN;
                float y = audioSim.waveform[i] * audioSim.waveScale * 0.4f + 0.5f;
                y = std::clamp(y, 0.05f, 0.95f);
                float vol = std::abs(audioSim.waveform[i]) * audioSim.sensitivity;
                float r, g, b;
                audioSim.getBarColor(i, waveN, vol, r, g, b);
                data.insert(data.end(), {x, y, r, g, b, 0.8f, 2.f});
            }
        }
        int waveVerts = (int)(data.size() / 7);

        // Spectrum bars
        int nb = audioSim.numBars;
        int specN = (int)audioSim.smoothSpec.size();
        if (specN > 0 && nb > 0) {
            float barW = 2.f / nb;
            for (int i = 0; i < nb; i++) {
                // Average bins for this bar
                int binStart = i * specN / nb;
                int binEnd   = (i+1) * specN / nb;
                float avg = 0.f;
                int cnt = 0;
                for (int b = binStart; b < binEnd && b < specN; b++) {
                    avg += audioSim.smoothSpec[b];
                    cnt++;
                }
                if (cnt > 0) avg /= cnt;

                float barH = std::clamp(avg * 2.f, 0.f, 0.45f);
                float x = -1.f + barW * i + barW * 0.5f;

                float r, g, b;
                audioSim.getBarColor(i, nb, avg, r, g, b);

                // Bottom of bar
                data.insert(data.end(), {x, -0.95f, r*0.3f, g*0.3f, b*0.3f, 0.5f, 3.f});
                // Top of bar
                data.insert(data.end(), {x, -0.95f + barH * 2.f, r, g, b, 1.f, 3.f});
            }
        }
        int barVerts = ((int)(data.size() / 7) - waveVerts);

        // Draw
        glUseProgram(neuralProgram);
        glUniform1i(glGetUniformLocation(neuralProgram, "uIsPoint"), 0);

        glBindVertexArray(audioVao);
        glBindBuffer(GL_ARRAY_BUFFER, audioVbo);
        glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);

        // Waveform
        if (waveVerts > 1) {
            glLineWidth(1.5f);
            glDrawArrays(GL_LINE_STRIP, 0, waveVerts);
        }
        // Bars as lines
        if (barVerts > 0) {
            glLineWidth(std::max(1.f, vpW / audioSim.numBars * 0.6f));
            glDrawArrays(GL_LINES, waveVerts, barVerts);
        }

    } else if (audioSim.vizMode == 1) {
        // === MODE 1: Circular visualizer ===
        int specN = (int)audioSim.smoothSpec.size();
        int points = std::min(256, specN);
        float baseR = audioSim.circleRadius;
        float scale = audioSim.circleScale;

        for (int i = 0; i < points; i++) {
            float angle = 2.f * 3.14159265f * i / points;
            float mag = (i < specN) ? audioSim.smoothSpec[i] * scale : 0.f;
            float r = baseR + mag;
            float x = r * std::cos(angle);
            float y = r * std::sin(angle);

            float cr, cg, cb;
            audioSim.getBarColor(i, points, mag, cr, cg, cb);
            data.insert(data.end(), {x, y, cr, cg, cb, 0.9f, 3.f});
        }

        // Also add mirrored circle
        for (int i = 0; i < points; i++) {
            float angle = 2.f * 3.14159265f * i / points;
            float mag = (i < specN) ? audioSim.smoothSpec[i] * scale : 0.f;
            float r = baseR - mag * 0.5f;
            if (r < 0.01f) r = 0.01f;
            float x = r * std::cos(angle);
            float y = r * std::sin(angle);
            data.insert(data.end(), {x, y, 0.f, 0.3f, 0.1f, 0.4f, 2.f});
        }

        glUseProgram(neuralProgram);
        glUniform1i(glGetUniformLocation(neuralProgram, "uIsPoint"), 0);

        glBindVertexArray(audioVao);
        glBindBuffer(GL_ARRAY_BUFFER, audioVbo);
        glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);

        glLineWidth(2.f);
        glDrawArrays(GL_LINE_LOOP, 0, points);
        glLineWidth(1.f);
        glDrawArrays(GL_LINE_LOOP, points, points);

    } else if (audioSim.vizMode == 2) {
        // === MODE 2: Spectrogram (scrolling time-frequency plot) ===
        int lines = (int)audioSim.spectrogram.size();
        int bins  = lines > 0 ? std::min(256, (int)audioSim.spectrogram[0].size()) : 0;

        for (int j = 0; j < lines; j++) {
            float y = -1.f + 2.f * (float)j / lines;
            for (int i = 0; i < bins; i++) {
                float x = -1.f + 2.f * (float)i / bins;
                float val = audioSim.spectrogram[j][i];
                float r, g, b;
                audioSim.getBarColor(i, bins, val, r, g, b);
                float alpha = std::clamp(val * 3.f, 0.05f, 1.f);
                data.insert(data.end(), {x, y, r, g, b, alpha, 3.f});
            }
        }

        glUseProgram(neuralProgram);
        glUniform1i(glGetUniformLocation(neuralProgram, "uIsPoint"), 1);

        glBindVertexArray(audioVao);
        glBindBuffer(GL_ARRAY_BUFFER, audioVbo);
        glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);

        glDrawArrays(GL_POINTS, 0, (GLsizei)(data.size() / 7));
    }

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render 4D hypercube into the viewport
// ============================================================================
static void render4D(float dt, float vpX, float vpY, float vpW, float vpH) {
    rot4d.update(dt);
    auto mat4 = rot4d.matrix();

    // Project 4D → 3D → upload
    std::vector<float> gpuData;
    gpuData.reserve(dataset.points.size() * 5);
    for (size_t i = 0; i < dataset.points.size(); i++) {
        hd::Vec4 r = mat4 * dataset.points[i];
        glm::vec4 p = hd::project4Dto3D(r, wDist);
        gpuData.insert(gpuData.end(), {
            p.x, p.y, p.z,
            p.w,
            (float)(i < dataset.labels.size() ? dataset.labels[i] : 0)
        });
    }
    glBindBuffer(GL_ARRAY_BUFFER, ptVbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, gpuData.size()*sizeof(float), gpuData.data());

    // Set viewport
    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    glm::mat4 view = glm::lookAt(cam4dPos, cam4dPos+cam4dFront, cam4dUp);
    glm::mat4 proj = glm::perspective(glm::radians(45.f), vpW/vpH, 0.01f, 100.f);

    // Draw edges
    if (showEdges && currentDataset == 0) {
        glUseProgram(edgeShader);
        glUniformMatrix4fv(glGetUniformLocation(edgeShader,"uView"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(edgeShader,"uProj"), 1, GL_FALSE, glm::value_ptr(proj));
        glUniform1f(glGetUniformLocation(edgeShader,"uAlpha"), 0.45f);
        glUniform1f(glGetUniformLocation(edgeShader,"uWNorm"), 1.f/wDist);
        glBindVertexArray(edgeVao4d);
        glDrawElements(GL_LINES, 64, GL_UNSIGNED_INT, 0); // 32 edges * 2
    }

    // Draw points
    glUseProgram(ptShader);
    glUniformMatrix4fv(glGetUniformLocation(ptShader,"uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(ptShader,"uProj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniform1f(glGetUniformLocation(ptShader,"uPointSize"), ptSize);
    glUniform1f(glGetUniformLocation(ptShader,"uAlpha"), ptAlpha);
    glUniform1f(glGetUniformLocation(ptShader,"uWNorm"), 1.f/wDist);
    glUniform1i(glGetUniformLocation(ptShader,"uColorMode"), colorMode);
    glUniform1i(glGetUniformLocation(ptShader,"uNumClasses"), std::max(1, dataset.numClasses()));
    glBindVertexArray(ptVao);
    glDrawArrays(GL_POINTS, 0, (GLsizei)dataset.points.size());

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Neural Viz into the viewport
// ============================================================================
static void renderNeural(float dt, float vpX, float vpY, float vpW, float vpH) {
    neuralSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    glUseProgram(neuralProgram);

    // Layer colors: green, cyan, amber, coral
    float lcolors[4][3] = {
        {0.f, 1.f, 0.25f},
        {0.f, 0.9f, 1.f},
        {1.f, 0.7f, 0.28f},
        {1.f, 0.42f, 0.21f}
    };

    // Project all nodes
    struct ProjNode { float px, py, z2, sc; int layer, idx; };
    std::vector<ProjNode> projected;
    for (auto& n : neuralSim.nodes) {
        auto p = neuralSim.project(n.x, n.y, n.z, vpW, vpH);
        // Convert to NDC
        float nx = (p.px / vpW) * 2.f - 1.f;
        float ny = 1.f - (p.py / vpH) * 2.f;
        projected.push_back({nx, ny, p.z2, p.sc, n.layer, n.index});
    }

    // === Draw edges as lines ===
    std::vector<float> lineData;
    for (auto& e : neuralSim.edges) {
        auto& f = projected[e.fromNode];
        auto& t = projected[e.toNode];
        float fa = neuralSim.animActivations[e.layer][e.fromIdx];
        float ta = (e.layer+1 < neural::NeuralSim::NUM_LAYERS)
                   ? neuralSim.animActivations[e.layer+1][e.toIdx] : 0.f;
        float act = std::clamp((fa + std::abs(ta)) * 0.5f, 0.f, 1.f);
        float w = e.weight * neuralSim.wScale;
        bool pos = w > 0;
        float alpha = 0.015f + std::abs(w)*0.1f + act*0.22f;
        float r = pos ? 0.f : (40.f + act*60.f)/255.f;
        float g = pos ? (55.f + act*70.f)/255.f : act*30.f/255.f;
        float b = pos ? act*30.f/255.f : act*40.f/255.f;

        lineData.insert(lineData.end(), {f.px, f.py, r, g, b, alpha, 1.f});
        lineData.insert(lineData.end(), {t.px, t.py, r, g, b, alpha, 1.f});
    }

    // Upload and draw edges
    glUniform1i(glGetUniformLocation(neuralProgram, "uIsPoint"), 0);
    glBindVertexArray(nLineVao);
    glBindBuffer(GL_ARRAY_BUFFER, nLineVbo);
    glBufferData(GL_ARRAY_BUFFER, lineData.size()*sizeof(float), lineData.data(), GL_DYNAMIC_DRAW);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, (GLsizei)(lineData.size() / 7));

    // === Draw pulses as thick bright lines ===
    std::vector<float> pulseData;
    for (auto& p : neuralSim.pulses) {
        if (p.edgeIdx < 0 || p.edgeIdx >= (int)neuralSim.edges.size()) continue;
        auto& e = neuralSim.edges[p.edgeIdx];
        auto& f = projected[e.fromNode];
        auto& t = projected[e.toNode];

        float t1 = p.progress;
        float t0 = std::max(0.f, t1 - 0.12f);
        float px0 = f.px + (t.px - f.px)*t0, py0 = f.py + (t.py - f.py)*t0;
        float px1 = f.px + (t.px - f.px)*t1, py1 = f.py + (t.py - f.py)*t1;

        auto& col = lcolors[p.layer % 4];
        // Trail start (transparent)
        pulseData.insert(pulseData.end(), {px0, py0, col[0], col[1], col[2], 0.f, 2.f});
        // Trail end (bright)
        pulseData.insert(pulseData.end(), {px1, py1, col[0], col[1], col[2], p.brightness*0.85f, 2.f});
    }

    if (!pulseData.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, nLineVbo);
        glBufferData(GL_ARRAY_BUFFER, pulseData.size()*sizeof(float), pulseData.data(), GL_DYNAMIC_DRAW);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, (GLsizei)(pulseData.size() / 7));
    }

    // === Draw nodes as points ===
    // Sort by depth (back to front)
    std::sort(projected.begin(), projected.end(),
        [](const ProjNode& a, const ProjNode& b) { return a.z2 < b.z2; });

    std::vector<float> pointData;
    for (auto& n : projected) {
        float act = std::clamp(neuralSim.animActivations[n.layer][n.idx], 0.f, 1.f);
        float breathe = 0.08f + 0.04f * std::sin(neuralSim.time*1.3f + n.idx*0.6f + n.layer*2.1f);
        float vis = std::max(breathe, act);
        auto& col = lcolors[n.layer % 4];
        float alpha = std::min(1.f, 0.15f + vis * 0.85f);
        float sz = (3.5f + n.sc * 26.f) * 2.f;

        pointData.insert(pointData.end(), {n.px, n.py, col[0], col[1], col[2], alpha, sz});
    }

    glUniform1i(glGetUniformLocation(neuralProgram, "uIsPoint"), 1);
    glBindVertexArray(nPointVao);
    glBindBuffer(GL_ARRAY_BUFFER, nPointVbo);
    glBufferData(GL_ARRAY_BUFFER, pointData.size()*sizeof(float), pointData.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, (GLsizei)(pointData.size() / 7));

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Particle System
// ============================================================================
static void renderParticles(float dt, float vpX, float vpY, float vpW, float vpH) {
    particleSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    // Build vertex data: pos3 + color4 + size1 = 8 floats
    std::vector<float> data;
    data.reserve(particleSim.particles.size() * 8);
    for (auto& p : particleSim.particles) {
        float r, g, b, a;
        particleSim.getColor(p, r, g, b, a);
        data.insert(data.end(), {p.x, p.y, p.z, r, g, b, a, p.size});
    }

    glUseProgram(particleProgram);

    // Camera orbit
    float cy = std::cos(particleSim.camYaw), sy = std::sin(particleSim.camYaw);
    float cp = std::cos(particleSim.camPitch), sp = std::sin(particleSim.camPitch);
    glm::vec3 eye = glm::vec3(
        particleSim.camDist * cy * cp,
        particleSim.camDist * sp + 1.f,
        particleSim.camDist * sy * cp
    );
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0,1,0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), vpW/vpH, 0.01f, 100.f);

    glUniformMatrix4fv(glGetUniformLocation(particleProgram,"uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(particleProgram,"uProj"), 1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(partVao);
    glBindBuffer(GL_ARRAY_BUFFER, partVbo);
    glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);

    glDepthMask(GL_FALSE);
    glDrawArrays(GL_POINTS, 0, (GLsizei)(data.size() / 8));
    glDepthMask(GL_TRUE);

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Mandelbrot
// ============================================================================
static void renderMandelbrot(float dt, float vpX, float vpY, float vpW, float vpH) {
    mandelbrotSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    int rw = std::min(mandelbrotRes, (int)vpW);
    int rh = std::min(mandelbrotRes, (int)vpH);
    if (rw > 0 && rh > 0) {
        mandelbrotSim.render(rw, rh);
        glBindTexture(GL_TEXTURE_2D, mandelbrotTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rw, rh, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     mandelbrotSim.pixels.data());
    }

    glUseProgram(mandelbrotProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mandelbrotTex);
    glUniform1i(glGetUniformLocation(mandelbrotProgram, "uTex"), 0);

    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(mandelbrotVao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glEnable(GL_DEPTH_TEST);

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Cloth
// ============================================================================
static void renderCloth(float dt, float vpX, float vpY, float vpW, float vpH) {
    clothSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    // Build vertex data: pos3 + normal3 + color3 = 9 floats per vertex
    std::vector<float> verts;
    verts.reserve(clothSim.points.size() * 9);
    for (size_t i = 0; i < clothSim.points.size(); i++) {
        auto& p = clothSim.points[i];
        float r, g, b;
        if (clothSim.colorMode == 0) {
            // Stress-based coloring
            int gi = (int)i % clothSim.gridW;
            int gj = (int)i / clothSim.gridW;
            float stress = 0.f;
            int cnt = 0;
            if (gi > 0) { stress += clothSim.getStress(i, i-1); cnt++; }
            if (gi < clothSim.gridW-1) { stress += clothSim.getStress(i, i+1); cnt++; }
            if (gj > 0) { stress += clothSim.getStress(i, i-clothSim.gridW); cnt++; }
            if (gj < clothSim.gridH-1) { stress += clothSim.getStress(i, i+clothSim.gridW); cnt++; }
            if (cnt > 0) stress /= cnt;
            stress = std::clamp(stress * 5.f, 0.f, 1.f);
            r = stress; g = 0.3f + (1.f-stress)*0.5f; b = 1.f - stress;
        } else if (clothSim.colorMode == 1) {
            r = std::abs(p.nx)*0.5f+0.5f; g = std::abs(p.ny)*0.5f+0.5f; b = std::abs(p.nz)*0.5f+0.5f;
        } else {
            r = 0.2f; g = 0.5f; b = 0.8f;
        }
        if (p.pinned) { r = 1.f; g = 0.3f; b = 0.1f; }
        verts.insert(verts.end(), {p.x, p.y, p.z, p.nx, p.ny, p.nz, r, g, b});
    }

    glUseProgram(clothProgram);

    float cy = std::cos(clothSim.camYaw), sy = std::sin(clothSim.camYaw);
    float cp = std::cos(clothSim.camPitch), sp = std::sin(clothSim.camPitch);
    glm::vec3 eye = glm::vec3(
        clothSim.camDist * cy * cp,
        clothSim.camDist * sp + 1.f,
        clothSim.camDist * sy * cp
    );
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0,1,0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), vpW/vpH, 0.01f, 100.f);

    glUniformMatrix4fv(glGetUniformLocation(clothProgram,"uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(clothProgram,"uProj"), 1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(clothVao);
    glBindBuffer(GL_ARRAY_BUFFER, clothVbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(float), verts.data(), GL_DYNAMIC_DRAW);

    if (clothSim.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, (GLsizei)clothSim.indices.size(), GL_UNSIGNED_INT, 0);
    if (clothSim.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Boids
// ============================================================================
static void renderBoids(float dt, float vpX, float vpY, float vpW, float vpH) {
    boidsSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    float gcolors[5][3] = {
        {0.f, 1.f, 0.4f}, {0.f, 0.8f, 1.f}, {1.f, 0.6f, 0.2f},
        {1.f, 0.3f, 0.5f}, {0.6f, 0.4f, 1.f}
    };

    std::vector<float> data;
    data.reserve(boidsSim.boids.size() * 8);
    for (auto& b : boidsSim.boids) {
        auto& c = gcolors[b.group % 5];
        float spd = boids::BoidsSim::len3(b.vx, b.vy, b.vz);
        float bright = std::clamp(spd / boidsSim.maxSpeed, 0.3f, 1.f);
        data.insert(data.end(), {b.x, b.y, b.z,
            c[0]*bright, c[1]*bright, c[2]*bright, 0.9f, boidsSim.boidSize});
    }
    if (boidsSim.predatorOn) {
        data.insert(data.end(), {boidsSim.predX, boidsSim.predY, boidsSim.predZ,
            1.f, 0.1f, 0.1f, 1.f, boidsSim.boidSize * 3.f});
    }

    glUseProgram(particleProgram);

    float cy = std::cos(boidsSim.camYaw), sy = std::sin(boidsSim.camYaw);
    float cp = std::cos(boidsSim.camPitch), sp = std::sin(boidsSim.camPitch);
    glm::vec3 eye = glm::vec3(
        boidsSim.camDist * cy * cp,
        boidsSim.camDist * sp,
        boidsSim.camDist * sy * cp
    );
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), vpW/vpH, 0.01f, 200.f);

    glUniformMatrix4fv(glGetUniformLocation(particleProgram,"uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(particleProgram,"uProj"), 1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(boidsVao);
    glBindBuffer(GL_ARRAY_BUFFER, boidsVbo);
    glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, (GLsizei)(data.size() / 8));

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Lorenz Attractor
// ============================================================================
static void renderLorenz(float dt, float vpX, float vpY, float vpW, float vpH) {
    lorenzSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    if (lorenzSim.trail.empty()) { glDisable(GL_SCISSOR_TEST); return; }

    float maxAge = lorenzSim.trail.empty() ? 1.f : lorenzSim.trail.back().age;

    // Build line data: pos3 + color4 = 7 floats per vertex
    std::vector<float> data;
    data.reserve(lorenzSim.trail.size() * 7);
    for (auto& p : lorenzSim.trail) {
        float r, g, b, a;
        lorenzSim.getColor(p, lorenzSim.colorMode, maxAge, r, g, b, a);
        // Center the attractor around origin
        data.insert(data.end(), {p.x, p.z - 25.f, p.y, r, g, b, a});
    }

    // Camera
    float cy = std::cos(lorenzSim.camYaw), sy = std::sin(lorenzSim.camYaw);
    float cp = std::cos(lorenzSim.camPitch), sp = std::sin(lorenzSim.camPitch);
    glm::vec3 eye = glm::vec3(
        lorenzSim.camDist * cy * cp,
        lorenzSim.camDist * sp,
        lorenzSim.camDist * sy * cp
    );
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), vpW/vpH, 0.01f, 200.f);

    glUseProgram(particleProgram);
    glUniformMatrix4fv(glGetUniformLocation(particleProgram,"uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(particleProgram,"uProj"), 1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(lorenzVao);
    glBindBuffer(GL_ARRAY_BUFFER, lorenzVbo);
    glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);
    glLineWidth(1.5f);
    glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)(data.size() / 7));

    // Draw second attractor if enabled
    if (lorenzSim.showSecond && !lorenzSim.trail2.empty()) {
        std::vector<float> data2;
        data2.reserve(lorenzSim.trail2.size() * 7);
        for (auto& p : lorenzSim.trail2) {
            data2.insert(data2.end(), {p.x, p.z - 25.f, p.y, 1.f, 0.3f, 0.1f, 0.7f});
        }
        glBufferData(GL_ARRAY_BUFFER, data2.size()*sizeof(float), data2.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)(data2.size() / 7));
    }

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Wave Surface
// ============================================================================
static void renderWave(float dt, float vpX, float vpY, float vpW, float vpH) {
    waveSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    int n = waveSim.gridSize;
    float halfN = n / 2.f;
    float scale = 8.f / n;

    // Build mesh: pos3 + normal3 + color3 = 9 floats
    std::vector<float> verts;
    verts.reserve(n * n * 9);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            float h = waveSim.height[j*n+i] * waveSim.heightScale;
            float px = (i - halfN) * scale;
            float pz = (j - halfN) * scale;
            float nx, ny, nz;
            waveSim.getNormal(i, j, nx, ny, nz);

            float r, g, b;
            if (waveSim.colorMode == 0) { // Height
                float t = std::clamp(h * 0.5f + 0.5f, 0.f, 1.f);
                r = t * 0.3f; g = 0.3f + (1.f-std::abs(h*0.3f))*0.5f; b = 1.f - t * 0.5f;
            } else if (waveSim.colorMode == 1) { // Velocity
                float v = std::abs(waveSim.velocity[j*n+i]);
                float t = std::clamp(v * 2.f, 0.f, 1.f);
                r = t; g = 0.5f * (1.f-t); b = 0.8f * (1.f-t);
            } else { // Gradient
                r = std::abs(nx)*0.5f+0.5f; g = std::abs(ny)*0.5f+0.5f; b = std::abs(nz)*0.5f+0.5f;
            }
            verts.insert(verts.end(), {px, h, pz, nx, ny, nz, r, g, b});
        }
    }

    glUseProgram(clothProgram); // reuse cloth shader

    float cy = std::cos(waveSim.camYaw), sy = std::sin(waveSim.camYaw);
    float cp = std::cos(waveSim.camPitch), sp = std::sin(waveSim.camPitch);
    glm::vec3 eye = glm::vec3(
        waveSim.camDist * cy * cp,
        waveSim.camDist * sp,
        waveSim.camDist * sy * cp
    );
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), vpW/vpH, 0.01f, 100.f);

    glUniformMatrix4fv(glGetUniformLocation(clothProgram,"uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(clothProgram,"uProj"), 1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(waveVao);
    glBindBuffer(GL_ARRAY_BUFFER, waveVbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(float), verts.data(), GL_DYNAMIC_DRAW);

    if (waveSim.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, (GLsizei)waveSim.indices.size(), GL_UNSIGNED_INT, 0);
    if (waveSim.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Double Pendulum
// ============================================================================
static void renderPendulum(float dt, float vpX, float vpY, float vpW, float vpH) {
    pendSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);

    float zoom = pendSim.zoom;
    float sx = 2.f / vpW * zoom;
    float sy = 2.f / vpH * zoom;

    // Compute bob positions
    float x1 = pendSim.L1 * std::sin(pendSim.a1);
    float y1 = -pendSim.L1 * std::cos(pendSim.a1);
    float x2 = x1 + pendSim.L2 * std::sin(pendSim.a2);
    float y2 = y1 - pendSim.L2 * std::cos(pendSim.a2);

    // NDC scale
    float ndcScale = 0.25f * zoom;

    // Build data: pos2 + color4 + size1 = 7 floats
    std::vector<float> data;

    // Trail
    if (pendSim.showTrail) {
        float maxAge = pendSim.trail.empty() ? 1.f : pendSim.trail.front().age;
        if (maxAge < 0.01f) maxAge = 1.f;
        for (auto& p : pendSim.trail) {
            float t = std::clamp(p.age / (maxAge * 1.1f), 0.f, 1.f);
            float alpha = 1.f - t;
            data.insert(data.end(), {
                p.x * ndcScale, (p.y + 1.f) * ndcScale,
                0.f, alpha * 0.8f, alpha * 0.3f, alpha * 0.6f, 2.f
            });
        }
    }
    int trailVerts = (int)(data.size() / 7);

    // Arms: origin -> bob1 -> bob2 (as lines)
    float ox = 0.f, oy = 0.f;
    float b1x = x1 * ndcScale, b1y = (y1 + 1.f) * ndcScale;
    float b2x = x2 * ndcScale, b2y = (y2 + 1.f) * ndcScale;

    // Arm 1
    data.insert(data.end(), {ox, 1.f * ndcScale, 0.8f, 0.8f, 0.8f, 1.f, 3.f});
    data.insert(data.end(), {b1x, b1y, 0.8f, 0.8f, 0.8f, 1.f, 3.f});
    // Arm 2
    data.insert(data.end(), {b1x, b1y, 0.6f, 0.6f, 0.6f, 1.f, 3.f});
    data.insert(data.end(), {b2x, b2y, 0.6f, 0.6f, 0.6f, 1.f, 3.f});
    int armStart = trailVerts;

    // Bob points
    data.insert(data.end(), {ox, 1.f * ndcScale, 0.f, 1.f, 0.3f, 1.f, 8.f}); // pivot
    data.insert(data.end(), {b1x, b1y, 0.f, 0.8f, 1.f, 1.f, 12.f}); // bob1
    data.insert(data.end(), {b2x, b2y, 1.f, 0.4f, 0.1f, 1.f, 14.f}); // bob2
    int bobStart = armStart + 4;

    glUseProgram(neuralProgram); // 2D shader
    glUniform1i(glGetUniformLocation(neuralProgram, "uIsPoint"), 0);

    glBindVertexArray(pendVao);
    glBindBuffer(GL_ARRAY_BUFFER, pendVbo);
    glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);

    // Draw trail
    if (trailVerts > 1) {
        glLineWidth(1.5f);
        glDrawArrays(GL_LINE_STRIP, 0, trailVerts);
    }
    // Draw arms
    glLineWidth(2.5f);
    glDrawArrays(GL_LINES, armStart, 4);

    // Draw bobs as points
    glUniform1i(glGetUniformLocation(neuralProgram, "uIsPoint"), 1);
    glDrawArrays(GL_POINTS, bobStart, 3);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// Render Game of Life 3D
// ============================================================================
static void renderGol3d(float dt, float vpX, float vpY, float vpW, float vpH) {
    golSim.update(dt);

    glViewport((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glScissor((int)vpX, (int)vpY, (int)vpW, (int)vpH);
    glEnable(GL_SCISSOR_TEST);

    int n = golSim.gridSize;
    float halfN = n / 2.f;
    float spacing = golSim.cellSize;

    float layerColors[8][3] = {
        {0.f, 1.f, 0.3f}, {0.f, 0.8f, 1.f}, {1.f, 0.6f, 0.f},
        {1.f, 0.3f, 0.5f}, {0.5f, 0.3f, 1.f}, {0.f, 1.f, 0.7f},
        {1.f, 1.f, 0.2f}, {0.8f, 0.4f, 0.f}
    };

    // Build vertex data for live cells
    std::vector<float> data;
    for (int z = 0; z < n; z++)
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++) {
                if (!golSim.cells[golSim.idx(x,y,z)]) continue;
                float px = (x - halfN) * spacing;
                float py = (y - halfN) * spacing;
                float pz = (z - halfN) * spacing;

                float r, g, b;
                if (golSim.colorMode == 0) { // Layer
                    auto& c = layerColors[z % 8];
                    r = c[0]; g = c[1]; b = c[2];
                } else if (golSim.colorMode == 1) { // Neighbors
                    int nb = golSim.countNeighbors(x, y, z);
                    float t = std::clamp(nb / 12.f, 0.f, 1.f);
                    r = t; g = 1.f - t * 0.5f; b = 1.f - t;
                } else { // Position
                    r = (float)x/n; g = (float)y/n; b = (float)z/n;
                }
                data.insert(data.end(), {px, py, pz, r, g, b, golSim.opacity, golSim.cellSize * 10.f});
            }

    glUseProgram(particleProgram);

    float cy = std::cos(golSim.camYaw), sy = std::sin(golSim.camYaw);
    float cp = std::cos(golSim.camPitch), sp = std::sin(golSim.camPitch);
    glm::vec3 eye = glm::vec3(
        golSim.camDist * cy * cp,
        golSim.camDist * sp,
        golSim.camDist * sy * cp
    );
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0), glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), vpW/vpH, 0.01f, 200.f);

    glUniformMatrix4fv(glGetUniformLocation(particleProgram,"uView"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(particleProgram,"uProj"), 1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(golVao);
    glBindBuffer(GL_ARRAY_BUFFER, golVbo);
    glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, (GLsizei)(data.size() / 8));

    glDisable(GL_SCISSOR_TEST);
}

// ============================================================================
// ImGui styling — dark green CRT terminal look
// ============================================================================
static void setupImGuiStyle() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 0.f;
    s.FrameRounding     = 0.f;
    s.GrabRounding      = 0.f;
    s.TabRounding       = 0.f;
    s.ScrollbarRounding = 0.f;
    s.WindowBorderSize  = 1.f;
    s.FrameBorderSize   = 1.f;
    s.WindowPadding     = {8, 6};
    s.FramePadding      = {6, 3};
    s.ItemSpacing       = {6, 4};

    ImVec4* c = s.Colors;
    ImVec4 bg     = {0.024f, 0.039f, 0.027f, 1.f};
    ImVec4 bg2    = {0.043f, 0.059f, 0.047f, 1.f};
    ImVec4 bg3    = {0.055f, 0.078f, 0.063f, 1.f};
    ImVec4 green  = {0.f, 1.f, 0.255f, 1.f};
    ImVec4 greenM = {0.f, 0.8f, 0.21f, 1.f};
    ImVec4 greenD = {0.f, 0.478f, 0.125f, 1.f};
    ImVec4 greenDk= {0.f, 0.18f, 0.05f, 1.f};
    ImVec4 cyan   = {0.f, 0.898f, 1.f, 1.f};
    ImVec4 border = {0.059f, 0.125f, 0.08f, 1.f};
    ImVec4 borderM= {0.094f, 0.188f, 0.125f, 1.f};
    ImVec4 text   = {0.816f, 1.f, 0.878f, 1.f};
    ImVec4 textD  = {0.478f, 0.749f, 0.533f, 1.f};
    ImVec4 textDD = {0.239f, 0.420f, 0.278f, 1.f};

    c[ImGuiCol_WindowBg]          = bg;
    c[ImGuiCol_ChildBg]           = bg;
    c[ImGuiCol_PopupBg]           = bg2;
    c[ImGuiCol_Border]            = borderM;
    c[ImGuiCol_BorderShadow]      = {0,0,0,0};
    c[ImGuiCol_FrameBg]           = bg3;
    c[ImGuiCol_FrameBgHovered]    = greenDk;
    c[ImGuiCol_FrameBgActive]     = greenDk;
    c[ImGuiCol_TitleBg]           = bg2;
    c[ImGuiCol_TitleBgActive]     = bg3;
    c[ImGuiCol_TitleBgCollapsed]  = bg;
    c[ImGuiCol_MenuBarBg]         = bg2;
    c[ImGuiCol_ScrollbarBg]       = bg;
    c[ImGuiCol_ScrollbarGrab]     = greenD;
    c[ImGuiCol_ScrollbarGrabHovered] = greenM;
    c[ImGuiCol_ScrollbarGrabActive]  = green;
    c[ImGuiCol_CheckMark]         = green;
    c[ImGuiCol_SliderGrab]        = greenD;
    c[ImGuiCol_SliderGrabActive]  = green;
    c[ImGuiCol_Button]            = bg3;
    c[ImGuiCol_ButtonHovered]     = greenDk;
    c[ImGuiCol_ButtonActive]      = greenD;
    c[ImGuiCol_Header]            = greenDk;
    c[ImGuiCol_HeaderHovered]     = {0.f, 0.18f, 0.05f, 0.8f};
    c[ImGuiCol_HeaderActive]      = greenD;
    c[ImGuiCol_Separator]         = border;
    c[ImGuiCol_SeparatorHovered]  = greenD;
    c[ImGuiCol_SeparatorActive]   = green;
    c[ImGuiCol_Tab]               = bg2;
    c[ImGuiCol_TabHovered]        = greenDk;
    c[ImGuiCol_TabActive]         = greenDk;
    c[ImGuiCol_Text]              = text;
    c[ImGuiCol_TextDisabled]      = textDD;
    c[ImGuiCol_ResizeGrip]        = greenD;
    c[ImGuiCol_ResizeGripHovered] = greenM;
    c[ImGuiCol_ResizeGripActive]  = green;
    c[ImGuiCol_PlotLines]         = green;
    c[ImGuiCol_PlotLinesHovered]  = cyan;
    c[ImGuiCol_PlotHistogram]     = greenM;
    c[ImGuiCol_PlotHistogramHovered] = green;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    // --- GLFW init ---
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "SimLab TUI — Simulation Explorer", nullptr, nullptr);
    if (!win) { std::cerr << "GLFW window creation failed\n"; return -1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    // --- GLEW init ---
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    // GLEW_ERROR_NO_GLX_DISPLAY (4) is expected on Wayland — safe to ignore
    if (glewErr != GLEW_OK && glewErr != 4) {
        std::cerr << "GLEW init failed: " << glewGetErrorString(glewErr) << "\n";
        return -1;
    }
    // Clear any spurious GL errors generated by GLEW in core profile
    while (glGetError() != GL_NO_ERROR) {}

    // --- ImGui ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    setupImGuiStyle();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    // Load a monospace font (if system font available, else use default)
    // Try to load JetBrains Mono or fall back to default
    ImFont* font = nullptr;
    const char* fontPaths[] = {
        "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf",
        "/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/jetbrains-mono/JetBrainsMono-Regular.ttf",
    };
    for (auto* p : fontPaths) {
        std::ifstream test(p);
        if (test.good()) {
            font = io.Fonts->AddFontFromFileTTF(p, 13.0f);
            break;
        }
    }

    // --- Init simulations ---
    init4D();
    initNeural();
    initParticles();
    initMandelbrot();
    initCloth();
    initBoids();
    initLorenz();
    initWave();
    initPendulum();
    initGol3d();
    initAudio();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);

    double lastTime = glfwGetTime();
    int fps = 60;
    float fpsTimer = 0.f;
    int frameCount = 0;

    // Start with 4D hypercube selected
    activeSim = ActiveSim::HYPERCUBE_4D;

    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float dt = (float)(now - lastTime);
        lastTime = now;
        dt = std::min(dt, 0.05f);

        // FPS counter
        frameCount++;
        fpsTimer += dt;
        if (fpsTimer >= 1.0f) {
            fps = frameCount;
            frameCount = 0;
            fpsTimer = 0.f;
        }

        glfwPollEvents();

        // --- Handle mouse for neural sim ---
        if (activeSim == ActiveSim::NEURAL_VIZ && !io.WantCaptureMouse) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                neuralSim.camYaw   += delta.x * 0.005f;
                neuralSim.camPitch += delta.y * 0.004f;
                neuralSim.camPitch  = std::clamp(neuralSim.camPitch, -0.75f, 0.75f);
                neuralSim.autoRotate = false;
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            }
            float wheel = io.MouseWheel;
            if (wheel != 0.f) {
                neuralSim.camDist = std::clamp(neuralSim.camDist - wheel * 0.3f, 2.f, 11.f);
            }
        }

        // --- Handle mouse for 4D sim ---
        if (activeSim == ActiveSim::HYPERCUBE_4D && !io.WantCaptureMouse) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                yaw4d   += delta.x * 0.15f;
                pitch4d  = std::clamp(pitch4d + delta.y * -0.15f, -89.f, 89.f);
                cam4dFront = glm::normalize(glm::vec3(
                    cos(glm::radians(yaw4d)) * cos(glm::radians(pitch4d)),
                    sin(glm::radians(pitch4d)),
                    sin(glm::radians(yaw4d)) * cos(glm::radians(pitch4d))
                ));
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            }
            float wheel = io.MouseWheel;
            if (wheel != 0.f) {
                cam4dPos += cam4dFront * wheel * 0.25f;
            }
        }

        // --- Handle mouse for particles/boids/cloth (orbit cameras) ---
        auto handleOrbitMouse = [&](float& yaw, float& pitch, float& dist, bool& autoRot, float distMin, float distMax) {
            if (!io.WantCaptureMouse) {
                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                    yaw   += delta.x * 0.005f;
                    pitch += delta.y * 0.004f;
                    pitch  = std::clamp(pitch, -1.2f, 1.2f);
                    autoRot = false;
                    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
                }
                float wheel = io.MouseWheel;
                if (wheel != 0.f)
                    dist = std::clamp(dist - wheel * 0.5f, distMin, distMax);
            }
        };

        if (activeSim == ActiveSim::PARTICLES)
            handleOrbitMouse(particleSim.camYaw, particleSim.camPitch, particleSim.camDist, particleSim.autoRotate, 3.f, 20.f);
        if (activeSim == ActiveSim::CLOTH)
            handleOrbitMouse(clothSim.camYaw, clothSim.camPitch, clothSim.camDist, clothSim.autoRotate, 3.f, 20.f);
        if (activeSim == ActiveSim::BOIDS)
            handleOrbitMouse(boidsSim.camYaw, boidsSim.camPitch, boidsSim.camDist, boidsSim.autoRotate, 5.f, 30.f);
        if (activeSim == ActiveSim::LORENZ)
            handleOrbitMouse(lorenzSim.camYaw, lorenzSim.camPitch, lorenzSim.camDist, lorenzSim.autoRotate, 10.f, 100.f);
        if (activeSim == ActiveSim::WAVE)
            handleOrbitMouse(waveSim.camYaw, waveSim.camPitch, waveSim.camDist, waveSim.autoRotate, 5.f, 30.f);
        if (activeSim == ActiveSim::GOL3D)
            handleOrbitMouse(golSim.camYaw, golSim.camPitch, golSim.camDist, golSim.autoRotate, 10.f, 60.f);

        // --- Handle mouse for Mandelbrot (pan/zoom) ---
        if (activeSim == ActiveSim::MANDELBROT && !io.WantCaptureMouse) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                mandelbrotSim.pan(-delta.x * 0.002 * mandelbrotSim.zoom,
                                   delta.y * 0.002 * mandelbrotSim.zoom);
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            }
            float wheel = io.MouseWheel;
            if (wheel != 0.f) {
                double factor = wheel > 0 ? 0.85 : 1.18;
                mandelbrotSim.zoomIn(factor);
            }
        }

        // --- Begin ImGui frame ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int winW, winH;
        glfwGetFramebufferSize(win, &winW, &winH);

        // ================================================================
        // TOP BAR
        // ================================================================
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({(float)winW, 28});
        ImGui::Begin("##topbar", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoCollapse);

        ImGui::TextColored({0,1,0.255f,1}, "SIMLAB");
        ImGui::SameLine();
        ImGui::TextColored({0,0.898f,1,1}, "_TUI");
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, " | ");
        ImGui::SameLine();
        ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "v1.0");
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, " | ");
        ImGui::SameLine();
        ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "11 SIMULATIONS");
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, " | ");
        ImGui::SameLine();

        // Status badge
        if (activeSim != ActiveSim::NONE) {
            float pulse = 0.5f + 0.5f * std::sin((float)glfwGetTime() * 3.f);
            ImGui::TextColored({0, pulse, 0.1f*pulse, 1}, "  LIVE");
        } else {
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "  IDLE");
        }

        // Clock on far right
        ImGui::SameLine(ImGui::GetWindowWidth() - 90);
        time_t t = std::time(nullptr);
        char clk[16]; std::strftime(clk, sizeof(clk), "%H:%M:%S", std::localtime(&t));
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "%s", clk);

        ImGui::End();

        // ================================================================
        // LEFT PANEL — FILE EXPLORER + CONTROLS
        // ================================================================
        float sideW = 240.f;
        float topH  = 28.f;
        float botH  = 32.f;
        float mainH = winH - topH - botH;

        ImGui::SetNextWindowPos({0, topH});
        ImGui::SetNextWindowSize({sideW, mainH});
        ImGui::Begin("##sidebar", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        // Section header
        ImGui::TextColored({0,0.478f,0.125f,1}, "▸");
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "EXPLORER");
        ImGui::Separator();
        ImGui::Spacing();

        // ---------- Folder: 4d/ ----------
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0,0,0,0));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0, 0.18f, 0.05f, 0.8f));

        bool sel4d = (activeSim == ActiveSim::HYPERCUBE_4D);
        if (ImGui::TreeNodeEx("##4d_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (sel4d ? ImGuiTreeNodeFlags_Selected : 0),
            "%s 4d/", folder4dOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folder4dOpen = true;

            // File entries
            bool clicked = false;
            if (sel4d) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 hypercube4d.cpp", sel4d);
            if (sel4d) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::HYPERCUBE_4D;

            ImGui::TreePop();
        } else { folder4dOpen = false; }

        // ---------- Folder: nural/ ----------
        bool selN = (activeSim == ActiveSim::NEURAL_VIZ);
        if (ImGui::TreeNodeEx("##nural_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selN ? ImGuiTreeNodeFlags_Selected : 0),
            "%s nural/", folderNuralOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderNuralOpen = true;

            bool clicked = false;
            if (selN) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 neural_viz", selN);
            if (selN) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::NEURAL_VIZ;

            ImGui::TreePop();
        } else { folderNuralOpen = false; }

        // ---------- Folder: particles/ ----------
        bool selP = (activeSim == ActiveSim::PARTICLES);
        if (ImGui::TreeNodeEx("##particle_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selP ? ImGuiTreeNodeFlags_Selected : 0),
            "%s particles/", folderParticleOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderParticleOpen = true;
            bool clicked = false;
            if (selP) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 particle_system", selP);
            if (selP) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::PARTICLES;
            ImGui::TreePop();
        } else { folderParticleOpen = false; }

        // ---------- Folder: fractals/ ----------
        bool selM = (activeSim == ActiveSim::MANDELBROT);
        if (ImGui::TreeNodeEx("##fractal_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selM ? ImGuiTreeNodeFlags_Selected : 0),
            "%s fractals/", folderFractalOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderFractalOpen = true;
            bool clicked = false;
            if (selM) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 mandelbrot", selM);
            if (selM) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::MANDELBROT;
            ImGui::TreePop();
        } else { folderFractalOpen = false; }

        // ---------- Folder: physics/ ----------
        bool selC = (activeSim == ActiveSim::CLOTH);
        if (ImGui::TreeNodeEx("##physics_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selC ? ImGuiTreeNodeFlags_Selected : 0),
            "%s physics/", folderPhysicsOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderPhysicsOpen = true;
            bool clicked = false;
            if (selC) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 cloth_sim", selC);
            if (selC) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::CLOTH;
            ImGui::TreePop();
        } else { folderPhysicsOpen = false; }

        // ---------- Folder: boids/ ----------
        bool selB = (activeSim == ActiveSim::BOIDS);
        if (ImGui::TreeNodeEx("##boids_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selB ? ImGuiTreeNodeFlags_Selected : 0),
            "%s boids/", folderBoidsOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderBoidsOpen = true;
            bool clicked = false;
            if (selB) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 flocking", selB);
            if (selB) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::BOIDS;
            ImGui::TreePop();
        } else { folderBoidsOpen = false; }

        // ---------- Folder: chaos/ ----------
        bool selLz = (activeSim == ActiveSim::LORENZ);
        if (ImGui::TreeNodeEx("##lorenz_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selLz ? ImGuiTreeNodeFlags_Selected : 0),
            "%s chaos/", folderLorenzOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderLorenzOpen = true;
            bool clicked = false;
            if (selLz) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 lorenz_attractor", selLz);
            if (selLz) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::LORENZ;
            ImGui::TreePop();
        } else { folderLorenzOpen = false; }

        // ---------- Folder: waves/ ----------
        bool selWv = (activeSim == ActiveSim::WAVE);
        if (ImGui::TreeNodeEx("##wave_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selWv ? ImGuiTreeNodeFlags_Selected : 0),
            "%s waves/", folderWaveOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderWaveOpen = true;
            bool clicked = false;
            if (selWv) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 wave_equation", selWv);
            if (selWv) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::WAVE;
            ImGui::TreePop();
        } else { folderWaveOpen = false; }

        // ---------- Folder: pendulum/ ----------
        bool selPd = (activeSim == ActiveSim::PENDULUM);
        if (ImGui::TreeNodeEx("##pend_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selPd ? ImGuiTreeNodeFlags_Selected : 0),
            "%s pendulum/", folderPendOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderPendOpen = true;
            bool clicked = false;
            if (selPd) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 double_pendulum", selPd);
            if (selPd) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::PENDULUM;
            ImGui::TreePop();
        } else { folderPendOpen = false; }

        // ---------- Folder: automata/ ----------
        bool selGol = (activeSim == ActiveSim::GOL3D);
        if (ImGui::TreeNodeEx("##gol_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selGol ? ImGuiTreeNodeFlags_Selected : 0),
            "%s automata/", folderGolOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderGolOpen = true;
            bool clicked = false;
            if (selGol) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 game_of_life_3d", selGol);
            if (selGol) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::GOL3D;
            ImGui::TreePop();
        } else { folderGolOpen = false; }

        // ---------- Folder: audio/ ----------
        bool selAu = (activeSim == ActiveSim::AUDIO);
        if (ImGui::TreeNodeEx("##audio_folder",
            ImGuiTreeNodeFlags_DefaultOpen | (selAu ? ImGuiTreeNodeFlags_Selected : 0),
            "%s audio/", folderAudioOpen ? "\xF0\x9F\x93\x82" : "\xF0\x9F\x93\x81"))
        {
            folderAudioOpen = true;
            bool clicked = false;
            if (selAu) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,1,0.255f,1));
            clicked = ImGui::Selectable("   \xE2\x94\x94\xE2\x94\x80 audio_visualizer", selAu);
            if (selAu) ImGui::PopStyleColor();
            if (clicked) activeSim = ActiveSim::AUDIO;
            ImGui::TreePop();
        } else { folderAudioOpen = false; }

        ImGui::PopStyleColor(2);

        ImGui::Spacing();
        ImGui::Separator();

        // ---------- Simulation Info ----------
        ImGui::Spacing();
        ImGui::TextColored({0,0.478f,0.125f,1}, "▸");
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "SIM_INFO");
        ImGui::Separator();
        ImGui::Spacing();

        if (activeSim == ActiveSim::HYPERCUBE_4D) {
            ImGui::TextColored({0,1,0.255f,1}, "4D HYPERCUBE EXPLORER");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Tesseract rotation,");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "4D perspective projection");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Points: %zu", dataset.points.size());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "▸");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator();
            ImGui::Spacing();

            // Dataset selector
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "DATASET:");
            if (ImGui::Button("Tesseract", {68, 0})) switchDataset(0);
            ImGui::SameLine();
            if (ImGui::Button("Klein", {50, 0})) switchDataset(1);
            ImGui::SameLine();
            if (ImGui::Button("Hopf", {45, 0})) switchDataset(2);
            ImGui::Spacing();

            ImGui::SliderFloat("W-dist", &wDist, 1.2f, 6.0f);
            ImGui::SliderFloat("Pt size", &ptSize, 2.0f, 24.0f);
            ImGui::SliderFloat("Opacity", &ptAlpha, 0.1f, 1.0f);
            ImGui::Checkbox("Show edges", &showEdges);

            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "COLOR_MODE:");
            ImGui::RadioButton("W-depth", &colorMode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Class", &colorMode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Z-dep", &colorMode, 2);

            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "ROTATION_PLANES:");
            for (int p = 0; p < hd::NUM_PLANES; p++) {
                ImGui::Checkbox(hd::PLANE_NAME[p], &rot4d.active[p]);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(80);
                ImGui::SliderFloat(
                    ("##rs"+std::to_string(p)).c_str(),
                    &rot4d.speeds[p], 0.f, 1.5f);
            }

        } else if (activeSim == ActiveSim::NEURAL_VIZ) {
            ImGui::TextColored({0,1,0.255f,1}, "NEURAL_VIZ");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "MNIST classifier, 4-layer");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "FFN with live inference");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Nodes: %d  Edges: %d",
                neuralSim.totalNodes(), neuralSim.totalEdges());
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Params: %d",
                neuralSim.totalParams());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "▸");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::SliderFloat("W Scale", &neuralSim.wScale, 0.f, 2.f);
            ImGui::SliderFloat("Noise", &neuralSim.noise, 0.f, 1.f);
            ImGui::SliderFloat("Pulse Spd", &neuralSim.pulseSpeed, 0.2f, 2.2f);

            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "ACTIVATION_FN:");
            ImGui::RadioButton("ReLU", &neuralSim.activationFn, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Sigmoid", &neuralSim.activationFn, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Tanh", &neuralSim.activationFn, 2);

            ImGui::Spacing();
            ImGui::Checkbox("Auto rotate", &neuralSim.autoRotate);
            ImGui::Checkbox("Auto infer", &neuralSim.autoInfer);

            if (ImGui::Button("INFER NOW")) neuralSim.triggerInference();
            ImGui::SameLine();
            if (ImGui::Button("RESET")) { neuralSim.init(); neuralSim.triggerInference(); }

            // Prediction display
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "▸");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "INFERENCE_OUTPUT");
            ImGui::Separator();
            ImGui::Spacing();

            if (neuralSim.prediction >= 0) {
                ImGui::TextColored({0,1,0.255f,1}, "PREDICTED: %d", neuralSim.prediction);
                ImGui::TextColored({0,0.898f,1,1}, "CONFIDENCE: %.1f%%",
                    neuralSim.confidence * 100.f);

                // Output bars
                ImGui::Spacing();
                auto& out = neuralSim.activations[neural::NeuralSim::NUM_LAYERS - 1];
                for (int i = 0; i < neuralSim.arch[neural::NeuralSim::NUM_LAYERS - 1]; i++) {
                    char label[8]; snprintf(label, sizeof(label), "[%d]", i);
                    float val = out[i];
                    ImVec4 barColor = (i == neuralSim.prediction)
                        ? ImVec4(0, 1, 0.255f, 1) : ImVec4(0, 0.478f, 0.125f, 1);
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barColor);
                    ImGui::ProgressBar(val, {-1, 12}, label);
                    ImGui::PopStyleColor();
                }
            }

            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Pulses: %zu",
                neuralSim.pulses.size());
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Status: %s",
                neuralSim.inferActive ? "INFERRING" : "READY");

        } else if (activeSim == ActiveSim::PARTICLES) {
            ImGui::TextColored({0,1,0.255f,1}, "PARTICLE SYSTEM");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "GPU particle effects");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "with gravity & wind");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Particles: %d", particleSim.liveCount());

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "EMITTER_MODE:");
            if (ImGui::Button("Fountain", {70,0})) { particleSim.emitterMode = 0; particleSim.gravity = -2.8f; }
            ImGui::SameLine();
            if (ImGui::Button("Fire", {45,0})) { particleSim.emitterMode = 1; particleSim.gravity = -0.5f; }
            if (ImGui::Button("Explode", {60,0})) { particleSim.emitterMode = 2; particleSim.gravity = -1.0f; }
            ImGui::SameLine();
            if (ImGui::Button("Vortex", {55,0})) { particleSim.emitterMode = 3; particleSim.gravity = -0.8f; }
            ImGui::Spacing();

            ImGui::SliderFloat("Emit Rate", &particleSim.emitRate, 50.f, 1500.f);
            ImGui::SliderFloat("Gravity", &particleSim.gravity, -8.f, 2.f);
            ImGui::SliderFloat("Wind X", &particleSim.windX, -5.f, 5.f);
            ImGui::SliderFloat("Wind Z", &particleSim.windZ, -5.f, 5.f);
            ImGui::SliderFloat("Spread", &particleSim.spread, 0.f, 2.f);
            ImGui::SliderFloat("Speed", &particleSim.initialSpeed, 0.5f, 8.f);
            ImGui::SliderFloat("Size##p", &particleSim.particleSize, 1.f, 20.f);
            ImGui::Checkbox("Auto rotate##p", &particleSim.autoRotate);
            if (ImGui::Button("RESET##p")) particleSim.init();

        } else if (activeSim == ActiveSim::MANDELBROT) {
            ImGui::TextColored({0,1,0.255f,1}, "MANDELBROT EXPLORER");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Interactive fractal");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "zoom with drag/scroll");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Zoom: %.2e", mandelbrotSim.zoom);
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Center: %.6f, %.6f",
                mandelbrotSim.centerX, mandelbrotSim.centerY);

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::SliderInt("Max Iter", &mandelbrotSim.maxIter, 32, 1024);
            ImGui::SliderInt("Resolution", &mandelbrotRes, 200, 800);

            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "COLOR_SCHEME:");
            ImGui::RadioButton("Electric", &mandelbrotSim.colorScheme, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Fire##m", &mandelbrotSim.colorScheme, 1);
            ImGui::RadioButton("Ocean", &mandelbrotSim.colorScheme, 2);
            ImGui::SameLine();
            ImGui::RadioButton("CRT", &mandelbrotSim.colorScheme, 3);

            ImGui::Spacing();
            ImGui::Checkbox("Julia mode", &mandelbrotSim.julia);
            if (mandelbrotSim.julia) {
                ImGui::SliderFloat("Julia R", (float*)&mandelbrotSim.juliaR, -1.5f, 1.5f);
                ImGui::SliderFloat("Julia I", (float*)&mandelbrotSim.juliaI, -1.5f, 1.5f);
                ImGui::Checkbox("Animate Julia", &mandelbrotSim.animate);
                if (mandelbrotSim.animate)
                    ImGui::SliderFloat("Anim Speed", &mandelbrotSim.animSpeed, 0.1f, 2.f);
            }
            if (ImGui::Button("RESET##m")) mandelbrotSim.init();

        } else if (activeSim == ActiveSim::CLOTH) {
            ImGui::TextColored({0,1,0.255f,1}, "CLOTH SIMULATION");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Verlet integration");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "with spring constraints");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Verts: %d  Tris: %d",
                clothSim.totalVerts(), clothSim.totalTris());

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::SliderFloat("Gravity##c", &clothSim.gravity, -12.f, 0.f);
            ImGui::SliderFloat("Wind X##c", &clothSim.windStrX, -5.f, 5.f);
            ImGui::SliderFloat("Wind Z##c", &clothSim.windStrZ, -5.f, 5.f);
            ImGui::SliderFloat("Turbulence", &clothSim.windTurb, 0.f, 2.f);
            ImGui::SliderFloat("Damping", &clothSim.damping, 0.9f, 1.f);
            ImGui::SliderInt("Solver Iters", &clothSim.iterations, 1, 20);

            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "COLOR_MODE:");
            ImGui::RadioButton("Stress", &clothSim.colorMode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Normal", &clothSim.colorMode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Solid", &clothSim.colorMode, 2);
            ImGui::Checkbox("Wireframe", &clothSim.wireframe);
            ImGui::Checkbox("Auto rotate##c", &clothSim.autoRotate);

            if (ImGui::Button("Drop Corner")) clothSim.dropCorner();
            ImGui::SameLine();
            if (ImGui::Button("RESET##c")) clothSim.reset();

        } else if (activeSim == ActiveSim::BOIDS) {
            ImGui::TextColored({0,1,0.255f,1}, "BOIDS FLOCKING");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Reynolds' algorithm");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "emergent behavior");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Boids: %d", boidsSim.liveCount());

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::SliderFloat("Separation", &boidsSim.separation, 0.f, 5.f);
            ImGui::SliderFloat("Alignment", &boidsSim.alignment, 0.f, 3.f);
            ImGui::SliderFloat("Cohesion", &boidsSim.cohesion, 0.f, 3.f);
            ImGui::SliderFloat("Perception", &boidsSim.perception, 0.5f, 5.f);
            ImGui::SliderFloat("Max Speed", &boidsSim.maxSpeed, 1.f, 8.f);
            ImGui::SliderFloat("Size##b", &boidsSim.boidSize, 2.f, 12.f);
            ImGui::SliderFloat("Boundary", &boidsSim.boundSize, 2.f, 10.f);

            ImGui::Spacing();
            ImGui::Checkbox("Predator", &boidsSim.predatorOn);
            ImGui::Checkbox("Auto rotate##b", &boidsSim.autoRotate);
            if (ImGui::Button("RESET##b")) boidsSim.init();

        } else if (activeSim == ActiveSim::LORENZ) {
            ImGui::TextColored({0,1,0.255f,1}, "LORENZ ATTRACTOR");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Strange attractor,");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "chaos theory demo");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Trail pts: %d", lorenzSim.trailSize());

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::SliderFloat("Sigma", &lorenzSim.sigma, 1.f, 20.f);
            ImGui::SliderFloat("Rho", &lorenzSim.rho, 1.f, 50.f);
            ImGui::SliderFloat("Beta", &lorenzSim.beta, 0.1f, 8.f);
            ImGui::SliderFloat("Speed##lz", &lorenzSim.trailSpeed, 0.2f, 3.f);
            ImGui::SliderInt("Trail len", &lorenzSim.maxTrail, 1000, 15000);

            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "COLOR_MODE:");
            ImGui::RadioButton("Speed##lz", &lorenzSim.colorMode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Age", &lorenzSim.colorMode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Z-pos", &lorenzSim.colorMode, 2);

            ImGui::Checkbox("Show 2nd path", &lorenzSim.showSecond);
            ImGui::Checkbox("Pause##lz", &lorenzSim.paused);
            ImGui::Checkbox("Auto rotate##lz", &lorenzSim.autoRotate);
            if (ImGui::Button("RESET##lz")) lorenzSim.init();

        } else if (activeSim == ActiveSim::WAVE) {
            ImGui::TextColored({0,1,0.255f,1}, "WAVE EQUATION");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "2D wave simulation,");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "heightmap rendering");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Grid: %dx%d  Tris: %d",
                waveSim.gridSize, waveSim.gridSize, waveSim.totalTris());

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "SOURCE_MODE:");
            if (ImGui::Button("Drop", {45,0})) { waveSim.dropMode = 0; waveSim.singleDrop(); }
            ImGui::SameLine();
            if (ImGui::Button("Rain", {45,0})) waveSim.dropMode = 1;
            if (ImGui::Button("Dual Slit", {70,0})) waveSim.dropMode = 2;
            ImGui::SameLine();
            if (ImGui::Button("Ripple", {55,0})) waveSim.dropMode = 3;
            if (waveSim.dropMode == 0 && ImGui::Button("ADD DROP")) waveSim.singleDrop();
            ImGui::Spacing();

            ImGui::SliderFloat("Wave Spd", &waveSim.speed, 0.5f, 5.f);
            ImGui::SliderFloat("Damping##w", &waveSim.damping, 0.9f, 1.f);
            ImGui::SliderFloat("Drop Force", &waveSim.dropForce, 0.5f, 8.f);
            ImGui::SliderFloat("Height##w", &waveSim.heightScale, 0.2f, 3.f);

            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "COLOR_MODE:");
            ImGui::RadioButton("Height##wc", &waveSim.colorMode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Velocity##wc", &waveSim.colorMode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Grad", &waveSim.colorMode, 2);
            ImGui::Checkbox("Wireframe##w", &waveSim.wireframe);
            ImGui::Checkbox("Auto rotate##w", &waveSim.autoRotate);
            if (ImGui::Button("RESET##w")) waveSim.reset();

        } else if (activeSim == ActiveSim::PENDULUM) {
            ImGui::TextColored({0,1,0.255f,1}, "DOUBLE PENDULUM");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Chaotic motion,");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "RK4 integration");
            ImGui::Spacing();
            float E = pendSim.totalEnergy();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Energy: %.2f", E);
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Time: %.1fs", pendSim.time);

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::SliderFloat("L1", &pendSim.L1, 0.5f, 3.f);
            ImGui::SliderFloat("L2", &pendSim.L2, 0.5f, 3.f);
            ImGui::SliderFloat("m1", &pendSim.m1, 0.5f, 5.f);
            ImGui::SliderFloat("m2", &pendSim.m2, 0.5f, 5.f);
            ImGui::SliderFloat("Gravity##pd", &pendSim.g, 1.f, 20.f);
            ImGui::SliderFloat("Zoom##pd", &pendSim.zoom, 0.5f, 3.f);

            ImGui::Checkbox("Show trail##pd", &pendSim.showTrail);
            ImGui::Checkbox("Pause##pd", &pendSim.paused);
            if (ImGui::Button("RESET##pd")) pendSim.reset();

        } else if (activeSim == ActiveSim::GOL3D) {
            ImGui::TextColored({0,1,0.255f,1}, "3D GAME OF LIFE");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Cellular automata");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "on a 3D grid");
            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Gen: %d  Live: %d",
                golSim.generation, golSim.liveCells);

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "RULE_PRESET:");
            if (ImGui::Button("Clouds", {55,0})) { golSim.setRulePreset(0); golSim.randomize(); }
            ImGui::SameLine();
            if (ImGui::Button("Crystal", {60,0})) { golSim.setRulePreset(1); golSim.randomize(); }
            if (ImGui::Button("Amoeba", {55,0})) { golSim.setRulePreset(2); golSim.randomize(); }
            ImGui::SameLine();
            if (ImGui::Button("Builder", {60,0})) { golSim.setRulePreset(3); golSim.randomize(); }
            ImGui::Spacing();

            ImGui::SliderFloat("Step Int", &golSim.stepInterval, 0.05f, 1.f);
            ImGui::SliderFloat("Cell Size##g", &golSim.cellSize, 0.3f, 1.2f);
            ImGui::SliderFloat("Opacity##g", &golSim.opacity, 0.2f, 1.f);

            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "COLOR_MODE:");
            ImGui::RadioButton("Layer", &golSim.colorMode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Neighbors", &golSim.colorMode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Pos", &golSim.colorMode, 2);

            ImGui::Checkbox("Pause##g", &golSim.paused);
            ImGui::Checkbox("Auto rotate##g", &golSim.autoRotate);
            if (ImGui::Button("Step")) golSim.step();
            ImGui::SameLine();
            if (ImGui::Button("Randomize")) golSim.randomize();
            ImGui::SameLine();
            if (ImGui::Button("Clear")) golSim.clear();

        } else if (activeSim == ActiveSim::AUDIO) {
            ImGui::TextColored({0,1,0.255f,1}, "AUDIO VISUALIZER");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Real-time microphone");
            ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "FFT spectrum analysis");
            ImGui::Spacing();
            if (audioSim.isConnected()) {
                ImGui::TextColored({0,1,0.255f,1}, "MIC: CONNECTED");
                ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Volume: %.3f", audioSim.getVolume());
                ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Peak: %.0f Hz", audioSim.getPeakFreq());
            } else {
                ImGui::TextColored({1.f, 0.3f, 0.1f, 1}, "MIC: DISCONNECTED");
                ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "Check audio permissions");
            }

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0,0.478f,0.125f,1}, "\xe2\x96\xb8");
            ImGui::SameLine();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "CONTROLS");
            ImGui::Separator(); ImGui::Spacing();

            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "VIZ_MODE:");
            ImGui::RadioButton("Bars+Wave", &audioSim.vizMode, 0);
            ImGui::RadioButton("Circular", &audioSim.vizMode, 1);
            ImGui::RadioButton("Spectrogram", &audioSim.vizMode, 2);

            ImGui::Spacing();
            ImGui::SliderFloat("Sensitivity", &audioSim.sensitivity, 0.5f, 5.f);
            ImGui::SliderFloat("Smoothing", &audioSim.smoothing, 0.5f, 0.98f);
            ImGui::SliderFloat("Wave Scale", &audioSim.waveScale, 0.5f, 3.f);
            ImGui::SliderInt("Bars##au", &audioSim.numBars, 16, 128);

            ImGui::Spacing();
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "COLOR:");
            ImGui::RadioButton("Terminal", &audioSim.colorMode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Rainbow", &audioSim.colorMode, 1);
            ImGui::RadioButton("Cyan", &audioSim.colorMode, 2);

            if (audioSim.vizMode == 1) {
                ImGui::SliderFloat("Radius", &audioSim.circleRadius, 0.1f, 0.5f);
                ImGui::SliderFloat("Scale##cr", &audioSim.circleScale, 0.1f, 1.f);
            }

        } else {
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "Select a simulation");
            ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "from the explorer above");
        }

        ImGui::End();

        // ================================================================
        // BOTTOM STATUS BAR
        // ================================================================
        ImGui::SetNextWindowPos({0, (float)winH - botH});
        ImGui::SetNextWindowSize({(float)winW, botH});
        ImGui::Begin("##bottombar", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoCollapse);

        const char* simName = "NONE";
        const char* status  = "IDLE";
        switch (activeSim) {
            case ActiveSim::HYPERCUBE_4D:
                simName = "4D_HYPERCUBE"; status = "RUNNING"; break;
            case ActiveSim::NEURAL_VIZ:
                simName = "NEURAL_VIZ";
                status = neuralSim.inferActive ? "INFERRING" : "READY"; break;
            case ActiveSim::PARTICLES:
                simName = "PARTICLES"; status = "RUNNING"; break;
            case ActiveSim::MANDELBROT:
                simName = "MANDELBROT"; status = "RENDERING"; break;
            case ActiveSim::CLOTH:
                simName = "CLOTH_SIM"; status = "SIMULATING"; break;
            case ActiveSim::BOIDS:
                simName = "BOIDS"; status = "FLOCKING"; break;
            case ActiveSim::LORENZ:
                simName = "LORENZ"; status = "TRACING"; break;
            case ActiveSim::WAVE:
                simName = "WAVE_SIM"; status = "PROPAGATING"; break;
            case ActiveSim::PENDULUM:
                simName = "PENDULUM"; status = pendSim.paused ? "PAUSED" : "SWINGING"; break;
            case ActiveSim::GOL3D:
                simName = "GOL_3D"; status = golSim.paused ? "PAUSED" : "EVOLVING"; break;
            case ActiveSim::AUDIO:
                simName = "AUDIO_VIZ"; status = audioSim.isConnected() ? "LISTENING" : "NO MIC"; break;
            default: break;
        }

        float pulse = 0.5f + 0.5f * std::sin((float)glfwGetTime() * 2.f);
        ImGui::TextColored({0, pulse*0.478f+0.5f, pulse*0.125f, 1}, "●");
        ImGui::SameLine();
        ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "STATUS: %s", status);
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, " | ");
        ImGui::SameLine();
        ImGui::TextColored({0,0.898f,1,1}, "SIM: %s", simName);
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, " | ");
        ImGui::SameLine();
        ImGui::TextColored({0,1,0.255f,1}, "FPS: %d", fps);
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, " | ");
        ImGui::SameLine();
        ImGui::TextColored({0.478f, 0.749f, 0.533f, 1}, "MODE: INTERACTIVE");
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, " | ");
        ImGui::SameLine();
        ImGui::TextColored({0.239f, 0.42f, 0.278f, 1}, "DRAG=ROTATE  SCROLL=ZOOM");

        ImGui::End();

        // ================================================================
        // RENDER SIM VIEWPORT
        // ================================================================
        float vpX = sideW;
        float vpY = botH;      // bottom bar height (flipped in GL)
        float vpW = winW - sideW;
        float vpH = mainH;

        // Clear the entire window
        glViewport(0, 0, winW, winH);
        glClearColor(0.024f, 0.039f, 0.027f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render active simulation into the right viewport
        switch (activeSim) {
            case ActiveSim::HYPERCUBE_4D:
                render4D(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::NEURAL_VIZ:
                renderNeural(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::PARTICLES:
                renderParticles(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::MANDELBROT:
                renderMandelbrot(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::CLOTH:
                renderCloth(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::BOIDS:
                renderBoids(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::LORENZ:
                renderLorenz(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::WAVE:
                renderWave(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::PENDULUM:
                renderPendulum(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::GOL3D:
                renderGol3d(dt, vpX, vpY, vpW, vpH); break;
            case ActiveSim::AUDIO:
                renderAudio(dt, vpX, vpY, vpW, vpH); break;
            default: break;
        }

        // Render ImGui on top
        ImGui::Render();
        glViewport(0, 0, winW, winH);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
    }

    // Cleanup
    audioSim.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
