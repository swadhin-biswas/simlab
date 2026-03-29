#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace gol3d {

struct GameOfLife3D {
    int gridSize = 16;  // 16x16x16 cube
    std::vector<bool> cells;
    std::vector<bool> nextCells;

    float time = 0.f;
    float stepInterval = 0.35f; // seconds between generations
    float stepTimer = 0.f;
    int   generation = 0;
    int   liveCells  = 0;
    bool  paused = false;

    // Rules: survive if neighbors in [sMin, sMax], born if neighbors in [bMin, bMax]
    int sMin = 5, sMax = 7;    // 5-7 neighbors to survive
    int bMin = 6, bMax = 6;    // exactly 6 neighbors to be born
    int rulePreset = 0;

    // Camera
    float camYaw   = 0.4f;
    float camPitch = 0.35f;
    float camDist  = 30.f;
    bool  autoRotate = true;

    // Display
    float cellSize = 0.6f;
    float opacity  = 0.7f;
    int   colorMode = 0; // 0=layer, 1=neighbors, 2=position

    std::mt19937 rng{42};

    int idx(int x, int y, int z) const {
        return z * gridSize * gridSize + y * gridSize + x;
    }

    void init() {
        int n = gridSize;
        cells.assign(n*n*n, false);
        nextCells.assign(n*n*n, false);
        generation = 0;
        time = 0.f;
        stepTimer = 0.f;

        // Random initial state ~15% density
        std::uniform_real_distribution<float> d(0.f, 1.f);
        for (int z = 0; z < n; z++)
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                    cells[idx(x,y,z)] = d(rng) < 0.15f;

        countLive();
    }

    void setRulePreset(int preset) {
        rulePreset = preset;
        switch (preset) {
            case 0: sMin=5; sMax=7; bMin=6; bMax=6; break;     // "Clouds"
            case 1: sMin=4; sMax=5; bMin=5; bMax=5; break;     // "Crystal"
            case 2: sMin=2; sMax=6; bMin=4; bMax=6; break;     // "Amoeba"
            case 3: sMin=9; sMax=26; bMin=5; bMax=8; break;    // "Builder"
        }
    }

    int countNeighbors(int cx, int cy, int cz) const {
        int n = gridSize;
        int count = 0;
        for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    int nx = cx+dx, ny = cy+dy, nz = cz+dz;
                    // Wrap around
                    if (nx < 0) nx += n; if (nx >= n) nx -= n;
                    if (ny < 0) ny += n; if (ny >= n) ny -= n;
                    if (nz < 0) nz += n; if (nz >= n) nz -= n;
                    if (cells[idx(nx,ny,nz)]) count++;
                }
        return count;
    }

    void step() {
        int n = gridSize;
        for (int z = 0; z < n; z++)
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++) {
                    int nb = countNeighbors(x, y, z);
                    int i = idx(x, y, z);
                    if (cells[i])
                        nextCells[i] = (nb >= sMin && nb <= sMax);
                    else
                        nextCells[i] = (nb >= bMin && nb <= bMax);
                }
        std::swap(cells, nextCells);
        generation++;
        countLive();
    }

    void countLive() {
        liveCells = 0;
        for (bool c : cells) if (c) liveCells++;
    }

    void update(float dt) {
        time += dt;
        if (autoRotate) camYaw += dt * 0.12f;

        if (!paused) {
            stepTimer += dt;
            if (stepTimer >= stepInterval) {
                step();
                stepTimer -= stepInterval;
            }
        }
    }

    void randomize() {
        std::uniform_real_distribution<float> d(0.f, 1.f);
        int n = gridSize;
        for (int i = 0; i < n*n*n; i++)
            cells[i] = d(rng) < 0.15f;
        generation = 0;
        countLive();
    }

    void clear() {
        std::fill(cells.begin(), cells.end(), false);
        generation = 0;
        liveCells = 0;
    }
};

} // namespace gol3d

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
