#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace mandelbrot {

struct MandelbrotSim {
    // View parameters
    double centerX  = -0.5;
    double centerY  =  0.0;
    double zoom     =  0.35;  // scale: smaller = more zoomed in

    int maxIter     = 256;
    int colorScheme = 0;  // 0=electric, 1=fire, 2=ocean, 3=grayscale
    bool julia      = false;
    double juliaR   = -0.7;
    double juliaI   =  0.27015;
    bool animate    = false;
    float time      = 0.f;
    float animSpeed = 0.3f;

    // CPU pixel buffer for the fractal
    int bufW = 0, bufH = 0;
    std::vector<unsigned char> pixels; // RGBA

    void init() {
        centerX = -0.5;
        centerY = 0.0;
        zoom    = 0.35;
        time    = 0.f;
    }

    void update(float dt) {
        if (animate) {
            time += dt * animSpeed;
            juliaR = -0.7 + 0.15 * std::sin(time * 0.8);
            juliaI =  0.27015 + 0.1 * std::cos(time * 1.1);
        }
    }

    void pan(double dx, double dy) {
        centerX += dx * zoom;
        centerY += dy * zoom;
    }

    void zoomIn(double factor, double mouseNX = 0.5, double mouseNY = 0.5) {
        // Zoom towards mouse position
        double worldX = centerX + (mouseNX - 0.5) * zoom * 2.0;
        double worldY = centerY + (mouseNY - 0.5) * zoom * 2.0;
        zoom *= factor;
        centerX = worldX - (mouseNX - 0.5) * zoom * 2.0;
        centerY = worldY - (mouseNY - 0.5) * zoom * 2.0;
    }

    // Map iteration count to color
    void iterToColor(int iter, int maxIt, float& r, float& g, float& b) const {
        if (iter >= maxIt) { r = g = b = 0.f; return; }

        float t = (float)iter / maxIt;
        float s = std::sqrt(t); // sqrt for better distribution

        switch (colorScheme) {
            case 0: // Electric (purple → cyan → white)
                r = 0.5f + 0.5f * std::sin(s * 12.f + 0.0f);
                g = 0.5f + 0.5f * std::sin(s * 12.f + 2.1f);
                b = 0.5f + 0.5f * std::sin(s * 12.f + 4.2f);
                break;
            case 1: // Fire (black → red → yellow → white)
                r = std::clamp(s * 3.f, 0.f, 1.f);
                g = std::clamp(s * 3.f - 1.f, 0.f, 1.f);
                b = std::clamp(s * 3.f - 2.f, 0.f, 1.f);
                break;
            case 2: // Ocean (deep blue → teal → white)
                r = s * s;
                g = s;
                b = std::sqrt(s) * 0.7f + 0.3f;
                break;
            case 3: // CRT Green (terminal aesthetic)
                r = 0.f;
                g = s;
                b = s * 0.25f;
                break;
        }
    }

    // Render fractal to CPU pixel buffer
    void render(int width, int height) {
        if (width != bufW || height != bufH) {
            bufW = width; bufH = height;
            pixels.resize(width * height * 4);
        }

        double aspect = (double)width / height;
        double xMin = centerX - zoom * aspect;
        double xMax = centerX + zoom * aspect;
        double yMin = centerY - zoom;
        double yMax = centerY + zoom;

        for (int py = 0; py < height; py++) {
            for (int px = 0; px < width; px++) {
                double x0 = xMin + (xMax - xMin) * px / width;
                double y0 = yMin + (yMax - yMin) * py / height;

                double zr, zi, cr, ci;
                if (julia) {
                    zr = x0; zi = y0;
                    cr = juliaR; ci = juliaI;
                } else {
                    zr = 0; zi = 0;
                    cr = x0; ci = y0;
                }

                int iter = 0;
                while (zr*zr + zi*zi < 4.0 && iter < maxIter) {
                    double tmp = zr*zr - zi*zi + cr;
                    zi = 2.0 * zr * zi + ci;
                    zr = tmp;
                    iter++;
                }

                float r, g, b;
                iterToColor(iter, maxIter, r, g, b);

                int idx = (py * width + px) * 4;
                pixels[idx+0] = (unsigned char)(r * 255);
                pixels[idx+1] = (unsigned char)(g * 255);
                pixels[idx+2] = (unsigned char)(b * 255);
                pixels[idx+3] = 255;
            }
        }
    }
};

} // namespace mandelbrot

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
