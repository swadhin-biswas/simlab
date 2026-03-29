#version 460 core

// ----------------------------------------------------------------
// Inputs (matches VBO layout set in main.cpp)
// ----------------------------------------------------------------
layout(location = 0) in vec3  aPos;    // 3D position after 4D->3D projection (CPU-side)
layout(location = 1) in float aW;      // original w value (for color + size)
layout(location = 2) in float aLabel;  // class label (float, cast to int in frag)

// ----------------------------------------------------------------
// Uniforms
// ----------------------------------------------------------------
uniform mat4  uView;
uniform mat4  uProj;
uniform float uPointSize;  // base point size (px)
uniform float uWNorm;      // 1/wDist — normalizes w to [-1,1] range
uniform float uAlpha;

// ----------------------------------------------------------------
// Outputs to fragment shader
// ----------------------------------------------------------------
out float vW;
out float vLabel;
out float vViewDepth;

void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0);
    gl_Position  = uProj * viewPos;

    // ---- Point size ----
    // Scale by:  (1) view distance so far points appear smaller
    //            (2) w-value so "foreground" 4D points are more prominent
    float viewDist   = -viewPos.z;
    float distScale  = uPointSize / max(viewDist, 0.5);
    float wEmphasis  = 0.5 + 0.5 * abs(aW * uWNorm);   // 0.5 → 1.0
    gl_PointSize = distScale * wEmphasis * 60.0;         // 60 = tuning constant

    vW         = aW * uWNorm;   // normalize to ~[-1, 1]
    vLabel     = aLabel;
    vViewDepth = viewDist;
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
