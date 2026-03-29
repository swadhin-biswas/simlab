#version 460 core

in float vW;
in float vLabel;
in float vViewDepth;

uniform float uAlpha;
uniform int   uColorMode;   // 0=w-depth, 1=class, 2=z-depth
uniform int   uNumClasses;

out vec4 FragColor;

// ----------------------------------------------------------------
// Palette: 8 visually distinct colors for class labels
// ----------------------------------------------------------------
vec3 classColor(int idx) {
    // purple, amber, teal, coral, blue, green, pink, gray
    vec3 palette[8] = vec3[8](
        vec3(0.325, 0.290, 0.718),   // purple 400
        vec3(0.729, 0.459, 0.090),   // amber 400
        vec3(0.114, 0.620, 0.459),   // teal 400
        vec3(0.847, 0.353, 0.188),   // coral 400
        vec3(0.216, 0.541, 0.867),   // blue 400
        vec3(0.388, 0.600, 0.133),   // green 400
        vec3(0.831, 0.325, 0.494),   // pink 400
        vec3(0.533, 0.529, 0.502)    // gray 400
    );
    return palette[clamp(idx, 0, 7)];
}

// ----------------------------------------------------------------
// W-depth color: purple (w=-1) → mid gray (w=0) → amber (w=+1)
// ----------------------------------------------------------------
vec3 wColor(float w) {
    float t = clamp(w * 0.5 + 0.5, 0.0, 1.0);
    vec3 cool = vec3(0.325, 0.290, 0.718);  // purple 400
    vec3 mid  = vec3(0.533, 0.529, 0.502);  // gray   400
    vec3 warm = vec3(0.729, 0.459, 0.090);  // amber  400
    return t < 0.5 ? mix(cool, mid, t*2.0) : mix(mid, warm, (t-0.5)*2.0);
}

// ----------------------------------------------------------------
// Z-depth color: blue (near) → red (far)
// ----------------------------------------------------------------
vec3 depthColor(float d) {
    float t = clamp(d / 8.0, 0.0, 1.0);
    return mix(vec3(0.216,0.541,0.867), vec3(0.886,0.294,0.290), t);
}

void main() {
    // ---- Circular point sprite ----
    vec2  uv   = gl_PointCoord * 2.0 - 1.0;
    float dist = dot(uv, uv);
    if (dist > 1.0) discard;

    // Soft anti-aliased edge + depth fade
    float edgeFade  = 1.0 - smoothstep(0.6, 1.0, dist);
    float depthFade = clamp(1.0 / (1.0 + vViewDepth * 0.2), 0.35, 1.0);
    float alpha     = uAlpha * edgeFade * depthFade;

    // ---- Choose color by mode ----
    vec3 color;
    if (uColorMode == 0)      color = wColor(vW);
    else if (uColorMode == 1) color = classColor(int(vLabel) % 8);
    else                      color = depthColor(vViewDepth);

    FragColor = vec4(color, alpha);
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
