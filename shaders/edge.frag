#version 460 core

in float vW;

uniform float uAlpha;

out vec4 FragColor;

void main() {
    float t = clamp(vW * 0.5 + 0.5, 0.0, 1.0);
    vec3 cool = vec3(0.325, 0.290, 0.718);
    vec3 warm = vec3(0.729, 0.459, 0.090);
    vec3 color = mix(cool, warm, t);
    FragColor = vec4(color, uAlpha);
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
