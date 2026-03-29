#version 460 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in float aW;

uniform mat4 uView;
uniform mat4 uProj;
uniform float uWNorm;

out float vW;

void main() {
    gl_Position = uProj * uView * vec4(aPos, 1.0);
    vW = aW * uWNorm;
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
