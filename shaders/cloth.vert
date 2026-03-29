#version 460 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vColor;
out vec3 vWorldPos;
out float vViewDepth;

void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0);
    gl_Position = uProj * viewPos;
    vNormal = aNormal;
    vColor = aColor;
    vWorldPos = aPos;
    vViewDepth = -viewPos.z;
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
