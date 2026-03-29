#version 460 core

in vec4 vColor;
out vec4 FragColor;

uniform int uIsPoint; // 1 = render as circle, 0 = line

void main() {
    if (uIsPoint == 1) {
        vec2 uv = gl_PointCoord * 2.0 - 1.0;
        float dist = dot(uv, uv);
        if (dist > 1.0) discard;
        float edge = 1.0 - smoothstep(0.5, 1.0, dist);
        FragColor = vec4(vColor.rgb, vColor.a * edge);
    } else {
        FragColor = vColor;
    }
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
