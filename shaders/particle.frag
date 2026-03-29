#version 460 core

in vec4 vColor;
out vec4 FragColor;

void main() {
    // Smooth circular point with glow
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float dist = dot(uv, uv);
    if (dist > 1.0) discard;

    float core = 1.0 - smoothstep(0.0, 0.4, dist);
    float glow = 1.0 - smoothstep(0.2, 1.0, dist);

    vec3 color = vColor.rgb * (0.6 + 0.4 * core);
    // Additive glow
    color += vColor.rgb * 0.3 * glow;

    float alpha = vColor.a * glow;
    FragColor = vec4(color, alpha);
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
