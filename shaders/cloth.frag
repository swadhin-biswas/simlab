#version 460 core

in vec3 vNormal;
in vec3 vColor;
in vec3 vWorldPos;
in float vViewDepth;

out vec4 FragColor;

void main() {
    // Simple directional lighting
    vec3 lightDir = normalize(vec3(0.3, 0.8, 0.5));
    vec3 norm = normalize(vNormal);

    // Two-sided lighting
    float diff = abs(dot(norm, lightDir));
    float ambient = 0.15;
    float light = ambient + diff * 0.85;

    // Subtle rim lighting
    vec3 viewDir = normalize(-vWorldPos);
    float rim = 1.0 - abs(dot(norm, viewDir));
    rim = pow(rim, 3.0) * 0.3;

    vec3 color = vColor * light + vec3(0.0, 0.5, 1.0) * rim;

    // Depth fog
    float fog = clamp(1.0 / (1.0 + vViewDepth * 0.05), 0.3, 1.0);
    color *= fog;

    FragColor = vec4(color, 0.92);
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
