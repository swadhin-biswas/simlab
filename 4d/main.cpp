#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "hypercube4d.hpp"
#include "shader.hpp"

#include <vector>
#include <iostream>

// ============================================================
// Window config
// ============================================================
static constexpr int WIN_W = 1440;
static constexpr int WIN_H = 900;

// ============================================================
// Camera state (free-fly, WASD + mouse)
// ============================================================
static glm::vec3 camPos   = {0.f, 0.f, 4.f};
static glm::vec3 camFront = {0.f, 0.f, -1.f};
static glm::vec3 camUp    = {0.f, 1.f,  0.f};
static float yaw = -90.f, pitch = 0.f;
static float lastX = WIN_W / 2.f, lastY = WIN_H / 2.f;
static bool  firstMouse = true;
static bool  captureMouse = false;

// ============================================================
// 4D state
// ============================================================
static hd::Rotation4D rot4d;
static hd::Dataset4D  dataset;

static float wDist    = 2.0f;   // 4D perspective distance (user-tunable)
static float ptSize   = 8.0f;
static float ptAlpha  = 0.85f;
static bool  showEdges = false;
static int   colorMode = 0;     // 0=by-w, 1=by-class, 2=by-depth

// ============================================================
// GPU buffers
// ============================================================
static GLuint vao, vbo;
static GLuint edgeVao, edgeVbo, edgeEbo;

// ============================================================
// Callbacks
// ============================================================
void mouse_callback(GLFWwindow* win, double xpos, double ypos) {
    if (!captureMouse) return;
    if (firstMouse) { lastX=(float)xpos; lastY=(float)ypos; firstMouse=false; }
    float dx = (float)xpos - lastX, dy = lastY - (float)ypos;
    lastX=(float)xpos; lastY=(float)ypos;
    yaw   += dx * 0.15f;
    pitch  = glm::clamp(pitch + dy * 0.15f, -89.f, 89.f);
    camFront = glm::normalize(glm::vec3(
        std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch)),
        std::sin(glm::radians(pitch)),
        std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch))
    ));
}

void scroll_callback(GLFWwindow*, double, double yoffset) {
    camPos += camFront * (float)yoffset * 0.25f;
}

void key_callback(GLFWwindow* win, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        captureMouse = !captureMouse;
        glfwSetInputMode(win, GLFW_CURSOR,
            captureMouse ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        firstMouse = true;
    }
}

void processMovement(GLFWwindow* win, float dt) {
    if (!captureMouse) return;
    float spd = 2.0f * dt;
    if (glfwGetKey(win,GLFW_KEY_W)==GLFW_PRESS) camPos += spd * camFront;
    if (glfwGetKey(win,GLFW_KEY_S)==GLFW_PRESS) camPos -= spd * camFront;
    if (glfwGetKey(win,GLFW_KEY_A)==GLFW_PRESS) camPos -= glm::normalize(glm::cross(camFront,camUp))*spd;
    if (glfwGetKey(win,GLFW_KEY_D)==GLFW_PRESS) camPos += glm::normalize(glm::cross(camFront,camUp))*spd;
}

// ============================================================
// Build edge index buffer for tesseract (or any point dataset)
// ============================================================
std::vector<unsigned int> buildTesseractEdges() {
    std::vector<unsigned int> idx;
    for (int i=0;i<16;i++) for (int j=i+1;j<16;j++) {
        int d = i^j;
        if (d && !(d&(d-1))) { idx.push_back(i); idx.push_back(j); }
    }
    return idx;
}

// ============================================================
// Main
// ============================================================
int main() {
    // --- GLFW / OpenGL init ---
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "4D Hypercube Data Explorer", nullptr, nullptr);
    if (!win) { std::cerr << "GLFW window creation failed\n"; return -1; }

    glfwMakeContextCurrent(win);
    glfwSetCursorPosCallback(win, mouse_callback);
    glfwSetScrollCallback(win, scroll_callback);
    glfwSetKeyCallback(win, key_callback);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n"; return -1;
    }

    // --- ImGui ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    // --- Load dataset (swap for your real data) ---
    dataset = hd::Dataset4D::tesseract();
    dataset.normalize();

    // --- Point cloud VBO ---
    // Layout per vertex: vec3 pos_3d  +  float w_value  +  float label
    //                    (projected live every frame on CPU, uploaded via SubData)
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
        dataset.points.size() * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(4*sizeof(float)));
    glEnableVertexAttribArray(2);

    // --- Edge IBO (tesseract only; skip for general point clouds) ---
    auto edgeIdx = buildTesseractEdges();
    glGenVertexArrays(1, &edgeVao);
    glGenBuffers(1, &edgeVbo);
    GLuint edgeEbo;
    glGenBuffers(1, &edgeEbo);
    glBindVertexArray(edgeVao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);        // share the same VBO
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        edgeIdx.size()*sizeof(unsigned int), edgeIdx.data(), GL_STATIC_DRAW);

    // --- Shaders ---
    Shader ptShader ("shaders/point.vert", "shaders/point.frag");
    Shader edgeShader("shaders/edge.vert",  "shaders/edge.frag");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float  dt  = (float)(now - lastTime); lastTime = now;

        processMovement(win, dt);

        // ---- 1. Update 4D rotation ----
        rot4d.update(dt);
        auto mat4 = rot4d.matrix();

        // ---- 2. Project all points: 4D → 3D on CPU, upload to GPU ----
        std::vector<float> gpuData;
        gpuData.reserve(dataset.points.size() * 5);
        for (size_t i = 0; i < dataset.points.size(); i++) {
            hd::Vec4 r = mat4 * dataset.points[i];     // rotate in 4D
            glm::vec4 p = hd::project4Dto3D(r, wDist); // project to 3D
            gpuData.insert(gpuData.end(), {
                p.x, p.y, p.z,
                p.w,                                    // w for color
                (float)(i < dataset.labels.size() ? dataset.labels[i] : 0)
            });
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
            gpuData.size()*sizeof(float), gpuData.data());

        // ---- 3. Render ----
        glClearColor(0.04f, 0.04f, 0.07f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = glm::lookAt(camPos, camPos+camFront, camUp);
        glm::mat4 proj = glm::perspective(
            glm::radians(45.f), (float)WIN_W/WIN_H, 0.01f, 100.f);

        // Draw edges
        if (showEdges) {
            edgeShader.use();
            edgeShader.setMat4("uView", view);
            edgeShader.setMat4("uProj", proj);
            edgeShader.setFloat("uAlpha", 0.35f);
            edgeShader.setFloat("uWNorm", 1.f/wDist);
            glBindVertexArray(edgeVao);
            glDrawElements(GL_LINES, (GLsizei)edgeIdx.size(), GL_UNSIGNED_INT, 0);
        }

        // Draw points
        ptShader.use();
        ptShader.setMat4("uView",      view);
        ptShader.setMat4("uProj",      proj);
        ptShader.setFloat("uPointSize", ptSize);
        ptShader.setFloat("uAlpha",     ptAlpha);
        ptShader.setFloat("uWNorm",     1.f / wDist);
        ptShader.setInt("uColorMode",   colorMode);
        ptShader.setInt("uNumClasses",  std::max(1, dataset.numClasses()));
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, (GLsizei)dataset.points.size());

        // ---- 4. ImGui panel ----
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos({20,20}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({300,420}, ImGuiCond_Once);
        ImGui::Begin("4D Explorer Controls");

        ImGui::Text("Camera: WASD + mouse (ESC toggles)");
        ImGui::Separator();

        ImGui::SliderFloat("W-distance",  &wDist,   1.2f, 6.0f);
        ImGui::SliderFloat("Point size",  &ptSize,  2.0f, 24.0f);
        ImGui::SliderFloat("Opacity",     &ptAlpha, 0.1f, 1.0f);
        ImGui::Checkbox("Show edges",     &showEdges);

        ImGui::Separator();
        ImGui::Text("Color mode");
        ImGui::RadioButton("W-depth",  &colorMode, 0); ImGui::SameLine();
        ImGui::RadioButton("Class",    &colorMode, 1); ImGui::SameLine();
        ImGui::RadioButton("Z-depth",  &colorMode, 2);

        ImGui::Separator();
        ImGui::Text("Rotation planes");
        for (int p=0; p<hd::NUM_PLANES; p++) {
            ImGui::Checkbox(hd::PLANE_NAME[p], &rot4d.active[p]);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::SliderFloat(
                ("##s"+std::to_string(p)).c_str(),
                &rot4d.speeds[p], 0.f, 1.5f);
        }

        ImGui::Separator();
        if (ImGui::Button("Load tesseract"))
            dataset = hd::Dataset4D::tesseract();
        ImGui::SameLine();
        if (ImGui::Button("Klein bottle"))
            dataset = hd::Dataset4D::kleinBottle();
        ImGui::SameLine();
        if (ImGui::Button("Hopf fibration"))
            dataset = hd::Dataset4D::hopfFibration();

        ImGui::Text("Points: %zu", dataset.points.size());
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
