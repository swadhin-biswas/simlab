cmake_minimum_required(VERSION 3.20)
project(SimLabTUI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm REQUIRED)
find_package(GLEW REQUIRED)

# ImGui — fetched automatically
include(FetchContent)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.4
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)

# ---------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------
add_executable(simlab_tui
    src/main.cpp
    src/hypercube4d.cpp
)

target_include_directories(simlab_tui PRIVATE
    src
)

target_link_libraries(simlab_tui PRIVATE
    OpenGL::GL
    glfw
    GLEW::GLEW
    imgui_lib
    pulse-simple
    pulse
    pthread
)

# ---------------------------------------------------------------
# Copy shaders next to the executable after build
# ---------------------------------------------------------------
add_custom_command(TARGET simlab_tui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:simlab_tui>/shaders"
)

# ---------------------------------------------------------------
# Build instructions:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#   ./simlab_tui
# ---------------------------------------------------------------
