cmake_minimum_required(VERSION 3.20)
project(Hypercube4D LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------
find_package(OpenGL REQUIRED)

# GLFW — install with: brew install glfw  /  sudo apt install libglfw3-dev
find_package(glfw3 3.3 REQUIRED)

# GLM — header-only: brew install glm  /  sudo apt install libglm-dev
find_package(glm REQUIRED)

# ImGui — clone into vendor/imgui or use FetchContent
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
# GLAD (put glad.c + glad/gl.h into vendor/glad/)
# ---------------------------------------------------------------
add_library(glad STATIC vendor/glad/src/glad.c)
target_include_directories(glad PUBLIC vendor/glad/include)

# ---------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------
add_executable(hypercube4d
    src/main.cpp
    src/hypercube4d.cpp
    src/shader.cpp           # standard file-based shader loader (not included here)
)

target_include_directories(hypercube4d PRIVATE
    include
    vendor/glad/include
    ${GLM_INCLUDE_DIRS}
)

target_link_libraries(hypercube4d PRIVATE
    OpenGL::GL
    glfw
    glad
    imgui_lib
)

# ---------------------------------------------------------------
# Copy shaders next to the executable after build
# ---------------------------------------------------------------
add_custom_command(TARGET hypercube4d POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:hypercube4d>/shaders"
)

# ---------------------------------------------------------------
# Build instructions:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#   ./hypercube4d
# ---------------------------------------------------------------
