<div align="center">

# 🌌 SimLab TUI: Interactive Computer Graphics Project
**High-Performance C++ & OpenGL Graphics Simulation Engine**

![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg?style=for-the-badge)
![C++20](https://img.shields.io/badge/std-c%2B%2B20-orange.svg?style=for-the-badge)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3+-green.svg?style=for-the-badge&logo=opengl)
![ImGui](https://img.shields.io/badge/UI-ImGui-red.svg?style=for-the-badge)

*SimLab TUI is a feature-rich **Computer Graphics Project** built natively in C++ using OpenGL. It features a sleek, terminal-inspired ImGui browser interface to run real-time physics, math, and generative simulations.*

[Explore Simulations](#-showcase--included-simulations) •
[Installation](#-building--running) •
[Controls](#%EF%B8%8F-controls--navigation)

</div>

SimLab TUI allows real-time execution and deep interaction with **11 distinct mathematical, physical, and generative computer graphics simulations**, all running seamlessly within a single shared OpenGL context. Whether you are searching for a **C++ Physics Engine**, a **Fractal Renderer**, or a comprehensive **Computer Graphics Simulation Engine**, this project provides a unified, developer-friendly interface to explore chaotic systems, cellular automata, and advanced 3D rendering.

---

## 📑 Table of Contents
1. [🌟 Showcase & Included Simulations](#-showcase--included-simulations)
2. [🔜 Coming Soon](#-coming-soon)
3. [⚙️ Requirements](#%EF%B8%8F-requirements)
4. [🚀 Building & Running](#-building--running)
    - [Linux Instructions](#-linux-arch--ubuntu--debian)
    - [macOS Instructions](#-macos)
5. [🕹️ Controls & Navigation](#%EF%B8%8F-controls--navigation)
6. [👨‍💻 Developer & Maintainer](#-developer--maintainer)

---

## 📸 Showcase & Included Simulations

Below are the **11 interactive graphics simulations** currently bundled in the engine. 

<details open>
<summary><b>📐 Mathematics & Higher Geometry</b></summary>

### 1. 🎲 4D Hypercube
Higher-dimensional geometry projection with controllable rotation planes in 4D space rendered down to 3D.

![4D Hypercube](images/4D.png)

### 2. 🌀 Mandelbrot Explorer
Interactive fractal rendering with deep zoom, pan capability, Julia set toggling, and vibrant dynamic color profiles.

![Mandelbrot Explorer](images/fractals.png)
</details>

<details open>
<summary><b>⚛️ Physics & Dynamics</b></summary>

### 3. ✨ Particle System
GPU-accelerated particle engine with 4 distinct emitter modes, tunable gravity, wind vectors, and precise life-cycling.

### 4. 🚩 Cloth Simulation
Verlet-integration based physics with structural constraints, wind turbulence, and dynamic stress visualization.

![Cloth Simulation](images/clothsim.png)

### 5. ⏳ Double Pendulum
RK4-integrated chaotic dynamics system displaying a breathtaking, ever-changing orbital trail history.

![Double Pendulum](images/doublepenulam.png)

### 6. 💧 Wave Equation
2D Partial Differential Equation simulated on a dynamic heightmap mesh with rain, single drops, and slit source modes.

### 7. 🌪️ Lorenz Attractor
Strange attractors and chaos theory visualization mapped vividly with 3D trail rendering.
</details>

<details open>
<summary><b>🧬 Generative & AI Vis</b></summary>

### 8. 🧠 Neural Network Viz
3D visualization and animation state-machine of a neural network inferencing process traversing its hidden layers.

![Neural Network Viz](images/nuralnet.png)

### 9. 🦅 Boids Flocking
Reynolds' artificial life algorithm demonstrating emergent separation, alignment, and cohesion across hundreds of entities.

![Boids Flocking](images/boids.png)

### 10. ⬛ Game of Life 3D
Cellular automata logic computed across a three-dimensional grid with fully adjustable survival and birth rules.

![Game of Life 3D](images/gameoflife.png)

### 11. 🎵 Audio Visualizer
Real-time microphone capture driving FFT spectrum bars, circular audio graphs, and spectrogram rendering modes.
</details>

---

## 🔜 Coming Soon
Development is highly active! Expect frequent updates as more exciting mathematical models, advanced rendering techniques, and robust physics simulations are continually being added to the engine.

---

## ⚙️ Requirements

Before compiling this **Computer Graphics Project**, ensure your system meets the following standard dependencies:

*   **Compiler:** GCC / Clang / Apple Clang (Must support standard **C++20**)
*   **Build System:** CMake 3.20 or newer
*   **Libraries:** OpenGL, GLFW3, GLEW, GLM, PulseAudio (`libpulse`)
*   *Note: [ImGui](https://github.com/ocornut/imgui) used for the UI is downloaded and linked automatically by CMake during the build process via `FetchContent`.*

---

## 🚀 Building & Running

<details>
<summary><b>🐧 Linux (Arch / Ubuntu / Debian)</b></summary>

**1. Install System Dependencies:**

**For Arch / Manjaro / CachyOS:**
```bash
sudo pacman -S cmake base-devel glfw-x11 glew glm pulseaudio
# Note: You can use `glfw-wayland` instead of `glfw-x11` if you are running exclusively on Wayland.
```

**For Ubuntu / Debian / Pop!_OS:**
```bash
sudo apt update
sudo apt install cmake build-essential libglfw3-dev libglew-dev libglm-dev libpulse-dev
```

**2. Build & Execute:**
```bash
# Clone the repository
mkdir -p build && cd build

# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build using all available CPU threads
make -j$(nproc)

# Run the simulation engine
./simlab_tui
```
</details>

<details>
<summary><b>🍏 macOS</b></summary>

**1. Install Dependencies (via [Homebrew](https://brew.sh/)):**
```bash
brew install cmake glfw glew glm pulseaudio pkg-config
```

**2. Configure PulseAudio Configuration (First time only):**
Since macOS uses CoreAudio natively, PulseAudio needs to be started as a daemon to capture audio for the FFT visualizer:
```bash
pulseaudio --start
```

**3. Build & Execute:**
```bash
mkdir -p build && cd build

# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build using all available CPU threads
make -j$(sysctl -n hw.ncpu)

# Run the simulation engine
./simlab_tui
```
</details>

---

## 🕹️ Controls & Navigation

*   **📁 File Explorer (Left Sidebar):** Click any directory/file to dynamically load its respective simulation engine into the main viewport.
*   **🎛️ Control Panel:** Expand the settings panel under the explorer to heavily tweak physics, speeds, colors, and specific simulation variables in real-time.
*   **🎥 Camera Orbit:** `Left-Click` & `Drag` to freely rotate the camera around 3D simulations.
*   **🔍 Camera Zoom:** `Scroll Wheel` to zoom in and out of the viewport scene.

---

## 👨‍💻 Developer & Maintainer

**Developed by:** [@swadhinbiswas](https://github.com/swadhinbiswas)  
**Email:** swadhinbiswas.cse@gmail.com  

If you found this **Open Source C++ Computer Graphics** project interesting, feel free to drop a ⭐ on the repository!

---
*Keywords for Search Optimization: Computer Graphics Project, C++ Simulation Engine, OpenGL Examples, ImGui Dashboard, Physics Engine C++, Generative Art C++, GLFW GLEW GLM Project, Math Visualizer, Open Source Graphics Simulation, 3D Rendering C++.*
