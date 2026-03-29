#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>

// PulseAudio simple API
#include <pulse/simple.h>
#include <pulse/error.h>

namespace audio {

// Simple radix-2 FFT (in-place, Cooley-Tukey)
static void fft(std::vector<float>& real, std::vector<float>& imag) {
    int n = (int)real.size();
    if (n <= 1) return;

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // FFT butterfly
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.f * 3.14159265f / len;
        float wR = std::cos(ang), wI = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float curR = 1.f, curI = 0.f;
            for (int j = 0; j < len / 2; j++) {
                float uR = real[i+j],           uI = imag[i+j];
                float vR = real[i+j+len/2]*curR - imag[i+j+len/2]*curI;
                float vI = real[i+j+len/2]*curI + imag[i+j+len/2]*curR;
                real[i+j]         = uR + vR;
                imag[i+j]         = uI + vI;
                real[i+j+len/2]   = uR - vR;
                imag[i+j+len/2]   = uI - vI;
                float tmpR = curR * wR - curI * wI;
                curI = curR * wI + curI * wR;
                curR = tmpR;
            }
        }
    }
}

struct AudioSim {
    // FFT size (must be power of 2)
    static constexpr int FFT_SIZE = 2048;
    static constexpr int SAMPLE_RATE = 44100;
    static constexpr int HALF_FFT = FFT_SIZE / 2;

    // Audio buffers (thread-safe)
    std::mutex mtx;
    std::vector<float> waveform;      // raw waveform [FFT_SIZE]
    std::vector<float> spectrum;      // magnitude spectrum [HALF_FFT]
    std::vector<float> smoothSpec;    // smoothed spectrum for display

    // Spectrogram (scrolling)
    std::vector<std::vector<float>> spectrogram; // [HIST_LINES][HALF_FFT]
    int spectrogramLines = 80;

    // Audio thread
    std::thread captureThread;
    std::atomic<bool> running{false};
    std::atomic<bool> connected{false};
    std::atomic<float> volume{0.f};
    std::atomic<float> peakFreq{0.f};

    // Display settings
    int  vizMode    = 0; // 0=waveform+bars, 1=circular, 2=spectrogram
    float sensitivity = 1.5f;
    float smoothing   = 0.85f;
    float barWidth    = 1.f;
    int   numBars     = 64;
    float waveScale   = 1.0f;
    float time        = 0.f;

    // Circular viz
    float circleRadius = 0.3f;
    float circleScale  = 0.4f;

    // Color
    int colorMode = 0; // 0=green, 1=spectrum rainbow, 2=cyan-pulse
    float colorPulse = 0.f;

    void init() {
        waveform.assign(FFT_SIZE, 0.f);
        spectrum.assign(HALF_FFT, 0.f);
        smoothSpec.assign(HALF_FFT, 0.f);
        spectrogram.assign(spectrogramLines, std::vector<float>(HALF_FFT, 0.f));
        time = 0.f;

        if (!running) {
            running = true;
            captureThread = std::thread(&AudioSim::captureLoop, this);
        }
    }

    void shutdown() {
        running = false;
        if (captureThread.joinable())
            captureThread.join();
    }

    ~AudioSim() { shutdown(); }

    void captureLoop() {
        // PulseAudio setup
        pa_sample_spec ss;
        ss.format = PA_SAMPLE_FLOAT32LE;
        ss.channels = 1;
        ss.rate = SAMPLE_RATE;

        int error;
        pa_simple* s = pa_simple_new(
            nullptr,           // default server
            "SimLabTUI",       // app name
            PA_STREAM_RECORD,  // direction
            nullptr,           // default device
            "Audio Visualizer",// stream description
            &ss,               // sample format
            nullptr,           // default channel map
            nullptr,           // default buffering
            &error
        );

        if (!s) {
            connected = false;
            return;
        }
        connected = true;

        std::vector<float> buf(FFT_SIZE);

        while (running) {
            // Read audio samples
            if (pa_simple_read(s, buf.data(), buf.size() * sizeof(float), &error) < 0) {
                break;
            }

            // Apply Hann window
            std::vector<float> windowed(FFT_SIZE);
            for (int i = 0; i < FFT_SIZE; i++) {
                float w = 0.5f * (1.f - std::cos(2.f * 3.14159265f * i / (FFT_SIZE - 1)));
                windowed[i] = buf[i] * w;
            }

            // Compute FFT
            std::vector<float> real = windowed;
            std::vector<float> imag(FFT_SIZE, 0.f);
            fft(real, imag);

            // Compute magnitude spectrum
            std::vector<float> mag(HALF_FFT);
            float maxMag = 0.f;
            float vol = 0.f;
            for (int i = 0; i < HALF_FFT; i++) {
                mag[i] = std::sqrt(real[i]*real[i] + imag[i]*imag[i]) / FFT_SIZE;
                maxMag = std::max(maxMag, mag[i]);
            }
            for (int i = 0; i < FFT_SIZE; i++)
                vol += buf[i] * buf[i];
            vol = std::sqrt(vol / FFT_SIZE);

            // Find peak frequency
            int peakBin = 0;
            for (int i = 1; i < HALF_FFT; i++)
                if (mag[i] > mag[peakBin]) peakBin = i;
            float pf = (float)peakBin * SAMPLE_RATE / FFT_SIZE;

            // Update shared state
            {
                std::lock_guard<std::mutex> lock(mtx);
                waveform = buf;
                spectrum = mag;
            }
            volume = vol;
            peakFreq = pf;
        }

        pa_simple_free(s);
        connected = false;
    }

    void update(float dt) {
        time += dt;

        // Smooth the spectrum for display
        std::lock_guard<std::mutex> lock(mtx);
        for (int i = 0; i < HALF_FFT; i++) {
            float target = spectrum[i] * sensitivity;
            smoothSpec[i] = smoothSpec[i] * smoothing + target * (1.f - smoothing);
        }

        // Update spectrogram (shift and add new line)
        spectrogram.erase(spectrogram.begin());
        spectrogram.push_back(smoothSpec);

        // Color pulse based on volume
        colorPulse = colorPulse * 0.9f + volume.load() * 0.1f * sensitivity;
    }

    void getBarColor(int barIdx, int totalBars, float value,
                     float& r, float& g, float& b) const {
        float t = (float)barIdx / totalBars;
        float v = std::clamp(value * 3.f, 0.f, 1.f);

        switch (colorMode) {
            case 0: // Terminal green
                r = v * 0.1f;
                g = 0.2f + v * 0.8f;
                b = v * 0.15f;
                break;
            case 1: { // Rainbow spectrum
                float h = t * 6.f;
                int hi = (int)h % 6;
                float f = h - (int)h;
                switch (hi) {
                    case 0: r=1; g=f; b=0; break;
                    case 1: r=1-f; g=1; b=0; break;
                    case 2: r=0; g=1; b=f; break;
                    case 3: r=0; g=1-f; b=1; break;
                    case 4: r=f; g=0; b=1; break;
                    case 5: r=1; g=0; b=1-f; break;
                    default: r=g=b=1; break;
                }
                r *= (0.3f + v * 0.7f);
                g *= (0.3f + v * 0.7f);
                b *= (0.3f + v * 0.7f);
                break;
            }
            case 2: // Cyan pulse
                r = v * 0.1f + colorPulse * 0.3f;
                g = v * 0.6f + colorPulse * 0.2f;
                b = 0.3f + v * 0.7f;
                break;
        }
    }

    bool isConnected() const { return connected; }
    float getVolume() const { return volume; }
    float getPeakFreq() const { return peakFreq; }
};

} // namespace audio

// @swadhinbiswas - swadhinbiswas.cse@gmail.com
