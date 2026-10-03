#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct Image {
    int width = 0, height = 0, channels = 0;
    std::vector<uint8_t> data;
    Image() = default;
    Image(int w, int h, int c) : width(w), height(h), channels(c), data((size_t)w * h * c) {}
};

struct Kernel {
    std::string name;
    int size = 3;
    std::vector<float> a;   // main kernel (size*size)
    std::vector<float> b;   // optional 2nd kernel (Sobel Gy)
    float bias = 0.f;
    bool hasSecond() const { return !b.empty(); }
};

enum class Mode { Sequential, Rows, Tiles };

bool readPNM(const std::string& path, Image& img);
bool writePNM(const std::string& path, const Image& img);
Image generateSynthetic(int w, int h, int c);
bool makeKernel(const std::string& name, int ksize, Kernel& k);
using RegionTrace = std::function<void(int, int, int, int, int, bool)>;
void convolve(const Image& in, Image& out, const Kernel& k, Mode mode, int tile, const RegionTrace& trace = {});
