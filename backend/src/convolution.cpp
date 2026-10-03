#include "convolution.hpp"
#include <omp.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

// ---------------------------------------------------------------- PNM I/O
static void skipWs(std::istream& f) {
    while (true) {
        int c = f.peek();
        if (c == EOF) break;
        if (c == '#') { std::string line; std::getline(f, line); }
        else if (std::isspace(c)) f.get();
        else break;
    }
}

bool readPNM(const std::string& path, Image& img) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic; f >> magic;
    if (magic != "P5" && magic != "P6") return false;
    int c = (magic == "P6") ? 3 : 1, w = 0, h = 0, maxv = 0;
    skipWs(f); f >> w; skipWs(f); f >> h; skipWs(f); f >> maxv; f.get();
    if (!f || w <= 0 || h <= 0 || maxv != 255) return false;
    img = Image(w, h, c);
    f.read(reinterpret_cast<char*>(img.data.data()), (std::streamsize)img.data.size());
    return (bool)f;
}

bool writePNM(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << (img.channels == 3 ? "P6" : "P5") << "\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), (std::streamsize)img.data.size());
    return (bool)f;
}

// Deterministic synthetic image (gradients + rings + checker) so we can benchmark
// huge sizes without shipping huge files.
Image generateSynthetic(int w, int h, int c) {
    Image img(w, h, c);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch) {
                double dx = x - w / 2.0, dy = y - h / 2.0;
                double ring = 0.5 + 0.5 * std::sin(std::sqrt(dx * dx + dy * dy) * 0.05 + ch);
                double grad = (double)(x + y * (ch + 1)) / (w + h * c);
                int chk = (((x >> 5) + (y >> 5)) & 1) ? 40 : 0;
                int v = (int)(120 * ring + 95 * grad) + chk;
                img.data[((size_t)y * w + x) * c + ch] = (uint8_t)std::min(255, std::max(0, v));
            }
    return img;
}

// ---------------------------------------------------------------- kernels
bool makeKernel(const std::string& name, int ksize, Kernel& k) {
    k = Kernel(); k.name = name;
    if (ksize % 2 == 0) ksize++;
    if (ksize < 3) ksize = 3;
    if (name == "blur") {
        k.size = ksize; k.a.assign((size_t)ksize * ksize, 1.f / (ksize * ksize));
    } else if (name == "gaussian") {
        k.size = ksize; k.a.resize((size_t)ksize * ksize);
        int r = ksize / 2; double sigma = std::max(0.8, ksize / 6.0), sum = 0;
        for (int y = -r; y <= r; ++y) for (int x = -r; x <= r; ++x) {
            double v = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            k.a[(y + r) * ksize + (x + r)] = (float)v; sum += v;
        }
        for (auto& v : k.a) v = (float)(v / sum);
    } else if (name == "sharpen") {
        k.size = 3; k.a = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    } else if (name == "edge") {          // Laplacian
        k.size = 3; k.a = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    } else if (name == "sobel") {         // gradient magnitude sqrt(Gx^2+Gy^2)
        k.size = 3;
        k.a = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        k.b = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    } else if (name == "emboss") {
        k.size = 3; k.a = {-2, -1, 0, -1, 1, 1, 0, 1, 2}; k.bias = 128.f;
    } else return false;
    return true;
}

// ---------------------------------------------------------------- core
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static inline void computePixel(const Image& in, Image& out, const Kernel& k, int x, int y) {
    const int W = in.width, H = in.height, C = in.channels, r = k.size / 2, ks = k.size;
    const bool two = k.hasSecond();
    for (int c = 0; c < C; ++c) {
        float sa = 0.f, sb = 0.f;
        for (int ky = -r; ky <= r; ++ky) {
            const int yy = clampi(y + ky, 0, H - 1);
            for (int kx = -r; kx <= r; ++kx) {
                const int xx = clampi(x + kx, 0, W - 1);
                const float p = in.data[((size_t)yy * W + xx) * C + c];
                const int ki = (ky + r) * ks + (kx + r);
                sa += p * k.a[ki];
                if (two) sb += p * k.b[ki];
            }
        }
        float v = two ? std::sqrt(sa * sa + sb * sb) : sa + k.bias;
        out.data[((size_t)y * W + x) * C + c] = (uint8_t)clampi((int)(v + 0.5f), 0, 255);
    }
}

void convolve(const Image& in, Image& out, const Kernel& k, Mode mode, int tile, const RegionTrace& trace) {
    const int W = in.width, H = in.height;
    if (out.data.size() != in.data.size()) out = Image(W, H, in.channels);

    if (mode == Mode::Sequential) {
        for (int y = 0; y < H; ++y) {
            if (trace) trace(0, 0, y, W, 1, true);
            for (int x = 0; x < W; ++x) computePixel(in, out, k, x, y);
            if (trace) trace(0, 0, y, W, 1, false);
        }

    } else if (mode == Mode::Rows) {
        // Strategy 1: each thread gets a block of rows
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const int worker = omp_get_thread_num();
            if (trace) trace(worker, 0, y, W, 1, true);
            for (int x = 0; x < W; ++x) computePixel(in, out, k, x, y);
            if (trace) trace(worker, 0, y, W, 1, false);
        }

    } else {
        // Strategy 2: 2D tiles, dynamic scheduling (better cache locality / load balance)
        const int tx = (W + tile - 1) / tile, ty = (H + tile - 1) / tile;
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int j = 0; j < ty; ++j)
            for (int i = 0; i < tx; ++i) {
                const int y0 = j * tile, y1 = std::min(H, y0 + tile);
                const int x0 = i * tile, x1 = std::min(W, x0 + tile);
                const int worker = omp_get_thread_num();
                if (trace) trace(worker, x0, y0, x1 - x0, y1 - y0, true);
                for (int y = y0; y < y1; ++y)
                    for (int x = x0; x < x1; ++x) computePixel(in, out, k, x, y);
                if (trace) trace(worker, x0, y0, x1 - x0, y1 - y0, false);
            }
    }
}
