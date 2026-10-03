// Parallel Image Convolution (OpenMP) - CLI engine.
// Prints ONE JSON line on stdout (consumed by the Flask frontend & benchmark script).
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "convolution.hpp"

static void usage() {
    fprintf(stderr,
        "Usage: convolve [--input img.ppm | --generate WxH] [--output out.ppm] [--preview-output preview.ppm]\n"
        "                [--filter blur|gaussian|sharpen|edge|sobel|emboss] [--ksize N]\n"
        "                [--mode seq|rows|tiles] [--threads N] [--tile N]\n"
                "                [--repeat N] [--channels 1|3] [--verify] [--trace]\n");
}

static bool writeInputPreview(const std::string& path, const Image& input) {
    constexpr int maxSide = 600;
    const int stride = std::max(1, (std::max(input.width, input.height) + maxSide - 1) / maxSide);
    const int width = (input.width + stride - 1) / stride;
    const int height = (input.height + stride - 1) / stride;
    Image preview(width, height, input.channels);
    for (int y = 0; y < height; ++y) {
        const int sourceY = std::min(input.height - 1, y * stride);
        for (int x = 0; x < width; ++x) {
            const int sourceX = std::min(input.width - 1, x * stride);
            for (int channel = 0; channel < input.channels; ++channel) {
                preview.data[((size_t)y * width + x) * input.channels + channel] =
                    input.data[((size_t)sourceY * input.width + sourceX) * input.channels + channel];
            }
        }
    }
    return writePNM(path, preview);
}

int main(int argc, char** argv) {
    std::string input, output, previewOutput, filter = "gaussian", modeStr = "rows";
    int ksize = 5, threads = omp_get_max_threads(), repeat = 1, tile = 64, gw = 0, gh = 0, gc = 3;
    bool verify = false, trace = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(2); }
            return argv[++i];
        };
        if (a == "--input") input = next();
        else if (a == "--output") output = next();
        else if (a == "--preview-output") previewOutput = next();
        else if (a == "--filter") filter = next();
        else if (a == "--ksize") ksize = atoi(next().c_str());
        else if (a == "--mode") modeStr = next();
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--tile") tile = atoi(next().c_str());
        else if (a == "--repeat") repeat = atoi(next().c_str());
        else if (a == "--channels") gc = atoi(next().c_str());
        else if (a == "--generate") { if (sscanf(next().c_str(), "%dx%d", &gw, &gh) != 2) { usage(); return 2; } }
        else if (a == "--verify") verify = true;
        else if (a == "--trace") trace = true;
        else { usage(); return 2; }
    }

    Mode mode;
    if (modeStr == "seq") mode = Mode::Sequential;
    else if (modeStr == "rows") mode = Mode::Rows;
    else if (modeStr == "tiles") mode = Mode::Tiles;
    else { fprintf(stderr, "bad --mode\n"); return 2; }
    if (threads < 1) threads = 1;
    if (tile < 8) tile = 8;
    if (repeat < 1) repeat = 1;

    Image in;
    if (!input.empty()) {
        if (!readPNM(input, in)) { fprintf(stderr, "cannot read PPM/PGM: %s\n", input.c_str()); return 1; }
    } else if (gw > 0 && gh > 0) {
        in = generateSynthetic(gw, gh, gc == 1 ? 1 : 3);
    } else { usage(); return 2; }

    if (!previewOutput.empty() && !writeInputPreview(previewOutput, in)) {
        fprintf(stderr, "cannot write input preview\n");
        return 1;
    }

    Kernel k;
    if (!makeKernel(filter, ksize, k)) { fprintf(stderr, "unknown filter\n"); return 2; }

    omp_set_num_threads(mode == Mode::Sequential ? 1 : threads);
    Image out(in.width, in.height, in.channels);

    double total = 0, best = 1e30;
    for (int r = 0; r < repeat; ++r) {
        double t0 = omp_get_wtime();
        convolve(in, out, k, mode, tile);
        double t = omp_get_wtime() - t0;
        total += t; if (t < best) best = t;
    }

    if (trace) {
        RegionTrace report = [](int worker, int x, int y, int width, int height, bool active) {
            #pragma omp critical(trace_output)
            {
                fprintf(stderr, "TRACE {\"thread\":%d,\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d,\"active\":%s}\n",
                        worker, x, y, width, height, active ? "true" : "false");
                fflush(stderr);
            }
        };
        convolve(in, out, k, mode, tile, report);
    }

    int maxDiff = -1;
    if (verify) {
        Image ref(in.width, in.height, in.channels);
        convolve(in, ref, k, Mode::Sequential, tile);
        maxDiff = 0;
        for (size_t i = 0; i < ref.data.size(); ++i) {
            int d = abs((int)ref.data[i] - (int)out.data[i]);
            if (d > maxDiff) maxDiff = d;
        }
    }

    if (!output.empty() && !writePNM(output, out)) { fprintf(stderr, "cannot write output\n"); return 1; }

    printf("{\"width\":%d,\"height\":%d,\"channels\":%d,\"filter\":\"%s\",\"ksize\":%d,"
           "\"mode\":\"%s\",\"threads\":%d,\"repeat\":%d,\"avg_ms\":%.3f,\"best_ms\":%.3f,"
           "\"verified\":%s,\"max_diff\":%d}\n",
           in.width, in.height, in.channels, filter.c_str(), k.size, modeStr.c_str(),
           mode == Mode::Sequential ? 1 : threads, repeat, total / repeat * 1000.0, best * 1000.0,
           verify ? (maxDiff == 0 ? "true" : "false") : "null", maxDiff);
    return 0;
}
