# Parallel Image Convolution / Filtering

A Parallel and Distributed Computing course project that applies image filters using C++17 and OpenMP on a single multi-core node. The Flask UI accepts real images, runs sequential and parallel versions, checks correctness, and reports performance. A separate benchmark measures how performance changes with image size and thread count.

## How the project works

1. The browser uploads up to five images. Flask/Pillow reads common image formats and converts each image to the PPM format used by the C++ engine. Grayscale is optional; otherwise pixels are processed as RGB.
2. The C++ engine applies a convolution kernel. For each output pixel it multiplies the surrounding `k x k` pixels by kernel weights and sums them. Image boundaries use clamp-to-edge padding.
3. The engine first measures the sequential implementation, then the selected OpenMP strategy. In row mode, OpenMP statically schedules rows. In tile mode, it dynamically schedules 64 x 64 image regions.
4. The engine compares parallel output with sequential output byte by byte. Flask returns the previews, timings, speedup, efficiency, and a full-resolution PNG download.

Each output pixel reads only the unchanged input image, so different output pixels can be computed independently. The approximate work is `O(width x height x channels x kernel_size^2)`.

## Build on Windows with MSYS2 UCRT64

Use an **MSYS2 UCRT64** terminal for compiling, not the old MinGW compiler. Install GCC, make, and the OpenMP runtime if needed:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-libgomp
```

Build the executable from the repository's `backend/` directory:

```bash
cd /c/path/to/pdc-convolution/backend
make CXX=g++ CXXFLAGS="-O2 -std=c++17 -fopenmp -Wall -Wextra"
```

The expected output is `backend/bin/convolve.exe`.

## Start the web UI on Windows

In PowerShell, from the repository directory:

```powershell
cd frontend
python -m pip install -r requirements.txt
$env:MSYS2_UCRT64_BIN = "C:\msys64\ucrt64\bin"
python app.py
```

Open <http://127.0.0.1:5000>. The app also checks the default MSYS2 path automatically. If MSYS2 is installed elsewhere, set `MSYS2_UCRT64_BIN` to that installation's `ucrt64\bin` directory before starting Flask. This ensures Windows loads the UCRT64 `libgomp-1.dll` used by the compiled engine instead of a conflicting runtime earlier on `PATH`.

In the UI, choose one or more images, filter, kernel size, row/tile strategy, and thread count, then select **Run image filter**. Each image displays its original and filtered preview, sequential and parallel times, speedup, efficiency, correctness result, and full-size PNG download. Open the filter and run help sections in the UI for control descriptions.

## Run the benchmark

In PowerShell, install benchmark dependencies and run the script:

```powershell
cd benchmark
python -m pip install -r requirements.txt
python run_benchmark.py --size 3000 --filter gaussian --ksize 9 --repeat 3 --threads 1,2,4,8
```

The script writes `benchmark/results/benchmark.csv` and execution-time, speedup, and efficiency charts. The web UI also displays a downscaled preview generated from the same deterministic C++ synthetic input used by the benchmark; the computation still uses the full-size image, not uploaded files. Use a smaller image/kernel for a quick smoke run; larger sizes and kernels take substantially longer. Close other heavy applications for more stable timing.

Speedup is `sequential_time / parallel_time`; values above `1` mean the parallel run was faster. Efficiency is `speedup / thread_count x 100%`. Efficiency can fall as threads increase because of memory bandwidth, scheduling overhead, and the serial portion of execution. Timings measure convolution only, not file conversion, image generation, or correctness verification.

## CLI examples

Run these from `backend/` in the MSYS2 UCRT64 terminal:

```bash
./bin/convolve.exe --generate 6000x6000 --filter gaussian --ksize 9 --mode seq
./bin/convolve.exe --generate 6000x6000 --filter gaussian --ksize 9 --mode rows --threads 8 --verify
./bin/convolve.exe --generate 6000x6000 --filter sobel --mode tiles --threads 8 --tile 64 --verify
./bin/convolve.exe --input input.ppm --output output.ppm --filter sharpen --mode rows --threads 8
```

Available filters: `blur`, `gaussian`, `sharpen`, `edge`, `sobel`, `emboss`. Available execution modes: `seq`, `rows`, `tiles`. The C++ command-line engine reads/writes PPM or PGM; the web UI handles common browser image formats through Pillow.

## Troubleshooting

- **Engine is not built:** build it from `backend/` in the MSYS2 UCRT64 terminal. Confirm `backend/bin/convolve.exe` exists.
- **Windows cannot load the OpenMP engine / status `0xC000007B`:** set `MSYS2_UCRT64_BIN` to the matching MSYS2 UCRT64 `bin` directory before starting Flask or the benchmark. Confirm `libgomp-1.dll` exists there and rebuild with UCRT64 GCC, not another MinGW distribution.
- **Unsupported or unreadable image:** use a valid PNG, JPEG, BMP, or TIFF file. Uploads are limited to 200 MB total and five images per batch.
- **Slow request:** reduce image dimensions or kernel size, or use fewer benchmark thread counts. The UI request has a 30-minute engine limit.
- **Low speedup:** small images may not provide enough work to offset thread scheduling; try larger images and compare several thread counts.