"""Flask frontend for the C++/OpenMP image convolution engine."""
import base64, io, json, os, queue, shutil, subprocess, threading, uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "backend" / "bin" / ("convolve.exe" if os.name == "nt" else "convolve")
TMP = Path(__file__).resolve().parent / "tmp"
TMP.mkdir(exist_ok=True)
DOWNLOADS = TMP / "downloads"
DOWNLOADS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

FILTERS = ["gaussian", "blur", "sharpen", "edge", "sobel", "emboss"]
MAX_THREADS = 256
MAX_BATCH_IMAGES = 5
MAX_KERNEL_SIZE = 51


def engine_environment():
    env = os.environ.copy()
    dll_dirs = [BIN.parent]
    if os.name == "nt":
        configured = os.environ.get("MSYS2_UCRT64_BIN")
        dll_dirs.append(Path(configured) if configured else Path(r"C:\msys64\ucrt64\bin"))
    existing = [str(path) for path in dll_dirs if path.is_dir()]
    if existing:
        env["PATH"] = os.pathsep.join(existing + [env.get("PATH", "")])
    return env


def run_engine(args, on_trace=None):
    if not BIN.exists():
        raise RuntimeError("The C++ engine is not built. Open an MSYS2 UCRT64 terminal, go to backend/, and run: make")
    try:
        if on_trace is None:
            p = subprocess.run(
                [str(BIN)] + [str(a) for a in args],
                capture_output=True, text=True, timeout=1800, env=engine_environment(),
            )
            stdout, stderr = p.stdout, p.stderr
        else:
            p = subprocess.Popen(
                [str(BIN)] + [str(a) for a in args],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                bufsize=1, env=engine_environment(),
            )
            stderr_lines = []

            def read_stderr():
                for line in p.stderr:
                    line = line.rstrip()
                    if line.startswith("TRACE "):
                        try:
                            on_trace(json.loads(line[6:]))
                        except json.JSONDecodeError:
                            stderr_lines.append(line)
                    elif line:
                        stderr_lines.append(line)

            reader = threading.Thread(target=read_stderr, daemon=True)
            reader.start()
            try:
                p.wait(timeout=1800)
            except subprocess.TimeoutExpired as exc:
                p.kill()
                p.wait()
                reader.join()
                raise RuntimeError("The engine exceeded its 30-minute time limit. Try a smaller image or kernel.") from exc
            reader.join()
            stdout = p.stdout.read()
            stderr = "\n".join(stderr_lines)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("The engine exceeded its 30-minute time limit. Try a smaller image or kernel.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start the C++ engine: {exc}") from exc
    if p.returncode != 0:
        detail = stderr.strip() or stdout.strip()
        if os.name == "nt" and p.returncode >= 0xC0000000:
            status = p.returncode & 0xFFFFFFFF
            detail = (
                f"Windows could not load the OpenMP engine (status 0x{status:08X}). "
                "Install MSYS2 UCRT64 GCC or set MSYS2_UCRT64_BIN to its ucrt64\\bin folder."
            )
        raise RuntimeError(detail or f"The engine exited with code {p.returncode}.")
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("The engine returned an unexpected response. Rebuild it with the backend Makefile.") from exc


def to_data_url(path_or_img, max_side=1600):
    img = Image.open(path_or_img) if not isinstance(path_or_img, Image.Image) else path_or_img
    img = img.copy()
    img.thumbnail((max_side, max_side))  # preview only; full-res kept on disk
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_integer(value, label, minimum, maximum):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a whole number.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def validate_kernel_size(value):
    ksize = parse_integer(value, "Kernel size", 3, MAX_KERNEL_SIZE)
    if ksize % 2 == 0:
        raise ValueError("Kernel size must be odd (for example, 3, 5, or 9).")
    return ksize


def stream_event(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


@app.route("/")
def index():
    return render_template(
        "index.html", filters=FILTERS, cpus=os.cpu_count() or 4,
        engine_ready=BIN.exists(), engine_path=str(BIN),
    )


@app.route("/api/process", methods=["POST"])
def process():
    source = request.form.get("source", "upload")
    if source not in ("upload", "synthetic"):
        return jsonify(error="Choose an uploaded or generated image source."), 400
    files = [f for f in request.files.getlist("images") if f and f.filename] if source == "upload" else []
    if source == "upload" and not files:
        single_file = request.files.get("image")
        files = [single_file] if single_file and single_file.filename else []
    if source == "upload" and not files:
        return jsonify(error="Choose an image, or switch the source to generate a synthetic image."), 400
    if len(files) > MAX_BATCH_IMAGES:
        return jsonify(error=f"Choose up to {MAX_BATCH_IMAGES} images per batch."), 400

    flt = request.form.get("filter", "gaussian")
    mode = request.form.get("mode", "rows")
    if flt not in FILTERS:
        return jsonify(error="Choose one of the listed filters."), 400
    if mode not in ("rows", "tiles"):
        return jsonify(error="Choose row-wise or tile-wise parallelism."), 400
    try:
        ksize = validate_kernel_size(request.form.get("ksize", 5))
        threads = parse_integer(request.form.get("threads", os.cpu_count() or 4), "Threads", 1, MAX_THREADS)
        size = parse_integer(request.form.get("size", 3000), "Synthetic image side", 500, 8000) if source == "synthetic" else None
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    gray = request.form.get("gray") == "on"
    work = TMP / uuid.uuid4().hex
    job_token = uuid.uuid4().hex
    download_dir = DOWNLOADS / job_token
    download_dir.mkdir()
    work.mkdir()
    staged_files = []
    if source == "synthetic":
        staged_files.append((f"Synthetic {size} × {size}", None, True))
    else:
        try:
            for index, image_file in enumerate(files):
                filename = Path(image_file.filename).name
                upload_path = work / f"{index}.upload"
                image_file.save(upload_path)
                staged_files.append((filename, upload_path, False))
        except Exception as exc:
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(download_dir, ignore_errors=True)
            return jsonify(error=f"Could not stage uploaded images: {exc}"), 500

    def generate():
        results = []
        try:
            for index, (filename, upload_path, is_synthetic) in enumerate(staged_files):
                try:
                    yield stream_event("progress", dict(
                        index=index + 1, total=len(staged_files), filename=filename,
                        phase="prepare", detail=(
                            "Generating the synthetic RGB image in C++."
                            if is_synthetic else "Opening image and converting it for the C++ engine."
                        ),
                    ))
                    image_dir = work / str(index)
                    image_dir.mkdir()
                    out_p = image_dir / "out.ppm"
                    preview_path = image_dir / "input-preview.ppm"
                    if is_synthetic:
                        common = ["--generate", f"{size}x{size}", "--channels", 1 if gray else 3,
                                  "--filter", flt, "--ksize", ksize]
                        width = height = size
                    else:
                        with Image.open(upload_path) as source_image:
                            img = source_image.convert("L" if gray else "RGB")
                        in_p = image_dir / "in.ppm"
                        img.save(in_p, format="PPM")
                        common = ["--input", in_p, "--filter", flt, "--ksize", ksize]
                        width, height = img.width, img.height
                    yield stream_event("progress", dict(
                        index=index + 1, total=len(staged_files), filename=filename,
                        phase="sequential", detail="Running the single-threaded reference convolution.",
                    ))
                    seq_args = common + ["--mode", "seq"]
                    if is_synthetic:
                        seq_args += ["--preview-output", preview_path]
                    seq = run_engine(seq_args)
                    original_preview = to_data_url(preview_path if is_synthetic else img)
                    yield stream_event("progress", dict(
                        index=index + 1, total=len(staged_files), filename=filename,
                        phase="parallel", detail=f"Measuring {mode} performance, then mapping the actual {threads} OpenMP workers.",
                        width=width, height=height, mode=mode, threads=threads,
                        **({"preview": original_preview} if is_synthetic else {}),
                    ))
                    trace_events = queue.Queue()
                    trace_done = threading.Event()
                    trace_result = {}

                    def trace_parallel_run():
                        try:
                            trace_result["value"] = run_engine(
                                common + ["--mode", mode, "--threads", threads, "--output", out_p, "--verify", "--trace"],
                                on_trace=trace_events.put,
                            )
                        except Exception as exc:
                            trace_result["error"] = exc
                        finally:
                            trace_done.set()

                    trace_thread = threading.Thread(target=trace_parallel_run, daemon=True)
                    trace_thread.start()
                    while not trace_done.is_set() or not trace_events.empty():
                        try:
                            region = trace_events.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        yield stream_event("trace", dict(index=index + 1, filename=filename, **region))
                    trace_thread.join()
                    if "error" in trace_result:
                        raise trace_result["error"]
                    par = trace_result["value"]
                    speedup = seq["avg_ms"] / par["avg_ms"] if par["avg_ms"] else 0
                    download_path = download_dir / f"{index}.png"
                    yield stream_event("progress", dict(
                        index=index + 1, total=len(staged_files), filename=filename,
                        phase="preview", detail="Building the preview and full-resolution download.",
                    ))
                    with Image.open(out_p) as filtered:
                        filtered.save(download_path, format="PNG")
                    result = dict(
                        filename=filename,
                        original=original_preview, result=to_data_url(out_p),
                        download_url=f"/api/download/{job_token}/{index}",
                        width=width, height=height,
                        megapixels=round(width * height / 1e6, 2),
                        seq=seq, par=par, speedup=round(speedup, 2),
                        efficiency=round(speedup / threads * 100, 1),
                    )
                except Exception as exc:
                    result = dict(filename=filename, error=str(exc))
                results.append(result)
                yield stream_event("result", result)
            successful = sum("error" not in item for item in results)
            yield stream_event("done", dict(
                successful=successful, total=len(staged_files),
                error="No images could be processed. Review the errors below." if successful == 0 else None,
            ))
        except Exception as exc:
            yield stream_event("error", dict(message=f"Image processing failed: {exc}"))
        finally:
            shutil.rmtree(work, ignore_errors=True)
            if not any("error" not in item for item in results):
                shutil.rmtree(download_dir, ignore_errors=True)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/download/<token>/<int:image_index>")
def download_result(token, image_index):
    try:
        uuid.UUID(hex=token)
    except ValueError:
        return jsonify(error="This download link is invalid or expired."), 404
    path = DOWNLOADS / token / f"{image_index}.png"
    if not path.is_file():
        return jsonify(error="This result is no longer available. Process the image again."), 404
    return send_file(path, as_attachment=True, download_name=f"filtered-{image_index + 1}.png", mimetype="image/png")


@app.route("/api/benchmark", methods=["POST"])
def benchmark():
    is_synthetic = request.is_json
    d = request.get_json(silent=True) if is_synthetic else request.form
    if not isinstance(d, dict) and not is_synthetic:
        return jsonify(error="Benchmark settings were not received. Reload the page and try again."), 400
    if is_synthetic and not isinstance(d, dict):
        return jsonify(error="Benchmark settings were not received. Reload the page and try again."), 400

    image_file = None if is_synthetic else request.files.get("image")
    if not is_synthetic and (not image_file or not image_file.filename):
        return jsonify(error="Choose one uploaded image for the scalability test."), 400

    flt = d.get("filter", "gaussian")
    mode = d.get("mode", "rows")
    if flt not in FILTERS:
        return jsonify(error="Choose one of the listed filters."), 400
    if mode not in ("rows", "tiles"):
        return jsonify(error="Choose row-wise or tile-wise parallelism."), 400
    try:
        ksize = validate_kernel_size(d.get("ksize", 9))
        size = parse_integer(d.get("size", 3000), "Synthetic image side", 500, 8000) if is_synthetic else None
        raw_threads = d.get("threads", [1, 2, 4, 8] if is_synthetic else "1,2,4,8")
        if not is_synthetic and isinstance(raw_threads, str):
            raw_threads = [value.strip() for value in raw_threads.split(",") if value.strip()]
        if not isinstance(raw_threads, list) or not raw_threads or len(raw_threads) > 16:
            raise ValueError("Enter between 1 and 16 thread counts.")
        threads = list(dict.fromkeys(parse_integer(t, "Thread count", 1, MAX_THREADS) for t in raw_threads))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    preview_dir = TMP / f"benchmark-{uuid.uuid4().hex}"
    preview_dir.mkdir()
    try:
        if is_synthetic:
            common = ["--generate", f"{size}x{size}", "--channels", 1 if d.get("gray") else 3,
                      "--filter", flt, "--ksize", ksize]
            width = height = size
            preview_path = preview_dir / "input-preview.ppm"
            sequential_args = common + ["--mode", "seq", "--preview-output", preview_path]
            original_preview_source = preview_path
            source_name = f"Synthetic {size} × {size}"
        else:
            gray = d.get("gray") == "on"
            with Image.open(image_file.stream) as source_image:
                image = source_image.convert("L" if gray else "RGB")
            width, height = image.size
            input_path = preview_dir / "input.ppm"
            image.save(input_path, format="PPM")
            common = ["--input", input_path, "--filter", flt, "--ksize", ksize]
            sequential_args = common + ["--mode", "seq"]
            original_preview_source = image
            source_name = Path(image_file.filename).name

        seq = run_engine(sequential_args)
        rows = []
        output_path = preview_dir / "filtered.ppm"
        for t in threads:
            r = run_engine(common + [
                "--mode", mode, "--threads", t,
                "--output", output_path, "--verify",
            ])
            sp = seq["avg_ms"] / r["avg_ms"] if r["avg_ms"] else 0
            rows.append(dict(
                threads=t, time_ms=r["avg_ms"], speedup=round(sp, 3),
                efficiency=round(sp / t * 100, 1), verified=r["verified"],
            ))
        return jsonify(
            seq_ms=seq["avg_ms"], rows=rows, size=size,
            width=width, height=height, source=source_name,
            original_preview=to_data_url(original_preview_source, max_side=600),
            filtered_preview=to_data_url(output_path, max_side=600),
            filtered_threads=threads[-1],
        )
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        shutil.rmtree(preview_dir, ignore_errors=True)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="Upload is larger than the 200 MB request limit. Try fewer or smaller images."), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
