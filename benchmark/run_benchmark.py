"""Benchmark for the PDC report: sequential vs OpenMP (rows & tiles) across thread counts.
Produces results/benchmark.csv + time / speedup / efficiency PNG graphs.

Usage:  python run_benchmark.py --size 6000 --filter gaussian --ksize 9 --repeat 3
"""
import argparse, csv, json, os, subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "backend" / "bin" / ("convolve.exe" if os.name == "nt" else "convolve")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def run(args):
    if not BIN.exists():
        raise SystemExit(f"Engine not built: {BIN}. Build it from an MSYS2 UCRT64 terminal in backend/.")
    env = os.environ.copy()
    dll_dirs = [BIN.parent]
    if os.name == "nt":
        configured = os.environ.get("MSYS2_UCRT64_BIN")
        dll_dirs.append(Path(configured) if configured else Path(r"C:\msys64\ucrt64\bin"))
    existing = [str(path) for path in dll_dirs if path.is_dir()]
    if existing:
        env["PATH"] = os.pathsep.join(existing + [env.get("PATH", "")])
    try:
        p = subprocess.run([str(BIN)] + [str(a) for a in args], capture_output=True, text=True, env=env)
    except OSError as exc:
        raise SystemExit(f"Could not start the engine: {exc}") from exc
    if p.returncode:
        detail = p.stderr.strip() or p.stdout.strip()
        if os.name == "nt" and p.returncode >= 0xC0000000:
            detail = (
                f"Windows could not load the OpenMP engine (status 0x{p.returncode & 0xFFFFFFFF:08X}). "
                "Install MSYS2 UCRT64 GCC or set MSYS2_UCRT64_BIN to its ucrt64\\bin folder."
            )
        raise SystemExit(detail or f"Engine exited with code {p.returncode}.")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=6000, help="image is size x size, RGB")
    ap.add_argument("--filter", default="gaussian")
    ap.add_argument("--ksize", type=int, default=9)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", default="", help="comma list, default: 1,2,4,... up to CPU count")
    a = ap.parse_args()

    cpus = os.cpu_count() or 4
    if a.threads:
        threads = [int(t) for t in a.threads.split(",")]
    else:
        threads, t = [], 1
        while t < cpus:
            threads.append(t); t *= 2
        threads.append(cpus)
    common = ["--generate", f"{a.size}x{a.size}", "--filter", a.filter, "--ksize", a.ksize, "--repeat", a.repeat]

    print(f"Image {a.size}x{a.size} ({a.size*a.size/1e6:.1f} MP), filter={a.filter}, k={a.ksize}, threads={threads}")
    seq = run(common + ["--mode", "seq"])
    t_seq = seq["best_ms"]
    print(f"sequential: {t_seq:.1f} ms")

    rows = [dict(mode="seq", threads=1, time_ms=t_seq, speedup=1.0, efficiency=100.0)]
    for mode in ("rows", "tiles"):
        for t in threads:
            r = run(common + ["--mode", mode, "--threads", t, "--verify"])
            sp = t_seq / r["best_ms"] if r["best_ms"] else 0
            rows.append(dict(mode=mode, threads=t, time_ms=r["best_ms"], speedup=round(sp, 3), efficiency=round(sp / t * 100, 1)))
            print(f"{mode:5s} threads={t:3d}  {r['best_ms']:9.1f} ms  speedup={sp:5.2f}  eff={sp/t*100:5.1f}%  verified={r['verified']}")

    with open(OUT / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    def plot(key, ylabel, fname, ideal=False):
        plt.figure(figsize=(6, 4))
        for mode in ("rows", "tiles"):
            pts = [r for r in rows if r["mode"] == mode]
            plt.plot([r["threads"] for r in pts], [r[key] for r in pts], marker="o", label=f"OpenMP {mode}")
        if ideal:
            plt.plot(threads, threads, "--", color="gray", label="Ideal")
        if key == "time_ms":
            plt.axhline(t_seq, color="red", ls=":", label="Sequential")
        plt.xlabel("Threads"); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.legend()
        plt.title(f"{ylabel} - {a.size}x{a.size}, {a.filter} k={a.ksize}")
        plt.tight_layout(); plt.savefig(OUT / fname, dpi=150); plt.close()

    plot("time_ms", "Execution time (ms)", "time.png")
    plot("speedup", "Speedup", "speedup.png", ideal=True)
    plot("efficiency", "Efficiency (%)", "efficiency.png")
    print(f"Saved CSV + graphs in {OUT}")


if __name__ == "__main__":
    main()
