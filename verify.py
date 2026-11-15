"""
Verification and benchmarking script for Squeeze the Scene.

Usage:
    # Verify one image (uses practice_1 defaults):
    python verify.py

    # Verify one specific image:
    python verify.py practice/practice_1.npy squeeze/out.bin squeeze/rebuilt.npy

    # Benchmark all 3 practice images end-to-end:
    python verify.py --all
"""

import os
import sys
import time
import subprocess
import numpy as np

MAX_FILE_SIZE = 524_288   # 512 KiB
EXPECTED_SHAPE = (2048, 2048)
PYTHON = sys.executable


def verify_one(orig_path, bin_path, rebuilt_path, label=""):
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}=== Squeeze the Scene Verification ===")
    print(f"  Original : {orig_path}")
    print(f"  Binary   : {bin_path}")
    print(f"  Rebuilt  : {rebuilt_path}")
    print("-" * 54)

    # 1. File size
    if not os.path.exists(bin_path):
        print(f"  [FAIL] Binary not found: {bin_path}"); return False
    bin_size = os.path.getsize(bin_path)
    size_ok = bin_size <= MAX_FILE_SIZE
    pct = (bin_size / MAX_FILE_SIZE) * 100
    print(f"  [1] Size:   {bin_size:,} / {MAX_FILE_SIZE:,} bytes ({pct:.1f}%)  -> {'PASS' if size_ok else 'FAIL (Score = 0)'}")

    # 2-4. Rebuilt array checks
    if not os.path.exists(rebuilt_path):
        print(f"  [FAIL] Rebuilt not found: {rebuilt_path}"); return False
    rebuilt = np.load(rebuilt_path)
    shape_ok = rebuilt.shape == EXPECTED_SHAPE
    dtype_ok = rebuilt.dtype == np.float32
    finite_ok = bool(np.all(np.isfinite(rebuilt)))
    print(f"  [2] Shape:  {rebuilt.shape}  -> {'PASS' if shape_ok else 'FAIL'}")
    print(f"  [3] Dtype:  {rebuilt.dtype}  -> {'PASS' if dtype_ok else 'FAIL'}")
    print(f"  [4] Finite: (no NaN/Inf)  -> {'PASS' if finite_ok else 'FAIL'}")

    # 5-6. Quality checks (need original)
    if not os.path.exists(orig_path):
        print(f"  [WARN] Original not found, skipping quality checks."); return size_ok and shape_ok and dtype_ok and finite_ok

    orig = np.load(orig_path)
    orig_db = 20 * np.log10(np.maximum(orig, 1e-6))
    rebuilt_db = 20 * np.log10(np.maximum(rebuilt, 1e-6))
    mse_db = float(np.mean((orig_db - rebuilt_db) ** 2))
    db_range = float(orig_db.max() - orig_db.min())
    psnr = 10 * np.log10(db_range ** 2 / max(mse_db, 1e-12))
    print(f"  [5] Scene:  dB MSE={mse_db:.2f}  dB PSNR={psnr:.2f} dB")

    # Ship / target retention (top 0.1% brightest = ships)
    threshold = np.percentile(orig, 99.9)
    ship_mask = orig >= threshold
    n_ships = int(ship_mask.sum())
    orig_ships = orig[ship_mask]
    rebuilt_ships = rebuilt[ship_mask]
    rel_err = float(np.mean(np.abs(orig_ships - rebuilt_ships) / orig_ships))
    retained = float(np.mean(rebuilt_ships >= 0.5 * orig_ships) * 100)
    print(f"  [6] Ships:  {n_ships} pixels @p99.9  |  Err={rel_err*100:.1f}%  |  Retained≥50%amp: {retained:.1f}%")

    all_pass = size_ok and shape_ok and dtype_ok and finite_ok
    result = "ALL CONSTRAINTS PASSED" if all_pass else "CONSTRAINT FAILED"
    print(f"  {'=' * 50}")
    print(f"  RESULT: {result}")
    return all_pass


def benchmark_all():
    base = os.path.dirname(__file__)
    squeeze_dir = os.path.join(base, "squeeze")
    practice_dir = os.path.join(base, "practice")
    encode_script = os.path.join(squeeze_dir, "encode.py")
    decode_script = os.path.join(squeeze_dir, "decode.py")

    all_ok = True
    for i in [1, 2, 3]:
        orig = os.path.join(practice_dir, f"practice_{i}.npy")
        out_bin = os.path.join(squeeze_dir, f"_test_{i}.bin")
        out_npy = os.path.join(squeeze_dir, f"_test_{i}.npy")

        # Encode
        t0 = time.time()
        enc_result = subprocess.run([PYTHON, encode_script, orig, out_bin], capture_output=True, text=True)
        enc_time = time.time() - t0
        if enc_result.returncode != 0:
            print(f"[practice_{i}] ENCODE FAILED:\n{enc_result.stderr}")
            all_ok = False
            continue

        # Decode
        t1 = time.time()
        dec_result = subprocess.run([PYTHON, decode_script, out_bin, out_npy], capture_output=True, text=True)
        dec_time = time.time() - t1
        if dec_result.returncode != 0:
            print(f"[practice_{i}] DECODE FAILED:\n{dec_result.stderr}")
            all_ok = False
            continue

        print(f"\n  Timing: encode={enc_time:.2f}s  decode={dec_time:.2f}s  (limits: 60s each)")
        ok = verify_one(orig, out_bin, out_npy, label=f"practice_{i}")
        all_ok = all_ok and ok

        # Cleanup temp test files
        for f in [out_bin, out_npy]:
            if os.path.exists(f):
                os.remove(f)

    print(f"\n{'='*54}")
    if all_ok:
        print("ALL IMAGES PASSED — ready to submit.")
    else:
        print("ONE OR MORE IMAGES FAILED — fix before submitting.")


def main():
    if "--all" in sys.argv:
        benchmark_all()
    else:
        orig = sys.argv[1] if len(sys.argv) > 1 else os.path.join("practice", "practice_1.npy")
        bin_p = sys.argv[2] if len(sys.argv) > 2 else os.path.join("squeeze", "out.bin")
        rebu = sys.argv[3] if len(sys.argv) > 3 else os.path.join("squeeze", "rebuilt.npy")
        verify_one(orig, bin_p, rebu)


if __name__ == "__main__":
    main()
