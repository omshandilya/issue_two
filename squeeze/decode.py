"""
Squeeze the Scene — Decoder
============================
Reconstructs a 2048×2048 float32 SAR image from the compressed binary
produced by encode.py.

This file must work with ONLY out.bin present — no original image, no network,
no other files.

Reconstruction steps:
  1. Parse header to recover image dimensions, dB range, and blob sizes.
  2. Decompress and dequantise the 8-bit downsampled background grid.
  3. Upsample the background 4× with nearest-neighbour (fast, keeps sharp edges).
  4. Convert background from dB back to linear amplitude.
  5. Decompress target records (row, col, exact float32 amplitude).
  6. Overwrite background pixels at target locations with exact amplitudes.
  7. Save as a float32 .npy array.
"""

import struct
import sys
import zlib

import numpy as np

MAGIC = b"SQZ1"


def decode(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        raw = f.read()

    # ── parse header ──────────────────────────────────────────────────────────
    # Header layout: magic(4) H(4) W(4) BG_FACTOR(4) db_min(8) db_max(8) bg_len(4) tgt_len(4) = 40 bytes
    hdr_size = 40
    magic, H, W, bg_factor, db_min, db_max, bg_len, tgt_len = struct.unpack_from(
        "<4sIIIddII", raw, 0
    )

    if magic != MAGIC:
        raise ValueError(f"Bad magic bytes: {magic!r}")

    bg_blob = raw[hdr_size : hdr_size + bg_len]
    tgt_blob = raw[hdr_size + bg_len : hdr_size + bg_len + tgt_len]

    # ── Layer 1: reconstruct background ───────────────────────────────────────
    h, w = H // bg_factor, W // bg_factor
    db_range = db_max - db_min if db_max > db_min else 1.0

    bg_raw = zlib.decompress(bg_blob)
    db_ds_q = np.frombuffer(bg_raw, dtype=np.uint8).reshape(h, w)

    # Dequantise: 8-bit → dB
    db_ds = db_ds_q.astype(np.float32) / 255.0 * db_range + db_min

    # Upsample 4× with nearest-neighbour
    db_up = np.repeat(np.repeat(db_ds, bg_factor, axis=0), bg_factor, axis=1)

    # Convert dB → linear amplitude
    rebuilt = np.power(10.0, db_up / 20.0).astype(np.float32)

    # ── Layer 2: restore bright targets ───────────────────────────────────────
    tgt_raw = zlib.decompress(tgt_blob)
    if len(tgt_raw) > 0:
        dtype = np.dtype([("r", "<u2"), ("c", "<u2"), ("v", "<f4")])
        tgt_data = np.frombuffer(tgt_raw, dtype=dtype)
        # Overwrite target pixels with exact recorded amplitudes
        rebuilt[tgt_data["r"], tgt_data["c"]] = tgt_data["v"]

    # ── sanity clamp (no NaN, no Inf, all finite) ─────────────────────────────
    rebuilt = np.where(np.isfinite(rebuilt), rebuilt, 0.0).astype(np.float32)

    np.save(output_path, rebuilt)


if __name__ == "__main__":
    decode(sys.argv[1], sys.argv[2])
