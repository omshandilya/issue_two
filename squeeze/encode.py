"""
Squeeze the Scene — Encoder
============================
Strategy: Two-layer hybrid compression optimised for SAR (radar) images.

Layer 1 — Background (scene structure, sea, coastline):
  - Convert amplitude to decibels: db = 20 * log10(max(a, 1e-6))
  - Spatially downsample 4x using block-mean (suppresses multiplicative speckle noise)
  - Quantise to 8-bit (256 levels across the dB range)
  - Losslessly compress with zlib

Layer 2 — Bright targets (ships, cranes, port infrastructure):
  - Keep every pixel >= p99 of raw amplitude (top 1% = ~41 944 pixels)
  - Store exact float32 amplitude + uint16 (row, col) coordinates
  - Losslessly compress with zlib

Both layers are packed into a single binary file. The file header records all
parameters needed to reconstruct the image with no outside information.

Binary layout of out.bin:
  [0:4]   magic 'SQZ1' (4 bytes)
  [4:8]   H  uint32  image height
  [8:12]  W  uint32  image width
  [12:16] BG_FACTOR  uint32  spatial downsample factor
  [16:24] DB_MIN  float64  min dB of downsampled grid
  [24:32] DB_MAX  float64  max dB of downsampled grid
  [32:36] BG_LEN  uint32  length of compressed background blob
  [36:40] TGT_LEN uint32  length of compressed targets blob
  [40 : 40+BG_LEN]           background blob (zlib)
  [40+BG_LEN : end]          targets blob    (zlib, structured array r,c,v)
"""

import struct
import sys
import zlib

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
MAGIC = b"SQZ1"
BG_FACTOR = 4           # spatial downsample factor for background layer
TARGET_PCT = 99.0       # keep top (100 - TARGET_PCT)% pixels as explicit targets
MAX_FILE_BYTES = 524_288  # hard limit: 512 KiB = 1 bit/pixel for 2048×2048 float32
SAFETY_MARGIN = 4_096   # keep this many bytes back from the hard limit
ZLIB_LEVEL = 9


def encode(input_path: str, output_path: str) -> None:
    # ── load ──────────────────────────────────────────────────────────────────
    a = np.load(input_path)                       # float32, shape (H, W)
    H, W = a.shape

    # ── Layer 1: background ───────────────────────────────────────────────────
    db = 20.0 * np.log10(np.maximum(a, 1e-6))    # convert amplitude → decibels

    h, w = H // BG_FACTOR, W // BG_FACTOR
    # block-mean: each 4×4 block of dB values → one value (suppresses speckle)
    db_ds = db.reshape(h, BG_FACTOR, w, BG_FACTOR).mean(axis=(1, 3))

    db_min = float(db_ds.min())
    db_max = float(db_ds.max())
    db_range = db_max - db_min if db_max > db_min else 1.0

    # 8-bit quantisation: 256 levels across the full dB range
    db_ds_q = np.clip((db_ds - db_min) / db_range * 255.0, 0, 255).astype(np.uint8)
    bg_comp = zlib.compress(db_ds_q.tobytes(), ZLIB_LEVEL)

    # ── Layer 2: bright targets ───────────────────────────────────────────────
    threshold = float(np.percentile(a, TARGET_PCT))
    target_mask = a >= threshold
    n_targets = int(target_mask.sum())

    rows = np.where(target_mask)[0].astype(np.uint16)
    cols = np.where(target_mask)[1].astype(np.uint16)
    vals = a[target_mask].astype(np.float32)      # exact float32 amplitudes

    # pack as structured array: each record is (row u16, col u16, val f32) = 8 bytes
    tgt_struct = np.empty(n_targets, dtype=[("r", "<u2"), ("c", "<u2"), ("v", "<f4")])
    tgt_struct["r"] = rows
    tgt_struct["c"] = cols
    tgt_struct["v"] = vals
    tgt_comp = zlib.compress(tgt_struct.tobytes(), ZLIB_LEVEL)

    # ── size safety valve ─────────────────────────────────────────────────────
    # Header is 40 bytes; check if the total fits.
    # If not, we iteratively drop the lowest-amplitude targets until it does.
    HEADER_SIZE = 40
    budget = MAX_FILE_BYTES - SAFETY_MARGIN - HEADER_SIZE

    if len(bg_comp) + len(tgt_comp) > budget:
        # Sort targets by amplitude ascending; drop the weakest ones first.
        order = np.argsort(vals)                  # ascending amplitude
        for drop_frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            keep = int(n_targets * (1.0 - drop_frac))
            keep_idx = np.sort(order[n_targets - keep:])  # keep the brightest
            tgt_struct_trim = tgt_struct[keep_idx]
            tgt_comp_try = zlib.compress(tgt_struct_trim.tobytes(), ZLIB_LEVEL)
            if len(bg_comp) + len(tgt_comp_try) <= budget:
                tgt_comp = tgt_comp_try
                break
        else:
            # Extreme fallback: trim aggressively
            keep = max(1, int(budget * 0.4) // 8)
            keep_idx = np.sort(order[n_targets - keep:])
            tgt_comp = zlib.compress(tgt_struct[keep_idx].tobytes(), ZLIB_LEVEL)

    # ── write output ──────────────────────────────────────────────────────────
    header = struct.pack(
        "<4sIIIddII",
        MAGIC,
        H, W,
        BG_FACTOR,
        db_min, db_max,
        len(bg_comp),
        len(tgt_comp),
    )
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(bg_comp)
        f.write(tgt_comp)


if __name__ == "__main__":
    encode(sys.argv[1], sys.argv[2])
