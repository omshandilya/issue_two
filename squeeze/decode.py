"""STARTER decoder: rebuilds a flat image from the 12-byte file made by encode.py.
It must work from out.bin alone (the original image is not available here)."""
import struct
import sys

import numpy as np

rows, cols, typical = struct.unpack("<IIf", open(sys.argv[1], "rb").read())
np.save(sys.argv[2], np.full((rows, cols), typical, dtype=np.float32))
