"""STARTER (valid but very weak): stores only the image size and one typical brightness value.
It shows the interface and the file handling. Replace everything below with your own idea."""
import struct
import sys

import numpy as np

a = np.load(sys.argv[1])                       # float32 amplitude image, shape (2048, 2048)
typical = float(np.median(a))
with open(sys.argv[2], "wb") as f:             # the whole "compressed file" is 12 bytes
    f.write(struct.pack("<IIf", a.shape[0], a.shape[1], typical))
