# Squeeze the Scene

> **Make this radar image 32 times smaller. When we rebuild it on the ground, it must still look right, and every bright target in it, ships above all, must still be there.**

You have a satellite that takes radar images. They're far too large to send down to Earth during its few minutes of contact with the ground station. Your job is to write a **compressor** and a **decompressor** that make each image **at least 32× smaller** while keeping as much as possible of what matters.

There is no single right answer. How you do it is up to you.

---

## 1. What you get

```
README.md              this file
practice/
  practice_1.npy       three practice images
  practice_2.npy
  practice_3.npy
starter/
  encode.py            a tiny, valid-but-weak example showing the interface
  decode.py
```

Each practice file is a NumPy array: **2048 × 2048, 32-bit float**, 16,777,216 bytes of image data. Load it with `np.load("practice_1.npy")`.

A `.npy` file is NumPy's format for saving a grid of numbers (here 2048 × 2048 pixel values) to disk. It is not a picture file, so you cannot double-click it. To look at one, run this (it needs `pip install matplotlib`, which is only for looking: your encoder and decoder must not use it):

```python
import numpy as np, matplotlib.pyplot as plt
a = np.load("practice/practice_1.npy")     # run this from the problem2-squeeze-the-scene folder
db = 20 * np.log10(np.maximum(a, 1e-6))          # decibels: the raw values are too dark to see
plt.imshow(db, cmap="gray", vmin=np.percentile(db, 1), vmax=np.percentile(db, 99.9))
plt.show()
```

We keep **other images secret**. Your final score comes from running your code on those, not on the practice images.

## 2. What the images are

- They are **real radar images** from a commercial radar satellite, cut from larger scenes: ports, quays and open sea.
- Each pixel is an **amplitude**: how strongly the radar echo came back. Strong echo = **bright** (metal ships, cranes, buildings). Weak echo = **dark** (calm water).
- Two facts you should think about:
  1. **The brightness range is enormous.** In one image the strongest pixel can be around 500,000 times stronger (in energy) than a typical sea pixel.
  2. **The images are grainy.** Radar images have a random speckle pattern that is part of the data. It looks like noise, and noise is expensive to store.
- Real-world context: the satellite can only talk to the ground for a few minutes per pass, so what it sends has to fit.

## 3. What you write

Two programs, in the folder `squeeze/` (next to this README):

```
python encode.py  input.npy   out.bin       # makes the small file
python decode.py  out.bin     rebuilt.npy    # rebuilds the image from ONLY out.bin
```

You may add helper `.py` files next to them.

### The rules

| Rule | Detail |
|---|---|
| **Size limit** | `out.bin` must be **at most 524,288 bytes** (that is 1/32 of the image data: about **1 bit per pixel**). **Over the limit scores 0 for that image.** |
| **The decoder sees only the small file** | We run `decode.py` in an **empty folder containing just `out.bin`**. Everything needed to rebuild the image must be inside `out.bin`. No reading the original image, no network, no other files. |
| **The encoder** | We run `encode.py` in an empty folder containing just `input.npy`. |
| **Output of the decoder** | `rebuilt.npy`: same shape as the input (2048 × 2048), **32-bit float**, every value finite (no `NaN`, no infinity). Anything else scores 0. |
| **Allowed libraries** | The Python standard library (including `zlib`, `lzma`, `bz2`), **NumPy** and **SciPy**. We run Python 3.13. |
| **Not allowed** | Any ready-made image or video compressor or codec: Pillow, OpenCV, imageio, scikit-image, JPEG / JPEG 2000 / WebP / AVIF libraries, or calling any external program. Writing your own transform or coder is the point. |
| **Time** | At most **60 seconds** for `encode.py` and **60 seconds** for `decode.py`, per image, on an ordinary laptop. Leave yourself a safety margin. |
| **Memory** | At most **2 GB**. |
| **Code size** | All your `.py` files together must be **under 200 KB**. Code is for the method, not for hiding image data inside it. |
| **Fair play** | Your code must not try to read the secret images, any file outside its folder, or the network. Trying is disqualifying. |

### Try it before you build anything

You need Python 3.11 or newer and two packages: `pip install numpy scipy`. Nothing else has to be installed or run.

1. Copy `encode.py` and `decode.py` from `starter/` into `squeeze/`.
2. Open a terminal **inside the `squeeze/` folder** and run:

```
python encode.py ../practice/practice_1.npy out.bin
python decode.py out.bin rebuilt.npy
```

3. Look at the size of `out.bin` (it must be at most 524,288 bytes) and load `rebuilt.npy` with `np.load`. The starter's file is only 12 bytes and its image is flat, which scores badly: now improve it. Do the same on all three practice images.

## 4. How you are scored

Scored automatically on secret images: how many bright targets (ships and similar) your rebuilt image keeps, and how close it looks to the original. You are not given the scoring program, and a file over the size limit scores 0.

## 5. Notes

- **AI tools are allowed.** State in `SUBMISSION.md` which tools you used, where they helped, and **one time the AI was wrong and how you caught it**. Sharing your AI conversation history is optional.
- Because you don't get our scorer, **building your own measurements is part of the task**. Check that your decoder really runs from `out.bin` alone, that your files stay under the size limit **on every practice image**, and that your time stays well under 60 s.
- A solution tuned only to the practice images can fail on the secret ones, especially if the size limit is exceeded on an image you haven't seen.
- The starter shows how files are read and written. It works, but it throws away almost everything; it is only a starting point.

## 6. What to hand in

In the folder `problem2-squeeze-the-scene/squeeze/`:

- `encode.py` and `decode.py` (plus any helper files);
- section 3 of the `SUBMISSION.md` in the main folder (your approach, what did not work, how you checked your results), plus the AI note in its section 4.

