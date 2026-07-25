# Getting Started — for regular computer users

This guide assumes **no programming experience**. It walks you through installing the free tools
this project needs, and running your first drawing check. Expect it to take 20–30 minutes the
first time.

---

## What this project does, in plain terms

You draw an object on paper the way a drafter would — a front view, a top view, a side view —
scan or photograph the drawings, and this software:

1. **Measures your drawings** and tells you the object's real-world size.
2. **Cross-checks your views against each other** — if your top view says the object is 65 mm
   deep but your side view says 60 mm, it tells you, with the exact percentage, *before* anything
   gets built from the wrong number.
3. **Tells you how much of the shape your drawings actually pin down**, and how much a computer
   would have to guess.
4. *(Eventually)* generates a full 3D model from the drawings, at true physical size.

### What works today vs. what needs special hardware

| | What you need |
|---|---|
| **Drawing checker** (steps 1–3 above) | Any ordinary computer — Windows, Mac, or Linux. This guide. |
| **3D model generation** (step 4) | A Linux computer with a large NVIDIA graphics card (24 GB+ of video memory). Most people don't have this — see the last section. |

The drawing checker is genuinely useful on its own: it's a drafting-consistency checker for
hand drawings, and everything it tells you is measured, not guessed.

---

## Step 1 — Install Python

Python is the free programming language this project runs on. You need version **3.10 or newer**.

**Windows:**
1. Go to <https://www.python.org/downloads/> and click the yellow download button.
2. Run the installer. **Important:** on the first screen, tick the box that says
   **"Add python.exe to PATH"** before clicking Install. If you miss this, nothing below will work.
3. When it finishes, open the **Command Prompt** (press the Windows key, type `cmd`, press Enter)
   and type:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`. If you see an error, restart the computer and try again.

**Mac:**
1. Go to <https://www.python.org/downloads/> and download the macOS installer. Run it.
2. Open the **Terminal** app (press Cmd+Space, type `terminal`, press Enter) and type:
   ```
   python3 --version
   ```

**Linux:** Python is almost certainly already installed. Check with `python3 --version`.

> From here on, this guide writes `python`. On Mac and Linux, type `python3` instead wherever
> you see it.

---

## Step 2 — Download this project

**Easiest way (no extra tools):**
1. On this project's GitHub page, click the green **Code** button, then **Download ZIP**.
2. Unzip it somewhere you can find again — for example your Documents folder. You'll end up with
   a folder called something like `TRELLIS.2-ORTHO`.

**If you have git installed** (optional, better for getting updates later):
```
git clone https://github.com/definitelynotaspren/TRELLIS.2-ORTHO.git
```

---

## Step 3 — Open a command window *in the project folder*

Every command below must be typed inside the project folder.

- **Windows:** open the `TRELLIS.2-ORTHO` folder in File Explorer, click in the address bar at
  the top, type `cmd`, and press Enter. A black command window opens, already in the right place.
- **Mac:** open Terminal, type `cd ` (with a space after it), drag the `TRELLIS.2-ORTHO` folder
  from Finder onto the Terminal window, and press Enter.
- **Linux:** `cd` into the folder.

---

## Step 4 — Install the three libraries the checker needs

In the command window, type:

```
python -m pip install numpy scipy pillow
```

This downloads three standard, widely-used free libraries (for math, science math, and image
reading). It takes a minute or two. Warnings in yellow are fine; red errors are not — see
Troubleshooting below.

**Optional but recommended — verify everything works:**

```
python -m pip install pytest
python -m pytest tests/ -q
```

After a few seconds you should see a line ending in **`32 passed`**. If you do, your setup is
correct and everything below will work.

---

## Step 5 — Prepare your drawings

1. **Draw each view on its own sheet** (or crop each view into its own image file). Dark lines on
   light paper work best.
2. **Scan or photograph each sheet** and save as PNG or JPG. A flatbed scanner is best because
   the scale is exact; a phone photo works if taken square-on.
3. **Name the files by view** so you don't mix them up: `front.png`, `top.png`, `right.png`, etc.

### The views, so we're speaking the same language

Imagine the object in a glass box:
- **front** — looking at its face (shows width and height)
- **top** — looking straight down (shows width and depth)
- **right** / **left** — looking at its side (shows depth and height)
- **back**, **bottom** — the remaining two, rarely needed

You need **at least two views from different directions** (e.g. front + top). Three is better —
with three, every dimension is measured twice, which is what lets the checker catch mistakes.

### Work out your "units per pixel" (one small calculation)

The checker needs to know how big one pixel of your scan is in the real world.

**If you scanned at a known DPI and drew at full size (1:1):**

> units-per-px = 25.4 ÷ DPI  (for millimetres)

- Scanned at 300 DPI → each pixel is 25.4 ÷ 300 = **0.0847 mm**
- Scanned at 150 DPI → **0.1693 mm**

**If you drew at a scale** (say 2:1, drawing twice actual size): divide the number above by 2.

**If you photographed it / don't know:** measure something on the drawing you know the true size
of. If a 100 mm edge is 800 pixels long in the image (any image editor shows pixel positions),
then units-per-px = 100 ÷ 800 = **0.125 mm**.

Use the same value for all views if they were all scanned the same way.

---

## Step 6 — Run the checker

In the command window (still in the project folder), all on one line:

```
python examples/check_drawings.py --front front.png --top top.png --right right.png --units mm --units-per-px 0.0847
```

Swap in your own file names, units (`mm`, `cm`, or `in`), and units-per-px number. Skip any view
you don't have — two views is the minimum.

### Reading the output

Here's a real run where the top view was drawn 5 mm too deep:

```
Measured size (width x depth x height):
  100.0 x 62.5 x 40.0 mm
  X axis measured by 2 views, they disagree by 0.0% -- OK
  Y axis measured by 2 views, they disagree by 8.0% -- CHECK YOUR DRAWINGS
  Z axis measured by 2 views, they disagree by 0.0% -- OK

How well your views pin down the shape:
  well-determined (seen by 2+ view directions): 100%
  ...

Warnings -- generation could proceed, but double-check:
  [!] Y axis: views disagree by 8.0% (sources: right, top).
```

- **Measured size** — the object's real-world dimensions, reconciled across all your views.
- **"disagree by X%"** — how much two views that measure the same dimension differ. Under 3% is
  normal hand-drawing wobble. More than that means one of the named views is drawn wrong — the
  `sources:` list tells you which two to compare.
- **BLOCKED items** (`[X]`) — things that must be fixed (e.g. you forgot `--units`, or all your
  views face the same direction so a dimension is completely unmeasured).
- **Warnings** (`[!]`) — worth a look, not fatal.

Fix, re-scan, re-run. When it says **"Ready to generate."**, your drawing set is internally
consistent.

---

## About full 3D generation

Generating the actual 3D model uses a 4-billion-parameter AI model and requires:

- **Linux** (not Windows or Mac),
- an **NVIDIA graphics card with at least 24 GB of video memory** (e.g. RTX 3090/4090, A100),
- and roughly 30 GB of downloads (the AI model weights).

If you have that hardware, the installation is scripted — see the *Installation* section of
[README.md](README.md) and run `. ./setup.sh --new-env --basic ...` as described there.

**One honest caveat:** as of this writing, the drawing checker and the 3D generator are not yet
connected — the piece that feeds your checked, correctly-scaled drawings into the generator is
the next item on the roadmap (see [PHASE0_REVIEW.md](PHASE0_REVIEW.md)). Today the generator
accepts a single photo/image the same way the original Microsoft project does, and the checker
is a standalone tool. This guide will be updated when they're wired together.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized...` (Windows) | Python isn't on your PATH. Re-run the installer, choose *Modify*, and tick *Add python to environment variables* — or just reinstall with the PATH box ticked. |
| `No module named numpy` (or scipy/pillow) | Step 4 didn't finish, or installed into a different Python. Run `python -m pip install numpy scipy pillow` again — using `python -m pip` (not plain `pip`) guarantees it installs into the same Python you're running. |
| `No ink found in sheet -- image appears blank` | The checker couldn't find your drawing in the image. Usually the scan is too faint — increase scanner contrast, or trace the outline with a darker pen. |
| `error: argument --units is required` | You must always state the units. There's deliberately no default — a silent mm/inch mix-up is a 25.4× error. |
| Sizes come out ~25× too big or small | Your units-per-px is in the wrong unit system. Redo the Step 5 calculation. |
| Command window says `Permission denied` on Mac | Use `python3`, not `python`, everywhere. |

If you get stuck on something not listed here, open an issue on the project's GitHub page and
paste the exact command you typed and the exact error message.
