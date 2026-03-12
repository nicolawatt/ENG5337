#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Auto Labeller----------------------------#
#-----------------------------------------------------------------------------#

# Run this OFFLINE after capture.py to automatically label saved frames.
# Analyses each image using the same blob detection pipeline as the QBot,
# then sorts frames into class subfolders ready for ImageFolder training.
#
# LABELLING LOGIC:
#   no_line         : no blob detected
#   straight        : one blob, centroid within centre_band of image centre
#   left            : one blob, centroid left of centre_band
#   right           : one blob, centroid right of centre_band
#   horizontal_line : one blob, width >> height (aspect ratio > threshold)
#   t_junction      : exactly two blobs detected
#   crossroads      : three or more blobs detected
#
# OUTPUT:
#   data_collection/
#       train/
#           straight/ left/ right/ no_line/
#           horizontal_line/ t_junction/ crossroads/
#       valid/
#           (same — every VALID_EVERY th image per class auto-routed here)
#
# USAGE:
#   python autolabel.py
#   Optionally set RAW_DIR to point at your frames folder.
#   Optionally set REVIEW=True to preview borderline cases before saving.

import os
import cv2
import numpy as np
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR    = "raw_frames"       # folder of unlabelled frames from capture.py
OUTPUT_DIR = "data_collection"  # root output for labelled dataset
VALID_EVERY = 5                 # every Nth image per class → valid/
REVIEW      = False             # set True to preview uncertain frames in a window

# Blob detection parameters — match values used in QBot pipeline
ROW_START     = 50    # subselect rows (same as subselect_and_threshold call)
ROW_END       = 100
MIN_THRESHOLD = 50
MAX_THRESHOLD = 255
CONNECTIVITY  = 8
MIN_AREA      = 50
MAX_AREA      = 3000

# Classification thresholds — tune these if labels look wrong
IMAGE_WIDTH   = 320
IMAGE_CENTRE  = IMAGE_WIDTH // 2       # 160
CENTRE_BAND   = 30    # pixels either side of centre → classified as straight
ASPECT_RATIO_THRESHOLD = 4.0  # blob width/height > this → horizontal_line
LARGE_BLOB_WIDTH_RATIO = 0.6  # blob width > this fraction of image → crossroads

# ── Create output directories ─────────────────────────────────────────────────

CLASSES = ['straight', 'left', 'right', 'no_line',
           'horizontal_line', 't_junction', 'crossroads']

for split in ['train', 'valid']:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ── Blob detection (mirrors QBot pipeline exactly) ────────────────────────────

def detect_blobs(gray_image):
    """
    Apply same subselect + threshold + connected components as QBot pipeline.
    Returns list of (col, row, area, width, height) for each valid blob.
    """
    sub   = gray_image[ROW_START:ROW_END, :]
    _, binary = cv2.threshold(sub, MIN_THRESHOLD, MAX_THRESHOLD,
                              cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(binary, CONNECTIVITY)
    labels, ids, values, centroids = output

    blobs = []
    for idx, val in enumerate(values):
        area = val[4]
        if MIN_AREA < area < MAX_AREA:
            col    = centroids[idx][0]
            row    = centroids[idx][1]
            width  = val[cv2.CC_STAT_WIDTH]
            height = val[cv2.CC_STAT_HEIGHT]
            blobs.append((col, row, area, width, height))

    return blobs, binary

# ── Labelling logic ───────────────────────────────────────────────────────────

def assign_label(blobs):
    """
    Assign a class label based on detected blobs.

    Returns (label, confidence_str) where confidence_str is a human-readable
    description of why this label was assigned — useful for debugging.
    """
    n = len(blobs)

    if n == 0:
        return 'no_line', 'no blobs detected'

    if n >= 3:
        return 'crossroads', f'{n} blobs detected'

    if n == 2:
        return 't_junction', '2 blobs detected'

    # n == 1 — single blob
    col, row, area, width, height = blobs[0]

    # Check for horizontal line: very wide relative to height
    aspect = width / max(height, 1)
    if aspect > ASPECT_RATIO_THRESHOLD:
        return 'horizontal_line', f'aspect ratio={aspect:.1f}'

    # Check lateral position
    offset = col - IMAGE_CENTRE
    if abs(offset) <= CENTRE_BAND:
        return 'straight', f'offset={offset:.1f}px (within centre band)'
    elif offset < 0:
        return 'left', f'offset={offset:.1f}px (left of centre)'
    else:
        return 'right', f'offset={offset:.1f}px (right of centre)'

# ── Main labelling loop ───────────────────────────────────────────────────────

raw_paths = sorted(Path(RAW_DIR).glob("*.png"))
total     = len(raw_paths)

if total == 0:
    print(f"\nNo .png files found in '{RAW_DIR}'.")
    print("Run capture.py first to collect frames.\n")
    exit()

print(f"\n{'='*60}")
print(f"  AUTO LABELLER")
print(f"{'='*60}")
print(f"  Input : {os.path.abspath(RAW_DIR)}  ({total} frames)")
print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
print(f"  Valid split: every {VALID_EVERY}th image per class")
print(f"{'='*60}\n")

class_counters = {cls: 0 for cls in CLASSES}
valid_counters  = {cls: 0 for cls in CLASSES}
total_saved     = 0
skipped         = 0

for i, fpath in enumerate(raw_paths):
    img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        skipped += 1
        continue

    # Resize to 320x200 in case capture saved different size
    img = cv2.resize(img, (320, 200))

    blobs, binary = detect_blobs(img)
    label, reason = assign_label(blobs)

    # ── Optional review of borderline cases ───────────────────────────────────
    if REVIEW:
        is_borderline = (
            label == 'straight' and abs(blobs[0][0] - IMAGE_CENTRE
                                        if blobs else 0) > CENTRE_BAND * 0.7
        )
        if is_borderline:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(display, f"{label} ({reason})", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Review — press Y to accept, any other key to skip', display)
            key = cv2.waitKey(0)
            if key != ord('y') and key != ord('Y'):
                skipped += 1
                continue

    # ── Route to train or valid ───────────────────────────────────────────────
    class_counters[label] += 1
    valid_counters[label]  += 1
    img_num = class_counters[label]

    split = 'valid' if valid_counters[label] % VALID_EVERY == 0 else 'train'

    fname    = f"img_{img_num:05d}.png"
    out_path = os.path.join(OUTPUT_DIR, split, label, fname)
    cv2.imwrite(out_path, img)
    total_saved += 1

    # Progress every 100 frames
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{total}  saved={total_saved}  "
              f"skipped={skipped}")

if REVIEW:
    cv2.destroyAllWindows()

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  LABELLING COMPLETE")
print(f"{'='*60}")
print(f"  Frames processed : {total}")
print(f"  Images saved     : {total_saved}")
print(f"  Skipped          : {skipped}")
print(f"\n  {'Class':20s}  {'Train':>6}  {'Valid':>6}  {'Total':>6}  Status")
print(f"  {'-'*55}")

for cls in CLASSES:
    n_total = class_counters[cls]
    n_valid = n_total // VALID_EVERY
    n_train = n_total - n_valid
    status  = "OK" if n_total >= 200 else "LOW — collect more"
    print(f"  {cls:20s}  {n_train:>6}  {n_valid:>6}  {n_total:>6}  {status}")

print(f"\n  Update TRAIN_DIR and VALID_DIR in your training script to:")
print(f"    TRAIN_DIR = '{os.path.abspath(OUTPUT_DIR)}/train'")
print(f"    VALID_DIR = '{os.path.abspath(OUTPUT_DIR)}/valid'")
print(f"{'='*60}\n")