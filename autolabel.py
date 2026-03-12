import os
import cv2
import math
import random
import shutil
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
INPUT_DIR = "raw_frames"          # folder containing unlabeled images
OUTPUT_DIR = "dataset"        # output dataset folder
VAL_RATIO = 0.2               # 20% validation split
RANDOM_SEED = 42

CLASSES = ["right", "left", "vertical", "horizontal", "junction", "no line"]

# Thresholds you may need to tune for your images
BIN_THRESHOLD = 180           # threshold for white line extraction
MIN_COMPONENT_AREA = 150      # ignore tiny blobs
MIN_LINE_LENGTH = 40          # ignore very short fitted lines
JUNCTION_MIN_COMPONENTS = 2   # if >= this many major line parts -> junction


# ----------------------------
# UTILS
# ----------------------------
def ensure_dirs(base_dir):
    for split in ["train", "val"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]


def copy_to_split(src_path, dst_root, split, label):
    dst_path = os.path.join(dst_root, split, label, src_path.name)
    shutil.copy2(src_path, dst_path)


# ----------------------------
# IMAGE ANALYSIS
# ----------------------------
def extract_white_mask(gray):
    # Threshold bright line on dark background
    _, mask = cv2.threshold(gray, BIN_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Clean up small noise
    kernel = np_kernel(3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def np_kernel(k):
    import numpy as np
    return np.ones((k, k), dtype=np.uint8)


def component_stats(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_COMPONENT_AREA:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        comp_mask = (labels == i).astype("uint8") * 255
        components.append({
            "label": i,
            "area": area,
            "bbox": (x, y, w, h),
            "mask": comp_mask
        })
    return components


def fit_line_angle_and_length(comp_mask):
    ys, xs = cv2.findNonZero(comp_mask).reshape(-1, 2)[:, 1], cv2.findNonZero(comp_mask).reshape(-1, 2)[:, 0]

    points = cv2.findNonZero(comp_mask)
    if points is None or len(points) < 2:
        return None, 0

    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    vx = float(line[0][0])
    vy = float(line[1][0])

    # angle in degrees, normalized to [-90, 90]
    angle = math.degrees(math.atan2(vy, vx))
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180

    # Approx line length from bounding box diagonal
    x, y, w, h = cv2.boundingRect(points)
    length = math.hypot(w, h)

    return angle, length


def is_vertical(angle, tol=20):
    return abs(abs(angle) - 90) < tol


def is_horizontal(angle, tol=20):
    return abs(angle) < tol


def is_right_diagonal(angle, tol=25):
    # "/" shape in image coordinates often gives negative angle
    return abs(angle + 45) < tol


def is_left_diagonal(angle, tol=25):
    # "\" shape in image coordinates often gives positive angle
    return abs(angle - 45) < tol


def classify_image(img_path, debug=False):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return "no line"

    mask = extract_white_mask(gray)
    components = component_stats(mask)

    if len(components) == 0:
        return "no line"

    # Fit line to each component
    line_like = []
    for comp in components:
        angle, length = fit_line_angle_and_length(comp["mask"])
        if angle is None or length < MIN_LINE_LENGTH:
            continue
        line_like.append({
            "angle": angle,
            "length": length,
            "area": comp["area"],
            "bbox": comp["bbox"]
        })

    if len(line_like) == 0:
        return "no line"

    # Sort by significance
    line_like.sort(key=lambda d: (d["area"], d["length"]), reverse=True)

    # If multiple major directions/components -> junction
    if len(line_like) >= JUNCTION_MIN_COMPONENTS:
        major = line_like[:3]
        direction_bins = set()

        for item in major:
            a = item["angle"]
            if is_vertical(a):
                direction_bins.add("vertical")
            elif is_horizontal(a):
                direction_bins.add("horizontal")
            elif is_right_diagonal(a):
                direction_bins.add("right")
            elif is_left_diagonal(a):
                direction_bins.add("left")

        if len(direction_bins) >= 2:
            return "junction"

    # Otherwise classify by the largest component
    a = line_like[0]["angle"]

    if is_vertical(a):
        return "vertical"
    if is_horizontal(a):
        return "horizontal"
    if is_right_diagonal(a):
        return "right"
    if is_left_diagonal(a):
        return "left"

    return "no line"


# ----------------------------
# MAIN
# ----------------------------
def main():
    random.seed(RANDOM_SEED)

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        return

    ensure_dirs(output_dir)

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    results = []
    for img_path in image_paths:
        label = classify_image(img_path)
        results.append((img_path, label))
        print(f"{img_path.name} -> {label}")

    random.shuffle(results)
    val_count = int(len(results) * VAL_RATIO)

    val_set = set(img for img, _ in results[:val_count])

    for img_path, label in results:
        split = "val" if img_path in val_set else "train"
        copy_to_split(img_path, output_dir, split, label)

    print("\nDone.")
    print(f"Created dataset at: {output_dir.resolve()}")
    print("Folder structure:")
    print("dataset/")
    print("  train/")
    print("    right/ left/ vertical/ horizontal/ junction/ no line/")
    print("  val/")
    print("    right/ left/ vertical/ horizontal/ junction/ no line/")


if __name__ == "__main__":
    main()