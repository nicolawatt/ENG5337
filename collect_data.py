#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Data Collection--------------------------#
#-----------------------------------------------------------------------------#

# Drives the QBot using the standard PID line follower.
# While driving, press number keys to label and save the current camera frame.
# Images are saved into class subfolders ready for ImageFolder training.
#
# CONTROLS:
#   SPACE       arm / disarm
#   7           toggle PID line following on/off
#   U           quit and print collection summary
#
#   LABELLING (press while driving to save current frame):
#   1           straight
#   2           left
#   3           right
#   4           no_line
#   5           horizontal_line
#   6           t_junction
#   8           crossroads
#
# HOW TO USE:
#   Drive the QBot to each situation, then press the corresponding number.
#   You can hold a key to rapid-save multiple frames of the same class.
#   Aim for ~200-500 images per class minimum.
#   A green confirmation message prints each time an image is saved.
#
# OUTPUT STRUCTURE:
#   data_collection/
#       train/
#           straight/        img_0001.png ...
#           left/            img_0001.png ...
#           right/           img_0001.png ...
#           no_line/         img_0001.png ...
#           horizontal_line/ img_0001.png ...
#           t_junction/      img_0001.png ...
#           crossroads/      img_0001.png ...
#       valid/
#           (same structure — every Nth image auto-routed here)

import os
import time
import numpy as np
import cv2
from qlabs_setup import setup
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, \
    QBotPlatformCSICamera
from qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR       = "data_collection"   # root output folder
VALID_EVERY      = 5    # every Nth saved image goes to valid/, rest to train/
SAVE_EVERY_N_FRAMES = 3 # when holding a key, save every Nth frame to avoid
                        # near-duplicate images (camera runs at 60Hz)

frameRate, sampleRate = 60.0, 1/60.0

# Key → class folder name mapping
# Keys 1-8 excluding 7 (7 is reserved for PID toggle)
LABEL_MAP = {
    'k_1': 'straight',
    'k_2': 'left',
    'k_3': 'right',
    'k_4': 'no_line',
    'k_5': 'horizontal_line',
    'k_6': 't_junction',
    'k_8': 'crossroads',
}

CLASSES = list(set(LABEL_MAP.values()))

# ── Create output directories ─────────────────────────────────────────────────

for split in ['train', 'valid']:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
print("Subfolders created for all classes in train/ and valid/\n")

# ── Counters ──────────────────────────────────────────────────────────────────

# Per-class image counters (shared across train+valid for unique filenames)
class_counters = {cls: 0 for cls in CLASSES}
# Per-class split counters (to route every Nth to valid)
valid_counters  = {cls: 0 for cls in CLASSES}
total_saved     = 0

# ── Setup ─────────────────────────────────────────────────────────────────────

setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True
counter, counterDown = 0, 0
endFlag, forSpd, turnSpd = False, 0.0, 0.0
lineFollow = False
frame_since_last_save = {cls: 0 for cls in CLASSES}

# Initialise image buffers
gray_sm = np.zeros((200, 320, 1), dtype=np.uint8)
binary  = np.zeros((50,  320, 1), dtype=np.uint8)

startTime = time.time()
def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

print("=" * 60)
print("  DATA COLLECTION SCRIPT")
print("=" * 60)
print("  SPACE = arm  |  7 = toggle PID  |  U = quit & summary")
print("  1=straight  2=left  3=right  4=no_line")
print("  5=horizontal_line  6=t_junction  8=crossroads")
print("=" * 60)
print(f"  Saving every {SAVE_EVERY_N_FRAMES} frames per key hold")
print(f"  Every {VALID_EVERY}th image per class → valid/")
print("=" * 60 + "\n")

try:
    myQBot   = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam  = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    keyboard = Keyboard()
    vision   = QBPVision()
    probe    = Probe(ip=ipHost)
    probe.add_display(imageSize=[200, 320, 1], scaling=True,  scalingFactor=2, name='Raw Image')
    probe.add_display(imageSize=[50,  320, 1], scaling=False, scalingFactor=2, name='Binary Image')

    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # ── Keyboard ──────────────────────────────────────────────────────
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm            = keyboard.k_space
                lineFollow     = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # ── Drive commands ────────────────────────────────────────────────
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            else:
                commands = np.array([forSpd, turnSpd], dtype=np.float64)

            newHIL = myQBot.read_write_std(
                timestamp=time.time() - startTime,
                arm=arm,
                commands=commands
            )

            if newHIL:
                timeHIL    = time.time()
                newDownCam = downCam.read()

                if newDownCam:
                    counterDown += 1

                    # ── Image processing (for PID + probe) ────────────────────
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm     = cv2.resize(undistorted, (320, 200))
                    binary      = vision.subselect_and_threshold(gray_sm, 50, 100, 50, 255)
                    col, row, area = vision.image_find_objects(binary, 8, 50, 3000)

                    # PID speed command
                    forSpd, turnSpd = line2SpdMap.send((col, 0.4, 0.4))

                    # ── Data collection ───────────────────────────────────────
                    # Check each label key — save image if pressed
                    if newkeyboard:
                        for key_attr, cls_name in LABEL_MAP.items():
                            if getattr(keyboard, key_attr, False):

                                # Throttle saves to avoid near-duplicates
                                frame_since_last_save[cls_name] += 1
                                if frame_since_last_save[cls_name] < SAVE_EVERY_N_FRAMES:
                                    continue
                                frame_since_last_save[cls_name] = 0

                                # Route to train or valid
                                class_counters[cls_name] += 1
                                img_num  = class_counters[cls_name]
                                valid_counters[cls_name] += 1

                                if valid_counters[cls_name] % VALID_EVERY == 0:
                                    split = 'valid'
                                else:
                                    split = 'train'

                                # Save full 320x200 image — training script
                                # will crop/resize as needed
                                fname    = f"img_{img_num:05d}.png"
                                out_path = os.path.join(OUTPUT_DIR, split,
                                                        cls_name, fname)
                                cv2.imwrite(out_path, gray_sm)

                                total_saved += 1
                                print(f"  [{split:5s}] {cls_name:16s} "
                                      f"#{img_num:4d}  total={total_saved}  "
                                      f"t={t:.1f}s")

                    # ── Probe display ─────────────────────────────────────────
                    if counterDown % 4 == 0:
                        probe.send(name='Raw Image',    imageData=gray_sm)
                        probe.send(name='Binary Image', imageData=binary)

                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('\nUser interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    # ── Collection summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATA COLLECTION SUMMARY")
    print("=" * 60)
    print(f"  Total images saved: {total_saved}")
    print(f"  Output folder:      {os.path.abspath(OUTPUT_DIR)}\n")

    grand_train, grand_valid = 0, 0
    for cls in sorted(CLASSES):
        n_total = class_counters[cls]
        n_valid = n_total // VALID_EVERY
        n_train = n_total - n_valid
        grand_train += n_train
        grand_valid += n_valid
        status = "OK" if n_total >= 200 else "LOW - collect more"
        print(f"  {cls:18s}  train={n_train:4d}  valid={n_valid:4d}  "
              f"total={n_total:4d}  [{status}]")

    print(f"\n  {'TOTAL':18s}  train={grand_train:4d}  valid={grand_valid:4d}  "
          f"total={total_saved:4d}")
    print("=" * 60)
    print("\n  Copy data_collection/ to your training machine and")
    print("  update TRAIN_DIR / VALID_DIR in your training script.\n")

    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()