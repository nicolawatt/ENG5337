#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Frame Capture----------------------------#
#-----------------------------------------------------------------------------#

# Runs the standard PID line follower and automatically saves raw camera
# frames to disk. No labelling required during capture.
# Run autolabel.py afterwards to sort frames into class folders.
#
# CONTROLS:
#   SPACE       arm / disarm
#   7           toggle PID line following on/off
#   R           start / stop recording frames
#   U           quit and print capture summary
#
# OUTPUT:
#   raw_frames/
#       frame_00001.png
#       frame_00002.png
#       ...

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

OUTPUT_DIR   = "raw_frames"    # folder to save frames into
SAVE_EVERY_N = 6               # save every Nth frame (~10Hz at 60Hz camera)
                               # increase to reduce duplicates on straights
                               # decrease to capture more junction detail

frameRate, sampleRate = 60.0, 1/60.0

# ── Create output directory ───────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")

# ── Setup ─────────────────────────────────────────────────────────────────────

setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True
counterDown = 0
endFlag, forSpd, turnSpd = False, 0.0, 0.0
lineFollow  = False
recording   = False
frame_count = 0
saved_count = 0

gray_sm = np.zeros((200, 320, 1), dtype=np.uint8)
binary  = np.zeros((50,  320, 1), dtype=np.uint8)

startTime = time.time()
def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

print("\n" + "=" * 55)
print("  FRAME CAPTURE")
print("=" * 55)
print("  SPACE=arm  7=PID toggle  R=record toggle  U=quit")
print(f"  Saving every {SAVE_EVERY_N} frames when recording")
print("=" * 55 + "\n")

try:
    myQBot   = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam  = QBotPlatformCSICamera(frameRate=frameRate,
                                     exposure=39.0, gain=17.0)
    keyboard = Keyboard()
    vision   = QBPVision()
    probe    = Probe(ip=ipHost)
    probe.add_display(imageSize=[200, 320, 1], scaling=True,
                      scalingFactor=2, name='Raw Image')
    probe.add_display(imageSize=[50,  320, 1], scaling=False,
                      scalingFactor=2, name='Binary Image')

    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate,
                                           saturation=75)
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

                # R key toggles recording
                if getattr(keyboard, 'k_r', False):
                    recording = not recording
                    state = "STARTED" if recording else "PAUSED"
                    print(f"  Recording {state} — frames saved so far: {saved_count}")

            # ── Drive commands ────────────────────────────────────────────────
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]],
                                    dtype=np.float64)
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

                    # ── Image processing ──────────────────────────────────────
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm     = cv2.resize(undistorted, (320, 200))
                    binary      = vision.subselect_and_threshold(
                                    gray_sm, 50, 100, 50, 255)
                    col, row, area = vision.image_find_objects(
                                        binary, 8, 50, 3000)
                    forSpd, turnSpd = line2SpdMap.send((col, 0.4, 0.4))

                    # ── Save frame ────────────────────────────────────────────
                    if recording and counterDown % SAVE_EVERY_N == 0:
                        frame_count += 1
                        saved_count += 1
                        fname    = f"frame_{frame_count:05d}.png"
                        out_path = os.path.join(OUTPUT_DIR, fname)
                        cv2.imwrite(out_path, gray_sm)

                        if saved_count % 50 == 0:
                            print(f"  Saved {saved_count} frames  t={t:.1f}s")

                    # ── Probe ─────────────────────────────────────────────────
                    if counterDown % 4 == 0:
                        probe.send(name='Raw Image',    imageData=gray_sm)
                        probe.send(name='Binary Image', imageData=binary)

                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('\nUser interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    print(f"\n{'='*55}")
    print(f"  Capture complete — {saved_count} frames saved")
    print(f"  Location: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Next step: run  python autolabel.py")
    print(f"{'='*55}\n")

    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()