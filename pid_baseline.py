#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - PID Baseline + Metrics-------------------#
#-----------------------------------------------------------------------------#

# Imports
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, \
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
import time
import numpy as np
import cv2
import csv
from qlabs_setup import setup

# =============================================================================
# METRICS EXPLANATION
# -----------------------------------------------------------------------------
# 1. Cross-Track Error (CTE)
#    The horizontal pixel distance between the detected line centroid and the
#    image centre column. A centred robot gives CTE=0. Larger magnitude = more
#    deviation. We record mean absolute CTE and std dev over the run.
#
# 2. Control Smoothness (Jerk proxy)
#    The frame-to-frame change in turn speed command |Δturn_spd| per second.
#    Smooth control → small values. Oscillating/jerky control → large values.
#    Gives a direct measure of how aggressively the controller corrects.
#
# 3. Line-Loss Rate
#    Fraction of frames in which no line blob was detected (col == -1 or
#    area below threshold). High line-loss rate → poor robustness.
#    Expressed as a percentage of total frames processed.
# =============================================================================

# ── Config ────────────────────────────────────────────────────────────────────

METRICS_CSV   = "pid_baseline_metrics.csv"   # per-frame log
SUMMARY_FILE  = "pid_baseline_summary.txt"   # end-of-run summary
IMAGE_CENTRE  = 160                          # centre column of 320px wide image

# ── Setup ─────────────────────────────────────────────────────────────────────

setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
commands  = np.zeros((2), dtype=np.float64)
arm, noKill = 0, True
frameRate, sampleRate = 60.0, 1/60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0
prevTurnSpd = 0.0

startTime = time.time()
def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

# FIX: initialise before loop — assigned inside keyboard handler only
lineFollow = False

# FIX: initialise image buffers so probe never sends uninitialised data
gray_sm = np.zeros((200, 320, 1), dtype=np.uint8)
binary  = np.zeros((50,  320, 1), dtype=np.uint8)

# ── Metrics accumulators ──────────────────────────────────────────────────────

cte_list          = []   # cross-track error per frame (signed, pixels)
smoothness_list   = []   # |Δturn_spd| per frame
line_loss_count   = 0    # frames with no line detected
frames_processed  = 0    # total frames processed

# Per-frame CSV log — useful for plotting in your presentation
csv_file   = open(METRICS_CSV, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp_s', 'cte_px', 'for_spd', 'turn_spd',
                     'delta_turn_spd', 'line_detected'])

# ── Main ──────────────────────────────────────────────────────────────────────

try:
    # Section B - Initialisation
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

    print("\n[BASELINE] PID line follower running. Press U to stop.\n")
    print(f"{'Time':>6}  {'CTE(px)':>8}  {'ForSpd':>7}  {'TurnSpd':>8}  "
          f"{'dTurn':>7}  {'Line?':>6}")
    print("─" * 55)

    # Main loop
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # Keyboard driver
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm        = keyboard.k_space
                lineFollow = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # Section C - Toggle line following
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            else:
                commands = np.array([forSpd, turnSpd], dtype=np.float64)

            # QBot hardware read/write
            newHIL = myQBot.read_write_std(
                timestamp=time.time() - startTime,
                arm=arm,
                commands=commands
            )

            if newHIL:
                timeHIL    = time.time()
                newDownCam = downCam.read()

                if newDownCam:
                    counterDown    += 1
                    frames_processed += 1

                    # ── Section D - Image processing ──────────────────────────

                    # D.1 - Undistort and resize
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm     = cv2.resize(undistorted, (320, 200))

                    # D.2 - Threshold and blob detection
                    binary = vision.subselect_and_threshold(gray_sm, 50, 100, 50, 255)
                    col, row, area = vision.image_find_objects(binary, 8, 500, 2000)
                    # image_find_objects returns (None,None,None) when no blob found
                    if col is None:
                        col, row, area = -1, -1, 0

                    # ── Metric 1: Cross-Track Error ────────────────────────────
                    line_detected = (col != -1 and area > 50)

                    if line_detected:
                        cte = col - IMAGE_CENTRE
                    else:
                        cte = None
                        line_loss_count += 1

                    if cte is not None:
                        cte_list.append(cte)

                    # ── D.3 - PID speed command ────────────────────────────────
                    kP = 0.4
                    kD = 0.4
                    forSpd, turnSpd = line2SpdMap.send((col if col != -1 else None, kP, kD))

                    # ── Metric 2: Control Smoothness ───────────────────────────
                    delta_turn = abs(turnSpd - prevTurnSpd) / sampleRate
                    smoothness_list.append(delta_turn)
                    prevTurnSpd = turnSpd

                    # ── Log to CSV ─────────────────────────────────────────────
                    csv_writer.writerow([
                        round(t, 3),
                        round(cte, 2) if cte is not None else 'NaN',
                        round(forSpd, 4),
                        round(turnSpd, 4),
                        round(delta_turn, 4),
                        1 if line_detected else 0
                    ])

                    # Console log every 20 frames
                    if counterDown % 20 == 0:
                        print(f"{t:6.1f}  "
                              f"{cte if cte is not None else 'LOST':>8}  "
                              f"{forSpd:7.3f}  {turnSpd:8.3f}  "
                              f"{delta_turn:7.3f}  "
                              f"{'YES' if line_detected else 'NO':>6}")

                    # FIX: probe send inside newDownCam — only fires after
                    # real frame data exists
                    if counterDown % 4 == 0:
                        probe.send(name='Raw Image',    imageData=gray_sm)
                        probe.send(name='Binary Image', imageData=binary)

                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('\nUser interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    # ── End-of-run summary ─────────────────────────────────────────────────────
    csv_file.close()

    cte_arr        = np.array(cte_list) if cte_list else np.array([0.0])
    smooth_arr     = np.array(smoothness_list) if smoothness_list else np.array([0.0])
    line_loss_rate = (line_loss_count / frames_processed * 100) if frames_processed > 0 else 0.0

    summary = f"""
=============================================================
  PID BASELINE — END OF RUN SUMMARY
=============================================================
  Frames processed     : {frames_processed}
  Run duration         : {elapsed_time():.1f} s

  --- Metric 1: Cross-Track Error (CTE) ---
  Mean absolute CTE    : {np.mean(np.abs(cte_arr)):.2f} px
  Std dev CTE          : {np.std(cte_arr):.2f} px
  Max absolute CTE     : {np.max(np.abs(cte_arr)):.2f} px

  --- Metric 2: Control Smoothness ---
  Mean |dTurn| /s      : {np.mean(smooth_arr):.4f} rad/s2
  Std dev smoothness   : {np.std(smooth_arr):.4f}
  (lower = smoother)

  --- Metric 3: Line-Loss Rate ---
  Frames with no line  : {line_loss_count} / {frames_processed}
  Line-loss rate       : {line_loss_rate:.2f} %

  Per-frame log saved  : {METRICS_CSV}
=============================================================
"""
    print(summary)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved → {SUMMARY_FILE}")

    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()