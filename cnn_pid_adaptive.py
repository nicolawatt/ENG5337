from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, \
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe
import time
import numpy as np
import cv2
import csv
from qlabs_setup import setup

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# =============================================================================
# CNN + PID HYBRID — HOW IT WORKS
# -----------------------------------------------------------------------------
# 1. Downward camera frame → preprocess (grayscale, resize 128x128, normalise)
# 2. MRI_CNN forward pass → 4-class logits
# 3. Softmax → probability vector [P(left), P(off_line), P(right), P(straight)]
# 4. Soft error = P(right)*1.0 + P(left)*-1.0  (straight/off_line = 0.0)
#    This gives a continuous error in ~[-1,1] proportional to CNN confidence
#    rather than a hard class lookup — smoother PID response
# 5. PID acts on soft error → turnSpd
# 6. off_line detection: P(off_line) > threshold → slow creep, reset integrator
# =============================================================================

# cnn architecture
class MRI_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1a = nn.Conv2d(1,  32,  3, padding=1); self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32,  3, padding=1); self.bn1b = nn.BatchNorm2d(32)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv2a = nn.Conv2d(32, 64,  3, padding=1); self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64,  3, padding=1); self.bn2b = nn.BatchNorm2d(64)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.conv3a = nn.Conv2d(64,  128, 3, padding=1); self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1); self.bn3b = nn.BatchNorm2d(128)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = 'best_mri_cnn.pt'
DEVICE     = torch.device('cpu')   # CPU for deployment stability on Jetson

# ImageFolder sorts alphabetically → indices are:
# 0=left, 1=off_line, 2=right, 3=straight
CLASSES = ['left', 'off_line', 'right', 'straight']
ERROR_WEIGHTS = {
    'left':     -1.0,   # robot left of line → steer right
    'off_line':  0.0,   # handled separately
    'right':    +1.0,   # robot right of line → steer left
    'straight':  0.0,   # centred → no correction
}

# Normalisation stats — paste values printed by training script here
# e.g. "Training set mean: 0.4521  Training set std: 0.2134"
TRAIN_MEAN = 0.0630
TRAIN_STD  = 0.2374

OFF_LINE_THRESHOLD = 0.6   # P(off_line) above this → trigger off-line behaviour

# PID gains
Kp = 0.4
Ki = 0.01
Kd = 0.08
INTEGRAL_LIMIT = 1.0

# Speed
BASE_SPEED     = 0.35   # matches original forSpd range from line_to_speed_map
OFF_LINE_SPEED = 0.05   # slow creep when line is lost

# load model
print(f"Loading model from {MODEL_PATH} ...")
cnn_model = MRI_CNN(num_classes=4).to(DEVICE)
cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
cnn_model.eval()
print("Model loaded.")

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[TRAIN_MEAN], std=[TRAIN_STD])
])

#cnn interface
def get_soft_error(frame_bgr):
    """
    Run CNN on a raw camera frame (numpy array from downCam.imageData).
    Returns:
        soft_error   : float ~[-1, 1], fed into PID as error signal
        off_line_prob: float [0, 1], triggers off-line behaviour if > threshold
        pred_class   : str, top predicted class for logging
        confidence   : float [0, 1], derived from softmax entropy
                       1.0 = fully confident, 0.0 = maximally uncertain
                       Used to scale PID gains adaptively.
    """
    img    = Image.fromarray(frame_bgr).convert('L')
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = cnn_model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Soft error: weighted sum of probabilities
    soft_err      = sum(probs[i] * ERROR_WEIGHTS[cls] for i, cls in enumerate(CLASSES))
    off_line_prob = float(probs[CLASSES.index('off_line')])
    pred_class    = CLASSES[int(np.argmax(probs))]

    # ── Confidence from Shannon entropy ───────────────────────────────────────
    # H = -sum(p * log(p)), maximum entropy for 4 classes = log(4) ≈ 1.386
    # Normalise to [0,1] then invert so confidence=1 means certain
    n_classes   = len(CLASSES)
    max_entropy = np.log(n_classes)
    # Clip probs to avoid log(0)
    entropy     = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)))
    confidence  = float(1.0 - (entropy / max_entropy))

    return float(soft_err), off_line_prob, pred_class, confidence

# pid controller 
class PIDController:
   MIN_GAIN_SCALE = 0.2   # floor — never scale gains below 20% of base

   def __init__(self, Kp, Ki, Kd, dt, integral_limit=1.0):
        self.Kp_base = Kp
        self.Ki_base = Ki
        self.Kd_base = Kd
        self.dt = dt
        self.integral_limit = integral_limit
        self.reset()

   def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

   def compute(self, error, confidence=1.0):
        # Scale gains by confidence, floored at MIN_GAIN_SCALE
        scale = max(self.MIN_GAIN_SCALE, confidence)
        Kp = self.Kp_base * scale
        Ki = self.Ki_base * scale
        Kd = self.Kd_base * scale

        self.integral += error * self.dt
        self.integral  = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative     = (error - self.prev_error) / self.dt
        self.prev_error = error
        return Kp * error + Ki * self.integral + Kd * derivative

#metrics
METRICS_CSV   = "cnn_pid_adaptive_metrics.csv"
SUMMARY_FILE  = "cnn_pid_adaptive_summary.txt"
IMAGE_CENTRE  = 160   # centre column of 320px wide image

#quanser setup
setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True
frameRate, sampleRate = 60.0, 1/60.0
counterDown = 0
endFlag, forSpd, turnSpd = False, 0.0, 0.0
lineFollow = False

# Initialise image buffers — prevents probe sending uninitialised data
gray_sm = np.zeros((200, 320, 1), dtype=np.uint8)
binary  = np.zeros((50,  320, 1), dtype=np.uint8)

startTime = time.time()
def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017
pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=sampleRate, integral_limit=INTEGRAL_LIMIT)

#accumulate metrics
cte_list         = []   # cross-track error per frame (pixels from centre)
smoothness_list  = []   # |delta turnSpd| per frame
off_line_count   = 0    # frames where off_line was triggered
frames_processed = 0
prevTurnSpd      = 0.0

csv_file   = open(METRICS_CSV, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp_s', 'pred_class', 'soft_err', 'confidence',
                     'cte_px', 'for_spd', 'turn_spd', 'delta_turn_spd',
                     'off_line_prob', 'off_line'])

print("\n[CNN+PID] Ready. Space=arm, 7=toggle line follow, U=quit\n")
print(f"{'Time':>6}  {'Class':>9}  {'SoftErr':>8}  {'PID':>7}  "
      f"{'ForSpd':>7}  {'TurnSpd':>8}  {'OffLine':>8}")
print("-" * 65)

try:
    # Section B - Initialisation
    myQBot   = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam  = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    keyboard = Keyboard()
    vision   = QBPVision()
    probe    = Probe(ip=ipHost)
    probe.add_display(imageSize=[200, 320, 1], scaling=True,  scalingFactor=2, name='Raw Image')
    probe.add_display(imageSize=[50,  320, 1], scaling=False, scalingFactor=2, name='Binary Image')

    startTime = time.time()
    time.sleep(0.5)

    # Main loop
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # Keyboard driver
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm            = keyboard.k_space
                lineFollow     = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # Section C - Toggle line following
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            else:
                commands = np.array([forSpd, turnSpd], dtype=np.float64)

            # QBot hardware
            newHIL = myQBot.read_write_std(
                timestamp=time.time() - startTime,
                arm=arm,
                commands=commands
            )

            if newHIL:
                timeHIL    = time.time()
                newDownCam = downCam.read()

                if newDownCam:
                    counterDown      += 1
                    frames_processed += 1

                    # Section D - Image processing
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm     = cv2.resize(undistorted, (320, 200))

                    # Still compute binary for probe visualisation
                    binary = vision.subselect_and_threshold(gray_sm, 50, 100, 50, 255)

                    # ── CNN inference ──────────────────────────────────────────
                    soft_err, off_line_prob, pred_class, confidence = get_soft_error(gray_sm)

                    off_line = off_line_prob > OFF_LINE_THRESHOLD
                    if off_line:
                        off_line_count += 1

                    if off_line:
                        # Lost the line — slow creep, reset PID integrator
                        pid.reset()
                        forSpd  = OFF_LINE_SPEED
                        turnSpd = 0.0
                    else:
                        # ── Adaptive PID step ──────────────────────────────────
                        # confidence scales all gains — uncertain CNN → gentler
                        # corrections, robot slows slightly on ambiguous frames
                        pid_out = pid.compute(soft_err, confidence=confidence)
                        forSpd  = BASE_SPEED * (1.0 - 0.5 * abs(soft_err))
                        turnSpd = float(np.clip(pid_out, -1.0, 1.0))

                    # ── Metric 1: Cross-Track Error ────────────────────────────
                    # Use blob detection centroid for CTE — same method as PID
                    # baseline so the metric is directly comparable
                    col, row, area = vision.image_find_objects(binary, 8, 50, 3000)
                    if col is not None and col != -1:
                        cte = col - IMAGE_CENTRE
                        cte_list.append(cte)
                    else:
                        cte = None

                    # ── Metric 2: Control Smoothness ───────────────────────────
                    delta_turn = abs(turnSpd - prevTurnSpd) / sampleRate
                    smoothness_list.append(delta_turn)
                    prevTurnSpd = turnSpd

                    # ── Log to CSV ─────────────────────────────────────────────
                    csv_writer.writerow([
                        round(t, 3),
                        pred_class,
                        round(soft_err, 4),
                        round(confidence, 4),
                        round(cte, 2) if cte is not None else 'NaN',
                        round(forSpd, 4),
                        round(turnSpd, 4),
                        round(delta_turn, 4),
                        round(off_line_prob, 4),
                        1 if off_line else 0
                    ])

                    # Console log every 20 frames
                    if counterDown % 20 == 0:
                        print(f"{t:6.1f}  {pred_class:>9}  {soft_err:+8.3f}  "
                              f"conf:{confidence:.2f}  {turnSpd:+7.3f}  "
                              f"{forSpd:7.3f}  {'YES' if off_line else 'no':>8}")

                    # Probe display every 4 frames
                    if counterDown % 4 == 0:
                        probe.send(name='Raw Image',    imageData=gray_sm)
                        probe.send(name='Binary Image', imageData=binary)

                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('\nUser interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    #run summary
    csv_file.close()

    cte_arr       = np.array(cte_list) if cte_list else np.array([0.0])
    smooth_arr    = np.array(smoothness_list) if smoothness_list else np.array([0.0])
    off_line_rate = (off_line_count / frames_processed * 100) if frames_processed > 0 else 0.0

    summary = f
    print(summary)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved -> {SUMMARY_FILE}")

    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()