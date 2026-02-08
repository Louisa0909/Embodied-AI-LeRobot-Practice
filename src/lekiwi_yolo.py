#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LeKiwi Robot Control System - Vision & Teleoperation
Integrates YOLO-based person tracking with teleoperation capabilities.
"""

import time
import os
import numpy as np
import cv2
import torch

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Import YOLO dependencies
import model.detector
import utils.utils

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Network & Robot Settings
REMOTE_IP = "172.20.10.3"       # Raspberry Pi / Host IP
ROBOT_ID = "my_awesome_kiwi"
LEADER_PORT = "COM6"            # Port for Leader Arm
FPS = 15                        # Control loop frequency (Hz)
CAM_KEY = 'front'               # Camera key for detection

# YOLO Model Configuration
MODEL_DATA_PATH = 'data/coco.data'
WEIGHTS_PATH = 'modelzoo/coco2017-0.241078ap-model.pth'
TARGET_CATS = {"person"}
CONF_THR = 0.3                  # Confidence threshold
IOU_THR = 0.4                   # IOU threshold
AREA_STOP = 0.70                # Stop if bounding box area ratio exceeds this
X_LEFT, X_RIGHT = 0.30, 0.70    # Horizontal thresholds for turning

# Base Movement Parameters (Fixed Velocity)
V_FORWARD = 0.15                # Forward linear velocity (m/s)
W_ROT = 2.0                     # Rotational velocity (rad/s)
V_SEARCH = 0.0                  # Linear velocity during search
SEARCH_PAUSE_S = 0.5            # Pause duration after detection for stability

# ==============================================================================
# WAVE ACTION SEQUENCE
# ==============================================================================

WAVE_SEQUENCE = [
    # Position 1
    {
        "arm_shoulder_pan.pos.pos": 38.6088, "arm_shoulder_lift.pos.pos": -51.2463,
        "arm_elbow_flex.pos.pos": 41.0803, "arm_wrist_flex.pos.pos": -79.4473,
        "arm_wrist_roll.pos.pos": 90.1343, "arm_gripper.pos.pos": 0.2381,
        "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0
    },
    # Position 2
    {
        "arm_shoulder_pan.pos.pos": 38.8296, "arm_shoulder_lift.pos.pos": -51.1618,
        "arm_elbow_flex.pos.pos": 41.2619, "arm_wrist_flex.pos.pos": -27.7202,
        "arm_wrist_roll.pos.pos": 89.8413, "arm_gripper.pos.pos": 1.0317,
        "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0
    },
    # Position 3
    {
        "arm_shoulder_pan.pos.pos": 38.6088, "arm_shoulder_lift.pos.pos": -51.1618,
        "arm_elbow_flex.pos.pos": 41.9882, "arm_wrist_flex.pos.pos": -90.3282,
        "arm_wrist_roll.pos.pos": 90.0366, "arm_gripper.pos.pos": 0.4762,
        "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0
    },
    # Position 4
    {
        "arm_shoulder_pan.pos.pos": 38.6088, "arm_shoulder_lift.pos.pos": -51.1618,
        "arm_elbow_flex.pos.pos": 42.3513, "arm_wrist_flex.pos.pos": -32.5561,
        "arm_wrist_roll.pos.pos": 89.9389, "arm_gripper.pos.pos": 0.9524,
        "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0
    },
    # Position 5
    {
        "arm_shoulder_pan.pos.pos": 38.6824, "arm_shoulder_lift.pos.pos": -99.5775,
        "arm_elbow_flex.pos.pos": 99.5461, "arm_wrist_flex.pos.pos": -100.0000,
        "arm_wrist_roll.pos.pos": 90.1343, "arm_gripper.pos.pos": 0.3968,
        "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
]

# Wave Control Parameters
WAVE_COUNT = 1                  # Number of wave cycles
WAVE_TRANSITION_TIME = 1.0      # Transition time between actions
WAVE_COOLDOWN = 10.0            # Cooldown to prevent repetitive triggering

# Global State Variables
is_waving = False
last_wave_time = 0
current_wave_step = 0
current_wave_cycle = 0
wave_start_time = 0

# ==============================================================================
# INITIALIZATION
# ==============================================================================

# 1. Initialize YOLO Detector
assert os.path.exists(WEIGHTS_PATH), "Error: Model weights not found."
cfg = utils.utils.load_datafile(MODEL_DATA_PATH)
device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))

detector = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
detector.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
detector.eval()
if device.type in ("mps", "cuda"):
    detector = detector.half()

LABEL_NAMES = [line.strip() for line in open(cfg["names"], 'r').readlines()]

# 2. Initialize Robot & Teleoperators
try:
    robot_cfg = LeKiwiClientConfig(remote_ip=REMOTE_IP, id=ROBOT_ID)
    leader_cfg = SO101LeaderConfig(port=LEADER_PORT, id="my_awesome_leader_arm")
    keyboard_cfg = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_cfg)
    leader = SO101Leader(leader_cfg)
    keyboard = KeyboardTeleop(keyboard_cfg)

    robot.connect()
    leader.connect()
    keyboard.connect()

    init_rerun(session_name="lekiwi_teleop")

    if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, leader arm, or keyboard is not connected!")
    print("✅ System initialized successfully.")

except Exception as e:
    print(f"❌ Initialization failed: {e}")
    exit(1)


# ==============================================================================
# CONTROL LOGIC FUNCTIONS
# ==============================================================================

def start_wave_action():
    """Triggers the waving sequence."""
    global is_waving, current_wave_step, current_wave_cycle, wave_start_time, last_wave_time
    is_waving = True
    current_wave_step = 0
    current_wave_cycle = 0
    wave_start_time = time.time()
    last_wave_time = time.time()
    print("🎯 Person detected! Starting wave action.")

def update_wave_action():
    """Updates the waving state and returns the next arm action."""
    global is_waving, current_wave_step, current_wave_cycle
    
    if not is_waving or current_wave_cycle >= WAVE_COUNT:
        is_waving = False
        return None
    
    wave_action = WAVE_SEQUENCE[current_wave_step].copy()
    current_wave_step += 1
    
    if current_wave_step >= len(WAVE_SEQUENCE):
        current_wave_step = 0
        current_wave_cycle += 1
        print(f"✅ Wave cycle {current_wave_cycle} completed.")
    
    return wave_action

def should_start_wave(box_area, current_time):
    """Checks conditions to trigger the wave action (area threshold & cooldown)."""
    global is_waving, last_wave_time
    
    if is_waving: return False
    if current_time - last_wave_time < WAVE_COOLDOWN: return False
    return box_area > AREA_STOP

# Base Movement Helpers
USE_THETA_IN_RAD = True
def base_cmd_forward():
    return {"x.vel": V_FORWARD, "y.vel": 0.0, "theta.vel": 0.0}

def base_cmd_left():
    wz = W_ROT if USE_THETA_IN_RAD else np.rad2deg(W_ROT)
    return {"x.vel": V_SEARCH, "y.vel": 0.0, "theta.vel": +wz}

def base_cmd_right():
    wz = W_ROT if USE_THETA_IN_RAD else np.rad2deg(W_ROT)
    return {"x.vel": V_SEARCH, "y.vel": 0.0, "theta.vel": -wz}

def base_cmd_stop():
    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

last_turn = "LEFT"
was_turning = False

def pick_frame_from_observation(obs: dict, prefer_key=None):
    """Extracts a valid image frame from robot observation."""
    if prefer_key and prefer_key in obs:
        frame = obs[prefer_key]
        if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] in (3,4):
            if frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame, prefer_key
    
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] in (3,4):
            frame = v
            if frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame, k
    return None, None

def yolo_base_action_from_frame(frame: np.ndarray):
    """Processes frame with YOLO to generate base velocity commands."""
    global last_turn, was_turning, is_waving
    
    h, w, _ = frame.shape
    current_time = time.time()
    
    # Preprocessing
    inp = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    inp = inp.reshape(1, cfg["height"], cfg["width"], 3).transpose(0,3,1,2)
    tens = torch.from_numpy(inp).to(device).float()/255.0
    
    param0 = next(detector.parameters())
    if param0.dtype == torch.float16: tens = tens.half()
    
    with torch.no_grad():
        preds = detector(tens)
    
    output = utils.utils.handel_preds(preds, cfg, device)
    boxes = utils.utils.non_max_suppression(output, conf_thres=CONF_THR, iou_thres=IOU_THR)
    
    viz = []
    action = None

    # No detections
    if len(boxes[0]) == 0:
        action = base_cmd_left() if last_turn == "LEFT" else base_cmd_right()
        was_turning = True
        return action, viz

    # Find best person detection
    best = None
    best_score = -1
    for b in boxes[0]:
        b = b.tolist()
        score = b[4]
        cls = LABEL_NAMES[int(b[5])]
        if cls in TARGET_CATS and score > best_score:
            best = b; best_score = score

    if best is None:
        action = base_cmd_stop()
        return action, viz

    # Process detection
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]
    x1, y1 = int(best[0]*scale_w), int(best[1]*scale_h)
    x2, y2 = int(best[2]*scale_w), int(best[3]*scale_h)
    viz.append((x1, y1, x2, y2, best_score, "person"))

    x_center = (best[0] + best[2]) / 2 / cfg["width"]
    box_area = (best[2] - best[0]) * (best[3] - best[1]) / (cfg["height"] * cfg["width"])

    # Check for wave trigger
    if should_start_wave(box_area, current_time):
        start_wave_action()
    
    # Movement logic
    if box_area > AREA_STOP:
        action = base_cmd_stop()
    else:
        if x_center < X_LEFT:
            action = base_cmd_left(); last_turn = "LEFT"; was_turning = True
        elif x_center > X_RIGHT:
            action = base_cmd_right(); last_turn = "RIGHT"; was_turning = True
        else:
            if was_turning:
                action = base_cmd_stop()
                time.sleep(SEARCH_PAUSE_S)
                was_turning = False
            action = base_cmd_forward()

    return action, viz

# ==============================================================================
# MAIN LOOP
# ==============================================================================

print("🚀 Teleop running... (Arm: Leader, Base: YOLO, Auto-Wave: Enabled)")
last_wave_update_time = 0
wave_update_interval = 1.0

try:
    while True:
        t0 = time.perf_counter()

        # 1. Get Observation
        obs = robot.get_observation()

        # 2. Get Camera Frame
        frame, used_key = pick_frame_from_observation(obs, CAM_KEY)
        if CAM_KEY is None and used_key is not None:
            CAM_KEY = used_key

        # 3. Determine Arm Action
        current_time = time.time()
        wave_arm_action = None
        
        if is_waving:
            if current_time - last_wave_update_time >= wave_update_interval:
                wave_arm_action = update_wave_action()
                last_wave_update_time = current_time
            else:
                if current_wave_step > 0:
                    wave_arm_action = WAVE_SEQUENCE[current_wave_step - 1].copy()
        else:
            arm_action = leader.get_action()
            wave_arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

        # 4. Determine Base Action (YOLO)
        if frame is not None:
            base_action, viz_boxes = yolo_base_action_from_frame(frame)
        else:
            base_action, viz_boxes = base_cmd_stop(), []

        # 5. Combine & Send Actions
        action = {**wave_arm_action, **base_action} if wave_arm_action else base_action
        robot.send_action(action)

        # 6. Visualization
        if frame is not None:
            for (x1, y1, x2, y2, score, cls) in viz_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f'{cls} {score:.2f}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            status_text = f'cam:{CAM_KEY} | '
            status_text += f'Waving {current_wave_cycle+1}/{WAVE_COUNT}' if is_waving else 'Tracking'
            cv2.putText(frame, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Teleop (Arm + YOLO Base + Wave)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        # 7. Logging (Optional)
        try:
            log_rerun_data(obs, action)
        except Exception:
            pass

        # 8. Loop Timing
        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

except KeyboardInterrupt:
    print("\n🛑 Teleop loop stopped by user.")
finally:
    cv2.destroyAllWindows()