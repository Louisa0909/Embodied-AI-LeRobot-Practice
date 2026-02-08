#!python
# -*- coding: utf-8 -*-

"""
Continuous Voice Control System for LeKiwi Robot
Features asynchronous voice command processing and continuous motion execution.
"""

import sys
import os
import time
import threading
import queue
import numpy as np
import cv2
from typing import Dict, Tuple, Optional

# Add lerobot module path
lerobot_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
if lerobot_src_path not in sys.path:
    sys.path.insert(0, lerobot_src_path)

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import busy_wait

# ==============================================================================
# CONFIGURATION
# ==============================================================================

REMOTE_IP = "172.20.10.3"       # Raspberry Pi IP
ROBOT_ID = "my_awesome_kiwi"
FPS = 30                        # Control loop frequency
VOICE_TIMEOUT = 5.0             # Voice recognition timeout

# Voice Command Mapping (Chinese keywords -> Action vectors)
# Note: Recognizes Chinese commands but internal logic uses English keys
VOICE_COMMANDS = {
    # Forward
    "前进": {"x.vel": 0.15, "y.vel": 0.0, "theta.vel": 0.0},
    "向前": {"x.vel": 0.15, "y.vel": 0.0, "theta.vel": 0.0},
    "往前": {"x.vel": 0.15, "y.vel": 0.0, "theta.vel": 0.0},
    "走": {"x.vel": 0.15, "y.vel": 0.0, "theta.vel": 0.0},
    
    # Backward
    "后退": {"x.vel": -0.15, "y.vel": 0.0, "theta.vel": 0.0},
    "向后": {"x.vel": -0.15, "y.vel": 0.0, "theta.vel": 0.0},
    "往后": {"x.vel": -0.15, "y.vel": 0.0, "theta.vel": 0.0},
    "退": {"x.vel": -0.15, "y.vel": 0.0, "theta.vel": 0.0},
    
    # Left Turn
    "左转": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 20},
    "向左": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 20},
    "往左": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 20},
    
    # Right Turn
    "右转": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -20},
    "向右": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -20},
    "往右": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -20},
    
    # Compound Movements
    "左前方": {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 10},
    "右前方": {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": -10},
    "左后退": {"x.vel": -0.1, "y.vel": 0.0, "theta.vel": 10},
    "右后退": {"x.vel": -0.1, "y.vel": 0.0, "theta.vel": -10},
    
    # Stop
    "停止": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
    "停下": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
    "停": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
    "停一下": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
    "别动了": {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
    
    # System Commands
    "退出": "exit",
    "结束": "exit",
    "关闭": "exit",
    "关机": "exit",
}

# ==============================================================================
# ROBOT CONTROLLER
# ==============================================================================

class ContinuousRobotController:
    def __init__(self):
        self.config = LeKiwiClientConfig(remote_ip=REMOTE_IP, id=ROBOT_ID)
        self.robot = LeKiwiClient(self.config)
        self.arm_positions = {}
        self.is_connected = False
        self.current_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self.current_action_name = "STOP"
    
    def connect(self):
        """Establish connection to the robot."""
        print("🔌 Connecting to robot...")
        try:
            self.robot.connect()
            self.is_connected = self.robot.is_connected
            
            if self.is_connected:
                print("✅ Robot connected successfully.")
                obs = self.robot.get_observation()
                self._init_arm_positions(obs)
                return True
            else:
                print("❌ Connection failed.")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def _init_arm_positions(self, observation: dict):
        """Initialize arm position holding."""
        arm_keys = [
            'arm_shoulder_pan.pos', 'arm_shoulder_lift.pos', 'arm_elbow_flex.pos',
            'arm_wrist_flex.pos', 'arm_wrist_roll.pos', 'arm_gripper.pos'
        ]
        for key in arm_keys:
            self.arm_positions[key] = float(observation.get(key, 0.0))
        print(f"📊 Arm positions initialized.")
    
    def update_action(self, action_name: str, action: dict):
        """Update the current continuous action."""
        self.current_action = action
        self.current_action_name = action_name
        print(f"🔄 Action Updated: {action_name}")
    
    def get_current_action(self):
        return self.current_action, self.current_action_name
    
    def send_continuous_action(self):
        """Send current action frame to robot."""
        if not self.is_connected: return False
        try:
            full_action = {**self.arm_positions, **self.current_action}
            self.robot.send_action(full_action)
            return True
        except Exception as e:
            print(f"❌ Action send failed: {e}")
            return False
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        self.current_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self.current_action_name = "EMERGENCY STOP"
        print("🛑 EMERGENCY STOP TRIGGERED!")
        self.send_continuous_action()
    
    def disconnect(self):
        """Safely disconnect from robot."""
        if self.is_connected:
            self.emergency_stop()
            time.sleep(0.1)
            self.robot.disconnect()
            self.is_connected = False
            print("🔌 Robot disconnected.")

# ==============================================================================
# VOICE RECOGNIZER
# ==============================================================================

class ContinuousVoiceRecognizer:
    def __init__(self, commands: Dict):
        self.commands = commands
        self.command_queue = queue.Queue()
        self.is_listening = True
        self.recognizer = None
        self.microphone = None
        self._init_speech_recognition()
    
    def _init_speech_recognition(self):
        """Initialize SpeechRecognition library."""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✅ Voice recognition initialized.")
            
            try:
                import zhconv
                self.zhconv = zhconv
                print("✅ Traditional/Simplified Chinese conversion enabled.")
            except ImportError:
                self.zhconv = None
                print("⚠️ zhconv not found, conversion disabled.")
        except ImportError:
            print("❌ 'speech_recognition' library missing. Install via pip.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Init failed: {e}")
            sys.exit(1)
    
    def start_listening(self):
        listen_thread = threading.Thread(target=self._listen_loop, name="VoiceListener")
        listen_thread.daemon = True
        listen_thread.start()
        print("🎤 Voice listening thread started.")
    
    def stop_listening(self):
        self.is_listening = False
    
    def _listen_loop(self):
        import speech_recognition as sr
        print("\n" + "="*60)
        print("VOICE CONTROL MODE: CONTINUOUS")
        print("="*60)
        print("📢 Supported Commands (Chinese):")
        print("  Forward/Go, Backward, Left, Right")
        print("  Front-Left, Front-Right")
        print("  Stop, Exit")
        print("="*60 + "\n")
        print("👂 Listening...")
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=VOICE_TIMEOUT, phrase_time_limit=3)
                
                try:
                    text = self.recognizer.recognize_google(audio, language='zh-CN')
                    print(f"🗣️ Heard: {text}")
                    if self.zhconv: text = self.zhconv.convert(text, 'zh-cn')
                    self._process_command(text)
                except sr.UnknownValueError:
                    print("🤔 Could not understand audio.")
                except sr.RequestError as e:
                    print(f"🌐 API Error: {e}")
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(1)
    
    def _process_command(self, text: str):
        if any(cmd in text for cmd in ["退出", "结束", "关闭", "关机"]):
            print("🛑 Exit command received.")
            self.command_queue.put(("exit", None))
            return
        
        matched = False
        for cmd_key, cmd_value in self.commands.items():
            if cmd_key in text:
                if cmd_value == "exit": continue
                self.command_queue.put((cmd_key, cmd_value))
                print(f"✅ Executing: {cmd_key}")
                matched = True
                break
        if not matched: print("❓ Command not recognized.")

# ==============================================================================
# STATUS DISPLAY
# ==============================================================================

class StatusDisplay:
    def __init__(self, robot_controller: ContinuousRobotController):
        self.robot = robot_controller
        self.is_running = True
        self.last_image = None
        self.image_lock = threading.Lock()
    
    def start_display(self):
        display_thread = threading.Thread(target=self._display_loop, name="StatusDisplay")
        display_thread.daemon = True
        display_thread.start()
        print("📊 Status display started.")
    
    def stop_display(self):
        self.is_running = False
        cv2.destroyAllWindows()
    
    def update_camera_image(self):
        try:
            if not self.robot.is_connected: return
            obs = self.robot.robot.get_observation()
            
            for key in ['front', 'wrist', 'camera']:
                if key in obs and isinstance(obs[key], np.ndarray):
                    frame = obs[key]
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    with self.image_lock: self.last_image = frame
                    return
        except Exception: pass
    
    def _display_loop(self):
        while self.is_running:
            try:
                display_image = None
                with self.image_lock:
                    if self.last_image is not None: display_image = self.last_image.copy()
                
                current_action, action_name = self.robot.get_current_action()
                
                if display_image is not None:
                    display_image = cv2.resize(display_image, (800, 600))
                    self._add_status_overlay(display_image, action_name, current_action)
                    cv2.imshow("Voice Control Car", display_image)
                else:
                    placeholder = self._create_status_placeholder(action_name, current_action)
                    cv2.imshow("Voice Control Car", placeholder)
                
                key = cv2.waitKey(30) & 0xFF
                if key in [ord('q'), 27]:
                    self.is_running = False
                    break
                elif key in [ord('s'), ord('S')]:
                    self.robot.emergency_stop()
            except Exception: time.sleep(0.1)

    def _add_status_overlay(self, image: np.ndarray, action_name: str, action: dict):
        h, w = image.shape[:2]
        status_bar = np.zeros((120, w, 3), dtype=np.uint8)
        image[0:120, 0:w] = cv2.addWeighted(image[0:120, 0:w], 0.3, status_bar, 0.7, 0)
        
        cv2.putText(image, "Voice Control System", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(image, f"State: {action_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(image, "Mode: Continuous | S: Stop | Q: Quit", (w-600, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)

    def _create_status_placeholder(self, action_name: str, action: dict):
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(image, "Voice Control System", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(image, f"Current State: {action_name}", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(image, "Waiting for Camera...", (280, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        return image

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Starting Voice Control System...")
    
    try:
        import speech_recognition
    except ImportError:
        print("❌ 'speech_recognition' not installed.")
        sys.exit(1)
        
    system = ContinuousVoiceCarSystem()
    try:
        system.start()
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
    finally:
        print("\n👋 System Shutdown.")