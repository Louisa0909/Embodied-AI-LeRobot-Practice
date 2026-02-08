# Embodied-LeKiwi-Intelligence: A Robotic System Based on LeRobot Open-Source Code

An Embodied AI System with Multimodal Control (Visual/Voice/Teleop) based on LeRobot

## 🌟 Project Overview
This project aims to build a flexible robotic control system. By integrating YOLO-based visual object detection, asynchronous voice recognition, and leader-follower teleoperation, it enables multimodal control and feedback for the LeKiwi robot in complex task scenarios.

- **Technical Approach**: The system adopts a modular architecture that decouples perception (Vision/Voice), decision-making (Logic Integration), and execution (LeRobot SDK), addressing engineering challenges such as multi-source command flow conflicts and hardware response latency.

## 🎬 Demo

1.  **Visual Tracking & Autonomous Obstacle Avoidance (YOLO-based)**
2.  **Continuous Motion Control via Voice (Voice Control)**
3.  **Coordinated Leader-Follower Arm Teleoperation**

![Real-time object detection and base following based on YOLO](assets/visual_tracking.mp4)
![Real-time object detection and base following based on YOLO](assets/waving.mp4)
![Voice command parsing and continuous closed-loop motion](assets/voice.mp4)
![SO100 master arm synchronously controlling the LeKiwi manipulator](assets/teleoperate.mp4)

## 🛠️ Architecture

**Core Control Logic**:
- **`lekiwi_yolo.py`**: Integrates the visual detection closed loop. It parses camera frames in real-time via `yolo_base_action_from_frame` and issues complex actions combined with leader-follower logic.
- **`voice.py`**: Implements non-blocking voice control. Uses `threading` for asynchronous command listening, supporting composite motions (e.g., "forward + turn") and emergency stop functionality.
- **`teleoperate.py`**: Low-latency leader-follower control. Supports state capture and command transmission at 30 FPS with low latency.
- **Motion Smoothing**: Leverages the built-in Action Chunking mechanism of the LeRobot framework to mitigate motion stutter caused by inference latency.

## 🚀 Engineering Challenges & Solutions

The project log details the process of making the "black-box system" transparent:

- **Hardware Conflict Optimization**: Addressed COM port occupancy and robot ID matching issues by implementing a unified configuration interface (Config Class) to decouple software from hardware.
- **Signal Flow Resilience**: Designed a continuous motion mode within the voice control module to ensure the robot maintains its intended motion state during wireless network fluctuations until a new command is received or a heartbeat timeout occurs.
- **Visual Feedback Loop**: Resolved target loss issues caused by inference latency through OpenCV visualization debugging, achieving stable target following.

## 📂 Repository Structure

```
.
├── src/
│   ├── lekiwi_yolo.py         # Core visual navigation & leader-follower coordination
│   ├── voice.py               # Asynchronous voice control system
│   └── teleoperate.py         # Teleoperation command stream processing
├── docs/
│   └── PROJECT_LOG.md         # Detailed development log & technical analysis
│   └── PROJECT_LOG_CN.md      # Chinese version
├── assets/
│   └── demo_gifs/             # Demonstration GIFs
└── requirements.txt           # Environment dependencies (lerobot, torch, opencv, etc.)
```

## Limitations & Future Work
While this project successfully achieves multimodal closed-loop control, the current technical approach, based on rule-driven logic and module stacking, has room for improvement in generalization capabilities.

- **Towards End-to-End Control**:
    The current system relies on hard-coded rule-based logic using bounding box coordinates from YOLO.
    **Future Direction**: Collect human teleoperation data to train Diffusion Policy or ACT policy networks, enabling **end-to-end** control directly from "image pixels" to "joint actions" for handling more delicate manipulation tasks.

- **Smoother Motion Generation**:
    Currently utilizes basic interpolation and Action Chunking. Future work could introduce more advanced trajectory smoothing algorithms to reduce jerky stop-and-start motions of the robotic arm.

## 📖 References & Acknowledgments

- **Hugging Face LeRobot**: Provides the core low-level drivers and data structure support.
- This project serves as the final project for the *Introduction to Embodied AI* course. Thanks to the course team for providing the hardware platform and foundational template support.
