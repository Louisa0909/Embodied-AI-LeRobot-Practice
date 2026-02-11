<style>
body {
    font-family: "仿宋", "FangSong";
    font-size: 12pt;
}
</style>
# LeKiwi Embodied AI System: 多模态具身智能机器人工程实践报告

**Author**: Liu Xiaoyu  
**Date**: 2024  
**Tech Stack**: Python, PyTorch, YOLOv8, LeRobot SDK, OpenCV, SpeechRecognition, Threading

---

## 1. 项目背景与工程目标 (Project Overview)

本项目旨在基于 **Hugging Face LeRobot** 开源框架，构建一个具备**视觉感知、语音交互与遥操作**能力的具身智能系统。项目并未止步于现有框架的简单调用，而是通过**逆向拆解与模块化重构**，解决了开源代码在特定硬件环境下的适配问题，并实现了多模态指令的闭环控制。

**核心工程目标**：
1.  **全链路信号打通**：实现从传感器（摄像头/麦克风）到执行器（六轴机械臂/底盘）的毫秒级信号闭环。
2.  **系统鲁棒性构建**：在非实时操作系统（Linux/Windows）上解决多进程通信延迟与硬件资源冲突，确保系统运行的稳定性。
3.  **混合架构探索**：验证“规则驱动（FSM）”与“数据驱动（VLA Model）”在边缘侧设备上的落地可行性与性能边界。

---

```
[外界环境]
     |
[感知层 Perception]
├── 视觉感知 (YOLO) ──> 人体检测框 / 距离估计
├── 语音感知 (SpeechRecognition) ──> 文本指令
└── 本体感知 (Robot State) ──> 关节角度 / 位置
     |
[决策与状态管理层 Cognition]
├── 视觉追踪策略：基于检测框位置生成底盘速度指令
├── 语音指令映射：将自然语言映射为标准化动作指令
├── 行为状态机：管理“追踪模式”与“挥手模式”的切换与冷却
└── 多模态仲裁（初步尝试）：同时处理语音与视觉指令
     |
[控制层 Control]
├── 运动控制：将高层指令(x.vel, θ.vel)转换为底层电机控制量
├── 动作序列执行器：按预设轨迹控制机械臂（如挥手）
└── 安全守护：紧急停止、超时检测
     |
[执行层 Actuation]
├── 底盘电机
└── 六轴机械臂
     |
[反馈与调试层 Feedback]
├── 实时图像显示（含视觉分析结果叠加）
├── 状态信息面板（当前动作、速度、模式）
└── 日志与错误流（用于问题追踪）
```

### 架构特点：

*   **标准化接口**：各层之间通过统一的 observation 与 action 字典进行通信，降低了模块耦合度。
*   **实时控制流 (Real-time Control Loop)**：主循环严格遵循 30Hz 的控制频率，采用时间片预算（Time Budgeting）机制，防止感知计算阻塞控制信号。

## 2.关键技术实现 (Key Implementations)

### Phase 1: 基础设施与硬件抽象 (Infrastructure & HAL)
针对异构硬件（So-100 机械臂 + LeKiwi 底盘）的联调，主要解决了以下工程痛点：

*   **环境隔离**：解决了 ROS、Conda 与 LeRobot 专有依赖库的冲突，构建了可复现的开发环境。
*   **分布式通信**：调试并打通了主从臂（Leader-Follower）在局域网内的低延迟通信链路，解决了丢包导致的动作卡顿问题。
*   **硬件抽象层 (HAL)**：通过配置文件对不同厂商的舵机协议进行统一映射，修正了关节限位与方向定义，确保物理安全。

### Phase 2: 视觉闭环与状态机设计 (Vision Loop & FSM)
为了赋予机器人自主跟随能力，集成了 YOLO 模型并引入有限状态机（Finite State Machine）管理行为逻辑。

*   **视觉伺服**：将 YOLO 输出的 Bounding Box 坐标实时映射为底盘的 (x_vel, theta_vel)，实现了基于位置误差的 P 控制。
*   **状态机去噪**：
    *   引入 **TRACKING (追踪)** 与 **WAVING (交互)** 双状态。
    *   设计了**冷却机制** 与**滞后阈值**，有效滤除了检测框抖动导致的动作误触发，提升了系统稳定性。
*   **动作平滑**：利用 LeRobot 的 **Action Chunking** 策略，平滑了离散推理带来的控制信号突变。

### Phase 3: 异步并发语音控制 (Async Voice Control)
针对语音识别 API 的高延迟特性（~1s），设计了生产者-消费者多线程模型。

*   **多线程并发**：
    *   **VoiceListener Thread**：阻塞式监听麦克风，将识别到的文本指令推入 Queue。
    *   **Control Thread**：以 30Hz 高频读取队列并刷新电机指令，确保在语音处理期间机器人不“死机”。
*   **指令映射与安全**：构建了支持模糊匹配的指令表（如“前进”、“往前走”映射为同一 Action），并实现了最高优先级的软急停 (Software E-Stop)功能。

---

## 3. 端到端尝试：VLA 大模型落地评估 (Experimental: SmolVLA)
项目后期尝试部署 Hugging Face 的 SmolVLA (Vision-Language-Action) 模型，探索端到端控制方案。

**实现方案**：将实时图像与自然语言指令（如“Pick up the red block”）作为输入，直接输出 6-DoF 关节动作序列。

**工程评估 (Evaluation)**：
*   **优势**：具备极强的泛化能力，无需针对特定物体编写规则代码。
*   **瓶颈**：在边缘设备上推理延迟较高（>500ms），难以满足实时避障需求；且模型是一个“黑盒”，在动作失败时难以进行逻辑调试,最终效果不佳。
*   **结论**：现阶段传统算法负责底层执行是更稳健的落地路径。

---

## 4.核心代码片段

A. 行为状态机逻辑
```python
# Implemented a hysteresis comparator to prevent rapid state toggling
def should_start_wave(box_area, current_time):
    global is_waving, last_wave_time
    
    # Condition 1: System is idle (not already performing an action)
    if is_waving: return False
    
    # Condition 2: Cooldown period has passed (Temporal filtering)
    if current_time - last_wave_time < WAVE_COOLDOWN: return False
    
    # Condition 3: Visual signal strength exceeds threshold
    return box_area > AREA_STOP
```

B. 异步语音指令处理
```python
class ContinuousVoiceRecognizer:
    """
    Uses threading to decouple blocking audio I/O from high-frequency robot control.
    """
    def _listen_loop(self):
        while self.is_running:
            # Blocking call (runs in separate thread)
            audio = self.recognizer.listen(source)
            text = self.recognizer.recognize_google(audio)
            
            # Atomic operation: put command into thread-safe queue
            self.command_queue.put(self._parse_intent(text))
```

## 5.局限性与未来展望 (Limitations & Future Work)
尽管本项目成功实现了多模态的闭环控制，但目前的技术路线仍存在以下优化空间：

- **从模块化到端到端 (Towards End-to-End)**
当前系统依赖 Rule-based 逻辑。未来计划采集人类遥操作数据，训练 Diffusion Policy 或 ACT 策略网络，实现从“图像像素”直接到“关节动作”的端到端控制。

- **增强抗干扰能力**
目前的语音模块在嘈杂环境下识别率下降。计划引入本地部署的 Whisper Tiny 模型替代云端 API，降低延迟并提升抗噪性。

- **Sim2Real 迁移**
引入 Isaac Gym 仿真环境，在虚拟环境中预训练导航策略，再迁移至实体机器人，降低硬件损耗风险。

## 7. 参考资料 (References)
*   LeRobot Framework: https://github.com/huggingface/lerobot
*   YOLOv8: Ultralytics Real-time Object Detection.
*   Course: Embodied AI Introduction (Final Project).