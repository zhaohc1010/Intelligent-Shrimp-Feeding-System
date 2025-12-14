# Intelligent Feeding Tray System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)](https://www.python.org/)

## 📖 Introduction

The **Intelligent Feeding Tray** is a multimodal AI system designed to bridge the gap between natural language interaction and industrial hardware control. This project demonstrates a prototype where a user can control a mechanical feeding tray using voice commands and query the system about the visual state of the tray using a Vision Language Model (VLM).



### ✨ Key Features
* **🗣️ Voice Command Control**: Parses natural language instructions (e.g., "Move up for 3 seconds") to control hardware relays and motors.
* **👁️ AI Vision Analysis**: Integrates a Vision Language Model to analyze real-time video frames and answer user questions (e.g., "Is the tray empty?").
* **🤖 Hardware Abstraction**: Includes a driver module for LCUS USB Relays, supporting precise timing and motor direction control.
* **🔊 Audio Feedback Loop**: Features real-time Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) for a seamless conversational experience.

---

## 🛠️ Hardware & Software Requirements

### Hardware Setup
This system is designed to run on a PC (Host) connected to the following peripherals:
1.  **Controller**: Windows PC (Recommended for COM port compatibility) or Linux device.
2.  **Actuators**:
    * USB Relay Module (LCUS-1/2/4 type).
    * DC Motor (Connected via Relay NO/NC ports).
3.  **Sensors**:
    * USB Webcam.
    * Microphone.
    * Speaker.

### Software Dependencies
* Python 3.8 or higher.
* **FFmpeg**: Required for audio processing (via `pydub`). Ensure FFmpeg is installed and added to your system PATH.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/intelligent_feeding_tray.git](https://github.com/YourUsername/intelligent_feeding_tray.git)
cd intelligent_feeding_tray